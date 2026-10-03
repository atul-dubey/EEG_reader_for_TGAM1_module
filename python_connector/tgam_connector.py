"""TGAM1/MindWave acquisition and bounded, asynchronous inference windows."""
from __future__ import annotations

import importlib
import math
import multiprocessing
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

BAND_NAMES = ("delta", "theta", "low_alpha", "high_alpha", "low_beta", "high_beta", "low_gamma", "mid_gamma")


def _port_probe(port, baudrate, observe_seconds, pipe):
    """Spawned process owns the port; the parent can stop a hung Bluetooth driver."""
    result = {"port": port, "raw_samples": 0, "band_updates": 0,
              "poor_signal": None, "status": "no EEG data"}
    try:
        import serial
        parser = ThinkGearParser()
        with serial.Serial(port, baudrate, timeout=.1, write_timeout=1) as connection:
            deadline = time.monotonic() + observe_seconds
            while time.monotonic() < deadline:
                data = connection.read(max(1, min(connection.in_waiting, 4096)))
                for code, value in parser.feed(data):
                    if code == 0x80 and len(value) == 2:
                        result["raw_samples"] += 1
                    elif code == 0x83 and len(value) == 24:
                        result["band_updates"] += 1
                    elif code == 2 and len(value) == 1:
                        result["poor_signal"] = value[0]
        result["bad_checksums"] = parser.bad_checksums
        if result["raw_samples"] >= 3 or result["band_updates"]:
            result["status"] = "EEG detected"
    except Exception as exc:
        result["status"] = "unavailable"
        result["error"] = str(exc)
    try:
        pipe.send(result)
    finally:
        pipe.close()


def _probe_one(port, baudrate, observe_seconds, timeout, cancel_event):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_port_probe, args=(port, baudrate, observe_seconds, sender), daemon=True)
    result = {"port": port, "raw_samples": 0, "band_updates": 0, "poor_signal": None, "status": "timeout"}
    started = False
    try:
        process.start()
        started = True
        sender.close()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                result["status"] = "cancelled"
                break
            if receiver.poll(.05):
                try:
                    result = receiver.recv()
                except EOFError:
                    result["status"] = "probe stopped"
                break
            if not process.is_alive():
                result["status"] = "probe stopped"
                break
    finally:
        if started:
            process.join(timeout=.1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            if process.is_alive():
                raise TimeoutError("Port probe could not release " + port)
            process.close()
        receiver.close()
        sender.close()
    return result


class ThinkGearParser:
    def __init__(self):
        self.buffer = bytearray()
        self.packets = self.bad_checksums = self.malformed = 0

    def feed(self, chunk: bytes):
        self.buffer.extend(chunk)
        rows = []
        while len(self.buffer) >= 3:
            if self.buffer[:2] != b"\xaa\xaa" or self.buffer[2] > 169:
                del self.buffer[0]
                continue
            length = self.buffer[2]
            if len(self.buffer) < length + 4:
                break
            payload = bytes(self.buffer[3:3 + length])
            if (~sum(payload) & 255) != self.buffer[3 + length]:
                self.bad_checksums += 1
                del self.buffer[0]
                continue
            del self.buffer[:length + 4]
            packet_rows = []
            p = 0
            try:
                while p < length:
                    extended = 0
                    while p < length and payload[p] == 0x55:
                        extended += 1
                        p += 1
                    code = payload[p]
                    p += 1
                    n = 1
                    if code >= 0x80:
                        n = payload[p]
                        p += 1
                    if p + n > length:
                        raise ValueError("Truncated row")
                    value = payload[p:p + n]
                    p += n
                    if not extended:
                        packet_rows.append((code, value))
            except (IndexError, ValueError):
                self.malformed += 1
                continue
            self.packets += 1
            rows.extend(packet_rows)
        return rows


@dataclass(frozen=True)
class EEGWindow:
    """Raw counts, not volts; timestamps are host receipt times, not device time."""
    samples: tuple[int, ...]
    sample_rate: int
    end_sample_index: int
    received_utc: str
    bands: tuple[int, ...] | None
    bands_age_seconds: float | None
    poor_signal: int | None
    attention: int | None
    meditation: int | None
    simulated: bool


class TGAMConnector:
    def __init__(self, port=None, *, baudrate=57600, demo=False,
                 window_seconds=2.0, stride_seconds=0.5,
                 on_sample=None, on_bands=None, on_error=None):
        if not math.isfinite(window_seconds) or not math.isfinite(stride_seconds):
            raise ValueError("Window and stride must be finite")
        self.window_size = round(window_seconds * 512)
        self.stride = round(stride_seconds * 512)
        if self.window_size < 1 or self.stride < 1 or self.stride > self.window_size:
            raise ValueError("Require 0 < stride <= window, at least one sample")
        if not demo and not port:
            raise ValueError("A serial port is required outside demo mode")
        self.port, self.baudrate, self.demo = port, baudrate, demo
        self.on_sample, self.on_bands, self.on_error = on_sample, on_bands, on_error
        self.windows = queue.Queue(maxsize=1)
        self.parser = ThinkGearParser()
        self.stop_event = threading.Event()
        self.thread = None
        self.total_samples = self.dropped_windows = 0
        self.error = None
        self.state = "idle"
        self.bands = self.band_time = None
        self.poor_signal = self.attention = self.meditation = None
        self._samples = deque(maxlen=self.window_size)
        self._until_window = self.window_size
        self._last_raw = None

    @staticmethod
    def list_ports():
        from serial.tools import list_ports
        return [(p.device, p.description) for p in list_ports.comports()]

    @staticmethod
    def find_best_port(*, ports=None, baudrate=57600, observe_seconds=3.0,
                       timeout_per_port=8.0, cancel_event=None, on_progress=None,
                       results=None):
        """Return the best verified EEG port or None. Synchronous: call off your UI thread.

        Scan disconnected ports only. Prefers raw EEG, then good contact, then
        sample count and band updates. `results` optionally collects per-port reports.
        `on_progress(report)` is called after each probe. Cancellation returns None.
        """
        if not (math.isfinite(observe_seconds) and math.isfinite(timeout_per_port)
                and observe_seconds > 0 and timeout_per_port > observe_seconds):
            raise ValueError("Require finite 0 < observe_seconds < timeout_per_port")
        names = list(dict.fromkeys(ports if ports is not None else
                                   [name for name, _ in TGAMConnector.list_ports()]))
        candidates = []
        for port in names:
            if cancel_event is not None and cancel_event.is_set():
                return None
            report = _probe_one(port, baudrate, observe_seconds, timeout_per_port, cancel_event)
            if results is not None:
                results.append(report)
            if on_progress:
                on_progress(report)
            if report["status"] == "cancelled":
                return None
            if report["status"] == "EEG detected":
                candidates.append(report)
        if cancel_event is not None and cancel_event.is_set():
            return None
        if not candidates:
            return None
        def rank(report):
            quality = report["poor_signal"]
            return (report["raw_samples"] >= 3, quality is not None,
                    -quality if quality is not None else -256,
                    report["raw_samples"], report["band_updates"])
        return max(candidates, key=rank)["port"]

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Create a new connector for each acquisition session")
        self.thread = threading.Thread(target=self._run, name="TGAM-acquisition", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise TimeoutError("Serial acquisition has not stopped yet")

    def _run(self):
        try:
            if self.demo:
                self.state = "demo"
                began = time.monotonic()
                generated, next_band = 0, 0
                while not self.stop_event.is_set():
                    target = int((time.monotonic() - began) * 512)
                    if generated >= next_band:
                        values = tuple(int((i + 1) * 1000 * (1 + .25 * math.sin(generated / 512 + i))) for i in range(8))
                        self._set_bands(values)
                        self.poor_signal = 0
                        next_band = generated + 512
                    while generated < target and not self.stop_event.is_set():
                        t = generated / 512
                        self._add_raw(int(180 * math.sin(2 * math.pi * 10 * t) + 45 * math.sin(2 * math.pi * 22 * t)))
                        generated += 1
                    self.stop_event.wait(.005)
            else:
                import serial
                self.state = "connecting"
                with serial.Serial(self.port, self.baudrate, timeout=.25, write_timeout=1) as connection:
                    self.state = "waiting for data"
                    while not self.stop_event.is_set():
                        data = connection.read(max(1, min(connection.in_waiting, 4096)))
                        if data:
                            for code, value in self.parser.feed(data):
                                self._handle_row(code, value)
                        elif self._last_raw is not None and time.monotonic() - self._last_raw > 3:
                            self.state = "stream stalled"
        except Exception as exc:
            self.error = str(exc)
            self.state = "error"
            if self.on_error:
                self.on_error(exc)
        finally:
            if self.state != "error":
                self.state = "stopped"

    def _handle_row(self, code, value):
        if code == 0x80 and len(value) == 2:
            self._add_raw(int.from_bytes(value, "big", signed=True))
        elif code == 0x83 and len(value) == 24:
            self._set_bands(tuple(int.from_bytes(value[i:i + 3], "big") for i in range(0, 24, 3)))
        elif len(value) == 1:
            if code == 2:
                self.poor_signal = value[0]
            elif code == 4:
                self.attention = value[0]
            elif code == 5:
                self.meditation = value[0]

    def _set_bands(self, values):
        self.bands, self.band_time = values, time.monotonic()
        if self.on_bands:
            self.on_bands(datetime.now(timezone.utc).isoformat(), dict(zip(BAND_NAMES, values)))

    def _add_raw(self, raw):
        now = time.monotonic()
        if self._last_raw is not None and now - self._last_raw > 3:
            self._samples.clear()
            self._until_window = self.window_size
        self._last_raw = now
        if not self.demo:
            self.state = "receiving EEG"
        self.total_samples += 1
        self._samples.append(raw)
        stamp = datetime.now(timezone.utc).isoformat()
        if self.on_sample:
            self.on_sample(stamp, self.total_samples - 1, raw)
        self._until_window -= 1
        if len(self._samples) == self.window_size and self._until_window <= 0:
            self._until_window = self.stride
            window = EEGWindow(tuple(self._samples), 512, self.total_samples - 1, stamp,
                               self.bands, None if self.band_time is None else now - self.band_time,
                               self.poor_signal, self.attention, self.meditation, self.demo)
            try:
                self.windows.put_nowait(window)
            except queue.Full:
                try:
                    self.windows.get_nowait()
                    self.dropped_windows += 1
                except queue.Empty:
                    pass
                self.windows.put_nowait(window)


def load_model(spec: str) -> Callable:
    """Load module:function. The function accepts EEGWindow and returns JSON-compatible data."""
    module, separator, function = spec.partition(":")
    if not separator or not module or not function:
        raise ValueError("Model must be specified as module:function")
    predict = getattr(importlib.import_module(module), function)
    if not callable(predict):
        raise TypeError("Model entry point must be callable")
    return predict


class InferenceWorker:
    def __init__(self, connector, predict, on_result):
        self.connector, self.predict, self.on_result = connector, predict, on_result
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="EEG-inference", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                window = self.connector.windows.get(timeout=.25)
            except queue.Empty:
                continue
            began = time.monotonic()
            result = {"received_utc": window.received_utc,
                      "end_sample_index": window.end_sample_index,
                      "simulated": window.simulated,
                      "poor_signal": window.poor_signal,
                      "bands_age_seconds": window.bands_age_seconds}
            try:
                result["prediction"] = self.predict(window)
            except Exception as exc:
                result["error"] = str(exc)
            result["inference_ms"] = round((time.monotonic() - began) * 1000, 2)
            try:
                self.on_result(result)
            except Exception as exc:
                self.connector.error = "Inference output failed: " + str(exc)
                self.stop_event.set()
