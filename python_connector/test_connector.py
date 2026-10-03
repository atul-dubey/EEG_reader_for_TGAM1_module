import json
import threading
import time
import unittest
from unittest.mock import patch

from tgam_connector import BAND_NAMES, InferenceWorker, TGAMConnector, ThinkGearParser


def stalled_probe(port, baudrate, seconds, pipe):
    time.sleep(30)


def packet(payload):
    return b"\xaa\xaa" + bytes([len(payload)]) + payload + bytes([~sum(payload) & 255])


class ConnectorTests(unittest.TestCase):
    def test_best_port_ranking_and_no_device(self):
        reports = [
            dict(port="COM1", raw_samples=0, band_updates=0, poor_signal=None, status="unavailable"),
            dict(port="COM2", raw_samples=0, band_updates=2, poor_signal=0, status="EEG detected"),
            dict(port="COM3", raw_samples=1500, band_updates=3, poor_signal=200, status="EEG detected"),
            dict(port="COM4", raw_samples=1400, band_updates=3, poor_signal=0, status="EEG detected"),
        ]
        collected, progress = [], []
        with patch("tgam_connector._probe_one", side_effect=reports):
            self.assertEqual(TGAMConnector.find_best_port(ports=[r["port"] for r in reports], results=collected, on_progress=progress.append), "COM4")
        self.assertEqual(collected, reports)
        self.assertEqual(progress, reports)
        with patch("tgam_connector._probe_one", return_value=reports[0]):
            self.assertIsNone(TGAMConnector.find_best_port(ports=["COM1"]))
        cancel = threading.Event()
        cancel.set()
        with patch("tgam_connector._probe_one") as probe:
            self.assertIsNone(TGAMConnector.find_best_port(ports=["COM1"], cancel_event=cancel))
            probe.assert_not_called()
        with self.assertRaises(ValueError):
            TGAMConnector.find_best_port(ports=[], observe_seconds=3, timeout_per_port=2)

    def test_probe_deadline_and_process_cleanup(self):
        import multiprocessing
        from tgam_connector import _probe_one
        before = {p.pid for p in multiprocessing.active_children()}
        with patch("tgam_connector._port_probe", stalled_probe):
            began = time.monotonic()
            result = _probe_one("COM_TEST", 57600, .05, .3, None)
        self.assertEqual(result["status"], "timeout")
        self.assertLess(time.monotonic() - began, 3)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_fragmentation_checksum_and_recovery(self):
        parser = ThinkGearParser()
        good = packet(b"\x80\x02\x80\x00")
        rows = []
        for byte in good:
            rows.extend(parser.feed(bytes([byte])))
        self.assertEqual(int.from_bytes(rows[0][1], "big", signed=True), -32768)
        bad = good[:-1] + bytes([good[-1] ^ 1])
        self.assertEqual(parser.feed(bad + good), [(0x80, b"\x80\x00")])
        self.assertEqual(parser.bad_checksums, 1)
        self.assertEqual(parser.feed(packet(b"\x80\x02\x00")), [])
        self.assertEqual(parser.malformed, 1)
        self.assertEqual(parser.feed(packet(b"\x55\x80\x02\x00\x01")), [])

    def test_unsigned_bands_and_record_callback(self):
        seen = []
        c = TGAMConnector(demo=True, on_bands=lambda stamp, values: seen.append(values))
        values = (0, 1, 255, 256, 65535, 65536, 8388608, 16777215)
        payload = b"".join(value.to_bytes(3, "big") for value in values)
        c._handle_row(0x83, payload[:-1])
        self.assertIsNone(c.bands)
        c._handle_row(0x83, payload)
        self.assertEqual(c.bands, values)
        self.assertEqual(seen[0], dict(zip(BAND_NAMES, values)))

    def test_window_stride_backpressure_and_gap(self):
        c = TGAMConnector(demo=True, window_seconds=4 / 512, stride_seconds=2 / 512)
        for i in range(8):
            c._add_raw(i)
        window = c.windows.get_nowait()
        self.assertEqual(window.samples, (4, 5, 6, 7))
        self.assertEqual(window.end_sample_index, 7)
        self.assertEqual(c.dropped_windows, 2)
        with patch("tgam_connector.time.monotonic", return_value=c._last_raw + 4):
            c._add_raw(8)
        self.assertEqual(tuple(c._samples), (8,))
        self.assertTrue(c.windows.empty())

    def test_inference_errors_and_slow_model_do_not_block_acquisition(self):
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        results = []
        c = TGAMConnector(demo=True, window_seconds=2 / 512, stride_seconds=1 / 512)
        def model(window):
            entered.set()
            release.wait(2)
            raise ValueError("model failure")
        def output(result):
            results.append(result)
            done.set()
        worker = InferenceWorker(c, model, output).start()
        try:
            c._add_raw(1)
            c._add_raw(2)
            self.assertTrue(entered.wait(1))
            for i in range(20):
                c._add_raw(i)
            self.assertGreater(c.dropped_windows, 0)
            release.set()
            self.assertTrue(done.wait(1))
            self.assertEqual(results[0]["error"], "model failure")
            json.dumps(results[0])
        finally:
            release.set()
            worker.stop()


if __name__ == "__main__":
    unittest.main()
