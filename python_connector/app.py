"""RENESON.in connector demo: CLI JSON output or optional Tk inference display."""
import argparse
import csv
import json
import sys
import queue
import time
from contextlib import ExitStack
from pathlib import Path

from tgam_connector import BAND_NAMES, InferenceWorker, TGAMConnector, load_model


def main():
    parser = argparse.ArgumentParser(description="RENESON.in TGAM1 EEG connector and inference")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--find-port", action="store_true", help="Scan, print best EEG port, and exit")
    parser.add_argument("--auto-port", action="store_true", help="Find the best EEG port before acquisition")
    parser.add_argument("--port", help="Outgoing Bluetooth or serial port, e.g. COM4")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--model", default="example_model:predict", help="module:function")
    parser.add_argument("--window", type=float, default=2)
    parser.add_argument("--stride", type=float, default=.5)
    parser.add_argument("--seconds", type=float, default=0, help="0 runs until stopped")
    parser.add_argument("--record", type=Path, help="New output directory: raw.csv, bands.csv, inference.jsonl")
    args = parser.parse_args()
    if args.list_ports:
        for port, description in TGAMConnector.list_ports():
            print(port, description)
        return
    if args.find_port or args.auto_port:
        if args.demo or args.port:
            parser.error("Port discovery cannot be combined with --demo or --port")
        reports = []
        print("Scanning ports for ThinkGear EEG data...", file=sys.stderr, flush=True)
        try:
            best = TGAMConnector.find_best_port(baudrate=args.baudrate, results=reports,
                on_progress=lambda report: print(json.dumps(report), file=sys.stderr, flush=True))
        except KeyboardInterrupt:
            parser.exit(130, "Scan cancelled.\n")
        if best is None:
            parser.exit(1, "No EEG port found. Check device power, pairing, and competing apps.\n")
        if args.find_port:
            print(best)
            return
        args.port = best
        print("Selected " + best, file=sys.stderr, flush=True)
    if not args.demo and not args.port:
        parser.error("Use --port COM4, --auto-port, or --demo")
    if args.seconds < 0:
        parser.error("--seconds must be nonnegative")
    predict = load_model(args.model)
    latest = queue.Queue(maxsize=1)
    with ExitStack() as files:
        sample_callback = band_callback = None
        inference_file = None
        if args.record:
            args.record.mkdir(parents=True, exist_ok=False)
            raw = csv.writer(files.enter_context((args.record / "raw.csv").open("x", newline="", encoding="utf-8")))
            raw.writerow(["received_utc", "session_sample_index", "raw_counts", "simulated"])
            bands = csv.writer(files.enter_context((args.record / "bands.csv").open("x", newline="", encoding="utf-8")))
            bands.writerow(["received_utc", *BAND_NAMES, "simulated"])
            sample_callback = lambda stamp, index, value: raw.writerow([stamp, index, value, args.demo])
            band_callback = lambda stamp, values: bands.writerow([stamp, *[values[name] for name in BAND_NAMES], args.demo])
            inference_file = files.enter_context((args.record / "inference.jsonl").open("x", encoding="utf-8"))

        def result_callback(result):
            # Serialization catches invalid output without losing all subsequent predictions.
            try:
                line = json.dumps(result, allow_nan=False)
            except (TypeError, ValueError) as exc:
                result.pop("prediction", None)
                result["error"] = "Model output must be JSON-compatible: " + str(exc)
                line = json.dumps(result, allow_nan=False)
            if inference_file:
                inference_file.write(line + "\n")
                inference_file.flush()
            if not args.gui:
                print(line, flush=True)
            try:
                latest.put_nowait(result)
            except queue.Full:
                try:
                    latest.get_nowait()
                except queue.Empty:
                    pass
                latest.put_nowait(result)

        connector = TGAMConnector(args.port, baudrate=args.baudrate, demo=args.demo,
                                  window_seconds=args.window, stride_seconds=args.stride,
                                  on_sample=sample_callback, on_bands=band_callback)
        worker = InferenceWorker(connector, predict, result_callback)
        root = None
        if args.gui:
            import tkinter as tk
            root = tk.Tk()
            root.title("RENESON.in EEG Reader for TGAM1 Modules")
            root.geometry("800x480")
            tk.Label(root, text="RENESON.in", font=("Segoe UI", 24, "bold")).pack(pady=12)
            tk.Label(root, text="TGAM1 EEG acquisition and model inference").pack()
            status = tk.Label(root, text="Starting acquisition", wraplength=760)
            status.pack(pady=10)
            output = tk.Text(root, height=16, font=("Consolas", 11))
            output.pack(fill="both", expand=True, padx=16, pady=10)
            closed = [False]
            root.protocol("WM_DELETE_WINDOW", lambda: (closed.__setitem__(0, True), root.quit()))
            def tick():
                if closed[0]:
                    return
                status.config(text=f"{connector.state} | samples: {connector.total_samples} | superseded inference windows: {connector.dropped_windows}" + (f" | {connector.error}" if connector.error else ""))
                try:
                    result = latest.get_nowait()
                    output.delete("1.0", "end")
                    output.insert("end", json.dumps(result, indent=2))
                except queue.Empty:
                    pass
                if args.seconds and time.monotonic() - began >= args.seconds:
                    root.quit()
                else:
                    root.after(100, tick)

        began = time.monotonic()
        connector.start()
        worker.start()
        try:
            if root:
                tick()
                root.mainloop()
            else:
                while connector.thread.is_alive() and not worker.stop_event.is_set():
                    if args.seconds and time.monotonic() - began >= args.seconds:
                        break
                    time.sleep(.1)
        except KeyboardInterrupt:
            pass
        finally:
            connector.stop()
            worker.stop()
            # Model hooks must eventually return. Do not close an output file under a running writer.
            if worker.thread.is_alive():
                worker.thread.join()
            if root:
                root.destroy()
        if connector.error:
            raise RuntimeError(connector.error)


if __name__ == "__main__":
    main()
