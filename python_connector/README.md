# RENESON.in Python EEG connector

Python 3.10+ connector for TGAM1 ThinkGear packet streams, including MindWave Mobile 2. Provides raw counts, eight band powers, signal quality and attention/meditation to your own model. The included model reports signal statistics to test the pipeline; it is **not a trained AI model**.

From this directory:

A project-local `.venv` with pyserial is already prepared on this machine. Run `./run.ps1 --demo --gui`, `./run.ps1 --list-ports`, or `./run.ps1 --port COM4 --gui --record recordings/session01` in PowerShell. Demo recording used for verification is in `verification_demo/` and is marked simulated.

```powershell
python -m pip install -r requirements.txt
python app.py --list-ports
python app.py --demo --gui
python app.py --port COM4 --gui --record recordings/session01
python app.py --port COM4 --model my_model:predict --window 2 --stride 0.5
```

Close the Windows EEG Reader, Tutorial and ThinkGear Connector before connecting. Select the outgoing Bluetooth COM port for a headset, or the module's serial adapter port. Default transport is 57600 baud, 8N1; use `--baudrate` if your module is configured differently. It must send ThinkGear packets; the connector sends no configuration commands. Demo runs without pyserial or a device. `--seconds 5` stops a CLI or GUI demo after five seconds. Tkinter is required only for `--gui`.

The bundled Python available on this machine is `C:\Users\atuld\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`. In PowerShell, replace `python` above with `& 'C:\Users\atuld\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'` if Python is not on PATH. A standard Python installation also works.

## Model integration

### Automatic port discovery

```powershell
.\run.ps1 --find-port
.\run.ps1 --auto-port --gui
.\run.ps1 --auto-port --model my_model:predict --record recordings/session02
```

Use the API in your own application:

```python
from tgam_connector import TGAMConnector

if __name__ == "__main__":  # Required for Windows multiprocessing/spawn
    reports = []
    port = TGAMConnector.find_best_port(results=reports, on_progress=print)
    if port is None:
        raise RuntimeError("No device sending valid EEG data was found")
    connector = TGAMConnector(port).start()
    # Consume connector.windows or attach your InferenceWorker.
```

Discovery scans all listed ports at 57600 baud, observing each open port for three seconds with an eight-second overall deadline per port (including open/close). Each probe runs in an isolated process so a stuck Bluetooth driver can be terminated. No commands are sent. It requires at least three valid raw samples or one complete band-power update, ranks raw-capable ports first, then known/better signal quality, raw sample count and band updates. Returns the port name or `None`; `results` receives status reports for busy, silent, timed-out and detected ports. This ranking is a heuristic; supply `ports=["COM4", "COM5"]` to restrict the candidates or `--port` to override discovery. Metadata such as a Bluetooth device name alone is not accepted as detection.

Call this synchronous method from a background thread in your GUI. An optional `threading.Event` passed as `cancel_event` cancels probing and returns `None`. Scan before connecting and close all other headset apps. All scan processes release their port before the method returns. Scanning many ports may take eight seconds per port; discovery does not persist an old result. The script's `if __name__ == "__main__"` guard is important on Windows; do not invoke discovery at module import time.

Create `my_model.py` beside `app.py`, or use an importable module. Load your trained model once at module import, then implement:

```python
def predict(window):
    # window.samples: immutable tuple of raw counts, chronological order
    # window.sample_rate: 512 Hz nominal
    # window.bands: tuple in BAND_NAMES order, or None
    # window.bands_age_seconds: age of most recent band reading, or None
    # window.poor_signal: 0 good, 200 no contact, None unknown
    # window.simulated: True for demo
    # Apply the SAME scaling/filtering/input shape used during training.
    # Return a JSON-compatible dict/list/number; convert tensor/NumPy output.
    return {"label": your_model_label, "confidence": float(your_model_confidence)}
```

Run `python app.py --port COM4 --model my_model:predict --gui`. PyTorch, TensorFlow, ONNX or a remote inference client can be called inside this function; install the dependencies your model needs separately. Set timeouts for remote requests. The connector does not infer preprocessing or label meanings. Your model determines how to handle poor signal, missing or stale bands. This application displays every result and its signal-quality metadata rather than silently classifying bad contact as reliable data.

Acquisition and model execution run on separate threads. Default windows contain 1024 samples (2 seconds) with a 256-sample stride (0.5 seconds). A one-window queue replaces pending windows when inference lags; the GUI reports this count. Raw recording still receives every decoded sample. Recording callbacks run in the acquisition thread: keep custom callbacks fast. Long blocking Python model code holding the GIL can still affect acquisition; use a separate process/service for heavy workloads. A model hook must eventually return; shutdown waits for an in-progress call to finish.

`TGAMConnector` and `InferenceWorker` can also be imported directly into your application. Results are delivered on the inference worker thread; marshal GUI updates onto your framework's UI thread, as this example does. Create a new connector for each connection session. Acquisition errors appear in `connector.error` and its `state`; the CLI exits with an error and the GUI displays it. Reconnection is explicit.

## Recording and timing

`--record` creates a new directory with `raw.csv`, `bands.csv` and `inference.jsonl`; it refuses to overwrite an existing directory. Band CSV contains one row per received band packet (typically once a second). Raw values are uncalibrated counts, band values are relative unitless power. Demo outputs are marked simulated. Shutdown flushes files; a process crash may lose buffered CSV rows.

Timestamps are host receipt times, not headset sampling timestamps. Bluetooth batches can arrive together. Sample indices count received samples; the TGAM stream has no per-sample sequence counter to detect all losses. Windows use a nominal 512 Hz and reset after a host-observed raw-data gap over three seconds, but short gaps or lost packets cannot be reconstructed reliably. This is a single-channel stream; use a model trained for that input.

```powershell
python -m unittest -v
python app.py --demo --seconds 5 --record recordings/demo01
```

Protocol: https://developer.neurosky.com/docs/doku.php?id=thinkgear_communications_protocol
