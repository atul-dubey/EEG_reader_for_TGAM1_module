"""Pipeline smoke test only. Replace predict with your trained model."""
import math


def predict(window):
    mean = sum(window.samples) / len(window.samples)
    rms = math.sqrt(sum(x * x for x in window.samples) / len(window.samples))
    return {"model": "signal statistics (no AI model loaded)",
            "mean_raw_counts": round(mean, 3), "rms_raw_counts": round(rms, 3),
            "samples": len(window.samples)}
