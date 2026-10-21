import os

if os.environ.get("OFFLINE_MODE", "").strip().lower() in ("1", "true", "yes", "on"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

__version__ = "0.1.0"
