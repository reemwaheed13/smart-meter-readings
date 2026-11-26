"""Machine and library versions, stored in every log and recording (README, "Recording format": env)."""
from __future__ import annotations

import os
import platform
import sys
from functools import lru_cache


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


@lru_cache(maxsize=1)
def machine_info() -> dict:
    """{python, flwr, torch, cpu_model, cpu_cores, gpu_visible}: plain str/int/bool values, so it
    survives torch.load(weights_only=True) and json."""
    import flwr
    import torch
    return {
        "python": str(platform.python_version()),
        "flwr": str(flwr.__version__),
        "torch": str(torch.__version__),
        "cpu_model": _cpu_model(),
        "cpu_cores": int(os.cpu_count() or 0),
        "gpu_visible": bool(torch.cuda.is_available()),
    }


if __name__ == "__main__":
    print(machine_info(), file=sys.stdout)
