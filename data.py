"""data.py: the one module the rest of the code imports. It routes to the real or the fake data.

    SMART_METER_DATA unset or "real"  -> data_real.py (London Smartmeter 2013; needs the processed
                                         .npz, see data_real.py)
    SMART_METER_DATA=fake             -> data_fake.py (random stand-in; tests/conftest.py sets this)

Interface (README, "Handshakes"):
    get_neighbourhood(n, k), valid_dates(house_ids, split), get_windows(house_ids, dates, split),
    attacker_houses(), unscale(x).  FAKE tells you which one is loaded.
"""
from __future__ import annotations

import os

SOURCE = os.environ.get("SMART_METER_DATA", "real").strip().lower()

if SOURCE == "fake":
    from data_fake import (FAKE, attacker_houses, get_neighbourhood, get_windows,  # noqa: F401
                           unscale, valid_dates)
elif SOURCE == "real":
    from data_real import (FAKE, attacker_houses, get_neighbourhood, get_windows,  # noqa: F401
                           unscale, valid_dates)
else:
    raise ValueError(f"SMART_METER_DATA must be 'real' or 'fake', got {SOURCE!r}")
