"""Tests run on the fake stub (data_fake.py), never on the real data. Set before anything imports data."""
import os

os.environ["SMART_METER_DATA"] = "fake"
