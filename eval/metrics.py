"""
metrics.py - every score in the guide's Scoring table.

All curve functions take numpy arrays in kWh (call data.unscale first) with the last
axis = readings. A window has 96 readings = 2 days of 48.

    smape(a, b)                 sMAPE in %, lower = closer
    match_windows(rebuilt, true) Hungarian matching on sMAPE -> rebuilt reordered to true
    morning_rise(day), evening_peak(day)   slot index of each event (0..47)
    routine_errors(rebuilt, true)          hours off, morning rise and evening peak
    away_label(day)             True if the house looks empty during the daytime
    paired_test(attack, blind, higher_is_better)  mean improvement, SE, "protection holds"
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

SLOT_HOURS = 0.5
# daytime windows, as half-hour slot numbers (slot 0 = 00:00-00:30)
MORNING = (8, 24)     # 04:00 - 12:00, where we look for the morning rise
EVENING = (32, 48)    # 16:00 - 24:00, where we look for the evening peak
DAYTIME = (18, 34)    # 09:00 - 17:00, used for home/away
NIGHT = (2, 10)       # 01:00 - 05:00, standby baseline
AWAY_RATIO = 1.3      # away if daytime mean < 1.3 x night baseline


# ---------- curve error ----------
def smape(a, b, axis=-1):
    """Symmetric mean absolute percentage error, in %. 0 = identical, 200 = worst.
    Readings where both values are 0 count as a perfect match."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    num = 2 * np.abs(a - b)
    den = np.abs(a) + np.abs(b)
    ratio = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return 100 * ratio.mean(axis=axis)


def match_windows(rebuilt, true):
    """The attacker doesn't know which rebuilt window is which, so pair each true window
    with one rebuilt window, minimising total sMAPE (Hungarian algorithm).

    rebuilt: [m, 96], true: [m, 96]  ->  rebuilt reordered so rebuilt[i] pairs with true[i]
    """
    rebuilt, true = np.asarray(rebuilt), np.asarray(true)
    cost = smape(true[:, None, :], rebuilt[None, :, :])  # [true, rebuilt]
    rows, cols = linear_sum_assignment(cost)
    out = np.empty_like(true)
    out[rows] = rebuilt[cols]
    return out


# ---------- routine timing ----------
def _smooth(day):
    k = np.ones(3) / 3
    return np.convolve(np.pad(day, 1, mode="edge"), k, mode="valid")


def morning_rise(day):
    """Slot of the biggest jump in use between 04:00 and 12:00 (people getting up)."""
    s = _smooth(np.asarray(day, float))
    lo, hi = MORNING
    jumps = np.diff(s[lo - 1:hi])        # jump INTO each slot lo..hi-1
    return lo + int(np.argmax(jumps))


def evening_peak(day):
    """Slot of the highest use between 16:00 and midnight."""
    s = _smooth(np.asarray(day, float))
    lo, hi = EVENING
    return lo + int(np.argmax(s[lo:hi]))


def routine_errors(rebuilt_window, true_window):
    """Hours off for morning rise and evening peak, averaged over the window's 2 days."""
    rise, peak = [], []
    for d in range(2):
        r, t = rebuilt_window[48 * d:48 * (d + 1)], true_window[48 * d:48 * (d + 1)]
        rise.append(abs(morning_rise(r) - morning_rise(t)) * SLOT_HOURS)
        peak.append(abs(evening_peak(r) - evening_peak(t)) * SLOT_HOURS)
    return float(np.mean(rise)), float(np.mean(peak))


# ---------- home or away ----------
def away_label(day):
    """True if the day looks unoccupied in the daytime: daytime use stays near standby."""
    day = np.asarray(day, float)
    night = np.median(day[NIGHT[0]:NIGHT[1]])
    daytime = day[DAYTIME[0]:DAYTIME[1]].mean()
    return bool(daytime < AWAY_RATIO * max(night, 1e-6))


def away_labels(window):
    return [away_label(window[48 * d:48 * (d + 1)]) for d in range(2)]


def balanced_accuracy(pred, true):
    """Mean of accuracy on 'away' days and on 'home' days (fair when one is rare)."""
    pred, true = np.asarray(pred, bool), np.asarray(true, bool)
    parts = [np.mean(pred[true == c] == c) for c in (True, False) if (true == c).any()]
    return float(np.mean(parts)) if parts else float("nan")


# ---------- statistics ----------
def mean_se(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(se)


def paired_test(attack, blind, higher_is_better=False):
    """Compare attack vs blind guess on the SAME houses (one value per house per repeat).

    improvement > 0 means the attack did better than blind guessing.
    Protection holds if improvement <= 2 standard errors.
    """
    attack, blind = np.asarray(attack, float), np.asarray(blind, float)
    diff = attack - blind if higher_is_better else blind - attack
    m, se = mean_se(diff)
    holds = bool(m <= 2 * se) if not np.isnan(se) else bool(m <= 0)
    return {"improvement": m, "se": se, "holds": holds, "count": int((~np.isnan(diff)).sum())}
