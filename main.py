"""Forecast GTA VI trophy earn rates on Exophase, one case at a time.

For every trophy the program writes five quantiles of its earn rate at the
case target, on the logit scale, from the case history alone. Standard
library only, no network, deterministic: the same input always writes the
same bytes.

How the forecast is built, in plain terms
------------------------------------------
An earn rate is earners divided by players tracked, and both move. On the
logit scale a trophy's rate drifts fairly smoothly against log time since
the release, because players keep earning trophies while new players keep
syncing their profiles, and both of those grow roughly like powers of time.
So for each trophy the program fits a straight line of logit(rate) against
ln(1 + hours since the release) through the history's distinct reads
(repeated reads that show the same rates carry no new information, since
Exophase only refreshes its counts every few hours and a read in between
just repeats the last refresh). The line is read at the target, anchored on
the trophy's last observed rate so a level error in the fit cannot shift
the forecast. A trophy with only a couple of distinct reads gets its slope
shrunk hard toward the average slope of its tier, and a trophy that never
moved at all is given slope zero: no evidence of movement means no trend
is claimed. Slopes are capped so one noisy rare trophy cannot explode.

The spread is the honest part. It combines the baseline random walk
(0.5 logit per root-day of horizon) with the fit's own prediction error,
which grows when the target sits far past the last read and when the
history is noisy, plus a floor for the fact that the answer is the rate as
of Exophase's last refresh before the target, not the rate at the target
itself. It is floored and capped so neither a flat history nor a wild fit
produces a degenerate band.

The special case is a history that never changed at all. That means no
refresh happened while the house was watching, so the next refresh is
still ahead of the target and it dilutes the rates: the first refresh
after launch night took tracked players from 638 to 3,868 while trophies
earned went from 943 to 4,276, so the average non-platinum rate fell to
about a fifth of what it was, roughly a 1.5 logit drop. When the whole
history is one unchanged page the median is therefore shifted down about
1.4 logit (except the trophy sitting at 100%, which the refresh left
alone) and the spread is widened to cover not knowing exactly how far
each trophy's earners grew in the meantime. This branch is a property of
the whole case, never of one trophy, because a single trophy standing
still while the rest move says nothing about a refresh.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_FOR_LEVEL = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}

# Random-walk spread of a rate, in logit units per square-root day.
WALK_SIGMA = 0.5
# Median logit shift applied when the whole history is one unchanged page:
# the next refresh dilutes the rates (see the module docstring).
FLAT_SHIFT = -1.4
FLAT_SIGMA = 0.8
# Spread floor and cap, in logit units.
SIGMA_FLOOR = 0.4
SIGMA_CAP = 1.8
# Slope caps, in logit per unit of ln(1 + hours).
SLOPE_LOW = -1.0
SLOPE_HIGH = 1.2
# Median shift from the last observed rate, capped either way.
SHIFT_CAP = 2.0
# Only reads from this many hours before the checkpoint feed the trend fit;
# older reads describe launch night, not the week that follows.
FIT_WINDOW_HOURS = 48.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(z):
    if z >= 0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as a plain float."""
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since(origin, moment):
    return (moment - origin).total_seconds() / 3600.0


def log_time(hours):
    return math.log(max(1.0 + hours, 0.05))


def distinct_points(history, index, origin):
    """(log-time, logit rate) pairs where this trophy's rate actually changed.

    Consecutive reads that repeat the rate are the same refresh seen twice,
    so only the first read carrying each new value is kept.
    """
    points = []
    seen = None
    for read in history:
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        rate = rates[index]
        if seen is not None and rate == seen:
            continue
        seen = rate
        try:
            moment = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        points.append((log_time(hours_since(origin, moment)), logit(rate)))
    return points


def fit_line(points):
    """Least-squares line of logit rate on log time. Returns slope, mean x,
    residual spread, and the Sxx the prediction error needs, or None."""
    n = len(points)
    if n < 2:
        return None
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    if sxx <= 1e-12:
        return None
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    slope = sxy / sxx
    if n >= 3:
        sse = sum((p[1] - (mean_y + slope * (p[0] - mean_x))) ** 2 for p in points)
        resid = math.sqrt(sse / max(n - 2, 1))
    else:
        resid = 0.35
    return {"slope": slope, "mean_x": mean_x, "resid": resid, "sxx": sxx, "n": n}


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")

    release = case.get("game", {}).get("release_date")
    if release:
        origin = datetime.fromisoformat(release + "T00:00:00+00:00")
    else:
        origin = parse_time(history[0]["at"])

    checkpoint = parse_time(case["at"])
    if case.get("target"):
        target = parse_time(case["target"])
    else:
        target = checkpoint + timedelta(minutes=case["horizon_minutes"])
    if target <= parse_time(history[-1]["at"]):
        # Nothing to extrapolate to: hold the last rates with a wide band.
        target = parse_time(history[-1]["at"])
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0
    x_target = log_time(hours_since(origin, target))
    x_last = log_time(hours_since(origin, parse_time(history[-1]["at"])))

    # One unchanged page through the whole history means no refresh was
    # seen yet; the next one is still ahead of the target.
    seen_pages = set()
    for read in history:
        rates = read.get("rates") or []
        seen_pages.add(tuple(rates))
    history_flat = len(seen_pages) <= 1

    # Per-trophy trend fits, from recent distinct reads only.
    window_start = log_time(
        hours_since(origin, checkpoint) - FIT_WINDOW_HOURS
    )
    fits = []
    for i in range(len(trophies)):
        points = distinct_points(history, i, origin)
        recent = [p for p in points if p[0] >= window_start]
        if len(recent) >= 3:
            points = recent
        elif len(points) >= 3:
            points = points
        fits.append((points, fit_line(points)))

    # Group slope per tier, from trophies whose fit has at least 3 points;
    # the fallback is the mean slope over all such trophies.
    by_tier = {}
    all_slopes = []
    for trophy, (_, fit) in zip(trophies, fits):
        if fit is not None and fit["n"] >= 3:
            by_tier.setdefault(trophy.get("tier", ""), []).append(fit["slope"])
            all_slopes.append(fit["slope"])
    global_slope = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        points, fit = fits[i]
        last_rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                last_rate = rates[i]
                break
        if last_rate is None:
            last_rate = 1.0
        last_logit = logit(last_rate)

        if history_flat:
            if last_rate >= 99.5:
                median_logit = last_logit
                sigma = 0.05
            else:
                median_logit = last_logit + FLAT_SHIFT
                sigma = FLAT_SIGMA
        else:
            if fit is not None and fit["n"] >= 3:
                tier_slopes = by_tier.get(trophy.get("tier", "")) or []
                group = (
                    sum(tier_slopes) / len(tier_slopes)
                    if tier_slopes
                    else global_slope
                )
                weight = fit["n"] / (fit["n"] + 4.0)
                slope = weight * fit["slope"] + (1.0 - weight) * group
                pred_var = fit["resid"] ** 2 * (
                    1.0
                    + 1.0 / fit["n"]
                    + (x_target - fit["mean_x"]) ** 2 / fit["sxx"]
                )
            elif fit is not None and fit["n"] == 2:
                group = global_slope
                slope = 0.3 * fit["slope"] + 0.7 * group
                pred_var = 0.35 ** 2 * 4.0
            else:
                slope = 0.0
                pred_var = 0.35 ** 2
            slope = min(SLOPE_HIGH, max(SLOPE_LOW, slope))
            shift = slope * (x_target - x_last)
            shift = min(SHIFT_CAP, max(-SHIFT_CAP, shift))
            median_logit = last_logit + shift
            sigma = math.sqrt(
                (WALK_SIGMA * math.sqrt(horizon_days)) ** 2
                + pred_var
                + 0.15 ** 2
            )
            sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
            if last_rate >= 99.5:
                median_logit = last_logit
                sigma = 0.05

        quantiles = []
        for level in case.get("levels", LEVELS):
            z = Z_FOR_LEVEL.get(level, 0.0)
            value = round4(inv_logit(median_logit + z * sigma))
            value = min(100.0, max(0.0, value))
            if quantiles and value < quantiles[-1]:
                value = quantiles[-1]
            quantiles.append(value)
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
