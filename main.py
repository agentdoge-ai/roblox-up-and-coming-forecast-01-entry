#!/usr/bin/env python3
"""Forecaster for make.fun's roblox-up-and-coming-forecast-01.

Reads the Up-and-Coming chart's history from $MAKEFUN_INPUT_DIR (else /input)
and writes one static forecast to $MAKEFUN_OUTPUT_DIR/forecast.json (else
/output/forecast.json): for each of the top 10 places, a probability for each
named game being at that place, plus a top-10 guess for the tie-break.

The chart churns hard -- most of the top 10 changes within a day -- so a
persistence forecast (the house baseline) scores poorly. This forecaster
instead estimates, for every game and place, the share of the coming window's
hours the game is expected to spend at that place, blending:

  * recency-weighted place occupancy over the whole history (half-life 12 h),
  * a wide Gaussian kernel around each game's place on the last chart, which
    carries the window's first hours,
  * a horizon-marginal table estimated from the history itself: for every
    pair of reads up to 72 h apart, where a game at place j ended up -- the
    average over the window's horizons, applied to the last two charts,
  * a momentum tilt from each game's rank velocity over the last 30 hours.

Probabilities are kept honest and smoothed: at most 80 games are named per
place, each place's named mass is 0.85 (the rest is left to the scoring
rule's even spread over the read's unnamed chart games, which is also the
only mass a brand-new game can receive), and 18% of the named mass is
spread uniformly over the named games so nothing plausible sits near zero.

Python 3 standard library only, no network, deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone

# Tunables, chosen by walk-forward backtests inside the practice history
# (train on a prefix of the reads, score the next 72 h with the official
# scoring rule): the blend beat the persistence baseline on 14 of 15 folds.
OCC_HALF_LIFE_H = 12.0   # half-life of the occupancy recency weights
W_OCC = 1.0              # weight: recency-weighted place occupancy
W_LAST = 0.7             # weight: Gaussian kernel around the last chart
SIG_LAST = 7.5           # that kernel's width, in places
W_HOR = 5.0              # weight: horizon marginal from the last chart
W_HOR_PREV = 0.8         # weight: horizon marginal from the previous chart
TILT = 0.7               # momentum tilt strength
MOM_WINDOW_H = 30.0      # rank velocity is measured over this window
MAX_ROWS_PER_PLACE = 80  # games named per place (the rules allow 100)
TARGET_SUM = 0.85        # named mass per place; the rest stays unnamed
EPS_UNIFORM = 0.18       # share of named mass spread evenly as a floor
CHART_AGE_LIMIT_D = 28.0 # games leave the chart 28 days after creation


def parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds, or None."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def load_input(input_dir):
    """Returns (places, window_from, window_until, reads).

    Each read is {"at": epoch, "chart": [universe_id, ...] in chart order,
    "created": {universe_id: epoch}} where "created" carries the games'
    Roblox creation times when the read's details provide them.
    """
    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as fh:
        task = json.load(fh)
    places = int(task.get("places", 10))
    window = task.get("window", {})
    win_from = parse_time(window.get("from")) or parse_time(task.get("history_until"))
    win_until = parse_time(window.get("until"))
    if win_from is not None and win_until is None:
        slots = int(window.get("slots", 72))
        slot_minutes = int(window.get("slot_minutes", 60))
        win_until = win_from + slots * slot_minutes * 60

    reads = []
    path = os.path.join(input_dir, "history.ndjson")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            rows = raw.get("up_and_coming") or []
            chart = [row["universe_id"] for row in
                     sorted(rows, key=lambda r: r.get("rank", 0))]
            if not chart:
                continue
            created = {}
            details = raw.get("details") or {}
            if isinstance(details, dict):
                for key, det in details.items():
                    try:
                        made = parse_time((det or {}).get("created"))
                        if made is not None:
                            created[int(key)] = made
                    except (TypeError, ValueError):
                        continue
            reads.append({"at": parse_time(raw.get("at")) or 0.0,
                          "chart": chart, "created": created})
    reads.sort(key=lambda r: r["at"])
    return places, win_from, win_until, reads


def gauss(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def occupancy_and_momentum(reads, places):
    """Recency-weighted place occupancy and rank-velocity momentum.

    Returns (occ, momentum, last_rank, prev_rank): occ[g][k] is the weighted
    share of reads with game g at place k; momentum[g] is g's rank velocity
    in places gained per day over the momentum window, clipped to +-8.
    """
    t_last = reads[-1]["at"]
    weights = [0.5 ** ((t_last - r["at"]) / (OCC_HALF_LIFE_H * 3600.0))
               for r in reads]
    total_w = sum(weights) or 1.0
    occ = {}
    for read, wgt in zip(reads, weights):
        for i, g in enumerate(read["chart"][:places]):
            slot = occ.setdefault(g, [0.0] * (places + 1))
            slot[i + 1] += wgt / total_w
    window = [r for r in reads
              if (t_last - r["at"]) / 3600.0 <= MOM_WINDOW_H]
    momentum = {}
    games = set()
    for read in reads:
        games.update(read["chart"])
    for g in games:
        seen = [(r["at"], r["chart"].index(g) + 1)
                for r in window if g in r["chart"]]
        vel = 0.0
        if len(seen) >= 2 and seen[-1][0] > seen[0][0]:
            days = (seen[-1][0] - seen[0][0]) / 86400.0
            vel = (seen[0][1] - seen[-1][1]) / days
        momentum[g] = max(-8.0, min(8.0, vel))
    last_rank = {g: i + 1 for i, g in enumerate(reads[-1]["chart"])}
    prev_rank = ({g: i + 1 for i, g in enumerate(reads[-2]["chart"])}
                 if len(reads) > 1 else {})
    return occ, momentum, last_rank, prev_rank


def horizon_marginals(reads, places, horizon_h=72.0):
    """H[j][k]: share of later reads (up to horizon_h after a read) in which
    a game that stood at place j is found at place k, estimated from every
    pair of reads in the history. This is the window-average behaviour of a
    game given where it stands now, learned from the chart's own past."""
    max_rank = max(len(r["chart"]) for r in reads) + 2
    table = [[0.0] * (places + 1) for _ in range(max_rank)]
    counts = [0.0] * max_rank
    for a, earlier in enumerate(reads):
        pos_a = {g: i + 1 for i, g in enumerate(earlier["chart"])}
        for later in reads[a + 1:]:
            if (later["at"] - earlier["at"]) / 3600.0 > horizon_h:
                break
            pos_b = {g: i + 1 for i, g in enumerate(later["chart"][:places])}
            for g, j in pos_a.items():
                counts[j] += 1
                if g in pos_b:
                    table[j][pos_b[g]] += 1
    for j in range(1, max_rank):
        if counts[j] > 0:
            for k in range(1, places + 1):
                table[j][k] /= counts[j]
    return table


def eligibility(reads, win_from, win_until):
    """Share of the scored window in which each game can still be on the
    chart: a game leaves 28 days after its Roblox creation time, so one
    whose 28th day falls inside the window is only present for part of it.
    Games whose creation time no read's details report keep a factor of 1.
    """
    created = {}
    for read in reads:
        created.update(read["created"])
    if not created or win_from is None or win_until is None \
            or win_until <= win_from:
        return {}
    span = win_until - win_from
    factors = {}
    for g, made in created.items():
        left = (made + CHART_AGE_LIMIT_D * 86400.0 - win_from) / span
        factors[g] = max(0.0, min(1.0, left))
    return factors


def build_forecast(places, win_from, win_until, reads):
    """Assembles the forecast: (probability rows, top10 list)."""
    if not reads:
        return [], []
    occ, momentum, last_rank, prev_rank = occupancy_and_momentum(reads, places)
    horizon = horizon_marginals(reads, places)
    elig = eligibility(reads, win_from, win_until)
    games = set()
    for read in reads:
        games.update(read["chart"])

    rows = []
    mass = {}
    expected_place = {}
    for place in range(1, places + 1):
        raw = {}
        for g in games:
            val = W_OCC * occ.get(g, [0.0] * (places + 1))[place]
            if g in last_rank:
                val += W_LAST * gauss(place, last_rank[g], SIG_LAST)
                val += W_HOR * horizon[last_rank[g]][place]
            if g in prev_rank:
                val += W_HOR_PREV * horizon[prev_rank[g]][place]
            if val <= 0.0:
                continue
            val *= 1.0 + TILT * momentum.get(g, 0.0) / 8.0
            val *= elig.get(g, 1.0)
            if val > 0.0:
                raw[g] = val
        named = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))
        named = named[:MAX_ROWS_PER_PLACE]
        total = sum(v for _, v in named)
        if total <= 0.0:
            continue
        share = len(named)
        for g, val in named:
            p = TARGET_SUM * ((1.0 - EPS_UNIFORM) * val / total
                              + EPS_UNIFORM / share)
            rows.append({"universe_id": g, "place": place, "p": p})
            mass[g] = mass.get(g, 0.0) + p
            expected_place[g] = expected_place.get(g, 0.0) + place * p
    # The tie-break guess: the likeliest top 10, ordered by expected place.
    by_mass = sorted(mass, key=lambda g: (-mass[g], g))[:places + 4]
    by_mass.sort(key=lambda g: (expected_place[g] / mass[g], g))
    top10 = by_mass[:places]
    if len(top10) < places:  # degenerate history: pad from the last chart
        for g in reads[-1]["chart"]:
            if g not in top10:
                top10.append(g)
            if len(top10) == places:
                break
    rows.sort(key=lambda r: (r["place"], r["universe_id"]))
    return rows, top10


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    places, win_from, win_until, reads = load_input(input_dir)
    rows, top10 = build_forecast(places, win_from, win_until, reads)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "forecast.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"probabilities": rows, "top10": top10}, fh)
        fh.write("\n")


if __name__ == "__main__":
    main()
