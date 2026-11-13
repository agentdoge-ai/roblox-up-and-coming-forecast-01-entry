#!/usr/bin/env python3
"""Forecaster for make.fun's roblox-up-and-coming-forecast-01 (v2).

Reads the Up-and-Coming chart's history from $MAKEFUN_INPUT_DIR (else /input)
and writes one static forecast to $MAKEFUN_OUTPUT_DIR/forecast.json (else
/output/forecast.json): for each of the top 10 places, a probability for each
named game being at that place, plus a top-10 guess for the tie-break.

The chart churns hard -- most of the top 10 changes within a day -- so a
persistence forecast (the house baseline) scores poorly. This forecaster
estimates, for every game and place, the share of the coming window's hours
the game is expected to spend at that place, blending:

  * recency-weighted place occupancy over the whole history (half-life 3 h:
    backtests preferred the freshest reads over v1's 12 h),
  * a wide Gaussian kernel around each game's place on the last chart, which
    carries the window's first hours,
  * a horizon-marginal table estimated from the history itself: for every
    pair of reads up to 72 h apart, where a game at place j ended up -- the
    average over the window's horizons, applied to the last two charts.
    Pairs are binned by their time gap and the bins averaged with equal
    weight, so the estimate mimics a uniform average over horizons 0..72 h
    instead of overweighting whichever gaps the reads cluster at (the final
    history's hourly house reads would otherwise drown the long horizons),
  * a momentum adjustment in two parts: each game's horizon-marginal row is
    read at an effective rank shifted by its rank velocity (a riser's
    destinations are those of the better place it is heading for), and a
    scalar tilt multiplies games by their velocity,
  * player-count and cross-chart features, the strongest additions over v1:
    a game's playing-count percentile on the last chart, its playing-count
    growth over the last ~24 h, and its presence near the top of the Top
    Playing Now chart in the last few reads each tilt its mass (a game
    people are actually playing, and gaining players, holds and gains
    places more than its rank alone suggests).

Probabilities are kept honest and smoothed: at most 80 games are named per
place, each place's named mass is 0.85 (the rest is left to the scoring
rule's even spread over the read's unnamed chart games, which is also the
only mass a brand-new game can receive), and 18% of the named mass is
spread uniformly over the named games so nothing plausible sits near zero.

Every constant was chosen by walk-forward backtests inside the practice
history (train on a prefix of the reads, score the next 72 h with the
official scoring rule): the blend beats v1 on both halves of the folds
and the house baseline by a wide margin.

Python 3 standard library only, no network, deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone

OCC_HALF_LIFE_H = 3.0    # half-life of the occupancy recency weights
W_OCC = 1.5              # weight: recency-weighted place occupancy
W_LAST = 0.7             # weight: Gaussian kernel around the last chart
SIG_LAST = 7.5           # that kernel's width, in places
W_HOR = 5.0              # weight: horizon marginal from the last chart
W_HOR_PREV = 0.8         # weight: horizon marginal from the previous chart
TILT = 0.7               # momentum scalar tilt strength
TAU_EFF = 0.35           # momentum effective-rank shift, in days
MOM_WINDOW_H = 30.0      # rank velocity is measured over this window
BIN_H = 6.0              # horizon-marginal time-gap bin width, hours
BIN_ALPHA = 20.0         # per-bin shrinkage toward the pooled table
F_PLAY_LEVEL = 1.0       # weight: playing-count percentile on last chart
F_PLAY_GROWTH = 0.65     # weight: playing-count growth over ~24 h
F_TPN = 0.8              # weight: Top Playing Now presence, last reads
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
    "created": {universe_id: epoch}, "playing": {universe_id: count},
    "tpn": {universe_id: rank}} where "created" carries the games' Roblox
    creation times when the read's details provide them, "playing" the
    up-and-coming rows' player counts, and "tpn" the same read's Top
    Playing Now ranks.
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
            rows = sorted(raw.get("up_and_coming") or [],
                          key=lambda r: r.get("rank", 0))
            chart = [row["universe_id"] for row in rows]
            if not chart:
                continue
            playing = {}
            for row in rows:
                if row.get("playing") is not None:
                    playing[row["universe_id"]] = row["playing"]
            tpn = {}
            for i, row in enumerate(sorted(
                    raw.get("top_playing_now") or [],
                    key=lambda r: r.get("rank", 0))):
                tpn[row["universe_id"]] = i + 1
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
                          "chart": chart, "created": created,
                          "playing": playing, "tpn": tpn})
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
    pair of reads in the history -- the window-average behaviour of a game
    given where it stands now, learned from the chart's own past. Pairs
    are binned by time gap (BIN_H-wide bins); each bin's table is shrunk
    toward the pooled table by BIN_ALPHA pseudo-counts, and the bins are
    averaged with equal weight, so dense clusters of reads (the final
    history's hourly stretch) cannot overweight the short horizons."""
    max_rank = max(len(r["chart"]) for r in reads) + 2
    n_bins = max(1, int(horizon_h // BIN_H))
    bin_tab = [[[0.0] * (places + 1) for _ in range(max_rank)]
               for _ in range(n_bins)]
    bin_cnt = [[0.0] * max_rank for _ in range(n_bins)]
    all_tab = [[0.0] * (places + 1) for _ in range(max_rank)]
    all_cnt = [0.0] * max_rank
    for a, earlier in enumerate(reads):
        pos_a = {g: i + 1 for i, g in enumerate(earlier["chart"])}
        for later in reads[a + 1:]:
            dt = (later["at"] - earlier["at"]) / 3600.0
            if dt > horizon_h:
                break
            b = min(n_bins - 1, int(dt // BIN_H))
            pos_b = {g: i + 1 for i, g in enumerate(later["chart"][:places])}
            for g, j in pos_a.items():
                bin_cnt[b][j] += 1
                all_cnt[j] += 1
                if g in pos_b:
                    bin_tab[b][j][pos_b[g]] += 1
                    all_tab[j][pos_b[g]] += 1
    table = [[0.0] * (places + 1) for _ in range(max_rank)]
    for j in range(1, max_rank):
        if all_cnt[j] <= 0:
            continue
        pooled = [all_tab[j][k] / all_cnt[j] for k in range(places + 1)]
        acc = [0.0] * (places + 1)
        used = 0
        for b in range(n_bins):
            if bin_cnt[b][j] <= 0:
                continue
            used += 1
            denom = bin_cnt[b][j] + BIN_ALPHA
            for k in range(1, places + 1):
                acc[k] += (bin_tab[b][j][k] + BIN_ALPHA * pooled[k]) / denom
        if used:
            for k in range(1, places + 1):
                table[j][k] = acc[k] / used
    return table


def hor_row(table, je, places):
    """Horizon-marginal row at a fractional rank, linearly interpolated
    between the neighbouring integer ranks."""
    max_rank = len(table) - 1
    je = max(1.0, min(float(max_rank - 1), je))
    lo = int(je)
    frac = je - lo
    return [table[lo][k] * (1.0 - frac) + table[lo + 1][k] * frac
            for k in range(places + 1)]


def game_features(reads):
    """Multiplicative feature factors per game, each centred near 1:
    playing-count percentile on the last chart (a game near the top of
    the player counts holds its place better), playing-count growth over
    the last ~24 h (log ratio, clipped), and presence near the top of
    the Top Playing Now chart over the last few reads. Games a feature
    says nothing about keep a factor of 1."""
    last = reads[-1]
    out = {}
    games = set()
    for r in reads:
        games.update(r["chart"])
    pct = {}
    lp = last.get("playing", {})
    if lp:
        order = sorted(lp, key=lambda g: (lp[g], g))
        for i, g in enumerate(order):
            pct[g] = (i + 0.5) / len(order)
    growth = {}
    target = last["at"] - 24 * 3600.0
    past = None
    for r in reads[:-1]:
        if r["at"] <= target:
            past = r
        else:
            break
    if past is not None:
        pp = past.get("playing", {})
        for g, now in lp.items():
            if g in pp and pp[g] > 0 and now > 0:
                growth[g] = max(-1.5, min(1.5, math.log(now / pp[g])))
    recent = reads[-3:]
    for g in games:
        f = 1.0
        if g in pct:
            f *= 1.0 + F_PLAY_LEVEL * (pct[g] - 0.5)
        if g in growth:
            f *= 1.0 + F_PLAY_GROWTH * growth[g]
        s = 0.0
        for r in recent:
            rk = r.get("tpn", {}).get(g)
            if rk:
                s += max(0.0, 1.0 - rk / 60.0)
        f *= 1.0 + F_TPN * (s / len(recent))
        if f != 1.0:
            out[g] = max(0.05, f)
    return out


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
    feat = game_features(reads)
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
                je = last_rank[g] - momentum.get(g, 0.0) * TAU_EFF
                val += W_HOR * hor_row(horizon, je, places)[place]
            if g in prev_rank:
                je = prev_rank[g] - momentum.get(g, 0.0) * TAU_EFF
                val += W_HOR_PREV * hor_row(horizon, je, places)[place]
            if val <= 0.0:
                continue
            val *= 1.0 + TILT * momentum.get(g, 0.0) / 8.0
            val *= feat.get(g, 1.0)
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
