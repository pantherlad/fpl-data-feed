"""
Model 1 — Distribution model (floor / median / ceiling), via Monte Carlo.

WHY THIS EXISTS
---------------
Mean xPts hides the shape of the outcome. GoalIQAI's point: a player projected
at 5.06 may most often score 2, with the mean lifted by a small chance of a
15-point haul. For captaincy — and especially when chasing a monthly podium —
the CEILING matters more than the mean. Conversely when protecting a paying
position, the FLOOR matters more.

This simulates each player's gameweek ~10,000 times using the same component
probabilities as the xPts model, then reports the distribution.

Outputs per player:
  mean      - the xPts figure (sanity check against the analytic model)
  floor     - 20th percentile (a bad-but-not-disastrous week)
  median    - the modal-ish outcome; usually LOWER than the mean
  ceiling   - 90th percentile (the haul scenario)
  p_blank   - P(<= 2 points), i.e. appearance points only
  p_haul    - P(>= 10 points)

USE:
  Chasing (need variance)  -> rank by ceiling / p_haul
  Protecting (need safety) -> rank by floor / (1 - p_blank)
"""

import math
import random


def simulate_player(player_ctx, n_sims=10000, seed=42):
    """
    player_ctx: dict with keys
      position, p_none, p_sub, p_full, exp_goals, exp_assists,
      p_cs, team_xg_against, dc_per90, defcon_threshold,
      bonus_rate, p_yellow, is_gk
    """
    rng = random.Random(seed)
    pos = player_ctx['position']
    goal_value = {'GKP': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}[pos]
    cs_value = {'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}[pos]

    results = []
    for _ in range(n_sims):
        pts = 0
        r = rng.random()
        if r < player_ctx['p_none']:
            results.append(0)
            continue
        played_full = r < (player_ctx['p_none'] + player_ctx['p_full'])
        # appearance
        pts += 2 if played_full else 1
        mins_scale = 1.0 if played_full else 0.45

        # goals (Poisson draw)
        goals = _poisson_draw(rng, player_ctx['exp_goals'] * mins_scale)
        pts += goals * goal_value

        # assists
        assists = _poisson_draw(rng, player_ctx['exp_assists'] * mins_scale)
        pts += assists * 3

        # clean sheet / conceded
        conceded = _poisson_draw(rng, player_ctx['team_xg_against'])
        if played_full:
            if conceded == 0:
                pts += cs_value
            if pos in ('GKP', 'DEF'):
                pts -= conceded // 2

        # goalkeeper saves
        if pos == 'GKP' and played_full:
            saves = _poisson_draw(rng, player_ctx['team_xg_against'] * 2.6)
            pts += saves // 3

        # DEFCON threshold
        if played_full and player_ctx['dc_per90'] > 0:
            actions = _poisson_draw(rng, player_ctx['dc_per90'])
            if actions >= player_ctx['defcon_threshold']:
                pts += 2

        # bonus — correlated with returns (only realistic if they returned)
        if goals + assists > 0 or (conceded == 0 and pos in ('GKP', 'DEF')):
            b = rng.random()
            if b < 0.28:
                pts += 3
            elif b < 0.50:
                pts += 2
            elif b < 0.70:
                pts += 1

        # yellow card
        if rng.random() < player_ctx['p_yellow']:
            pts -= 1

        results.append(pts)

    results.sort()
    n = len(results)
    return {
        'mean': round(sum(results) / n, 2),
        'floor_p20': results[int(0.20 * n)],
        'median': results[int(0.50 * n)],
        'ceiling_p90': results[int(0.90 * n)],
        'p_blank': round(sum(1 for x in results if x <= 2) / n, 3),
        'p_haul': round(sum(1 for x in results if x >= 10) / n, 3),
    }


def _poisson_draw(rng, lam):
    """Knuth's algorithm for sampling from a Poisson distribution."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1
