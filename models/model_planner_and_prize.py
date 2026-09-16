"""
Model 2 — Multi-gameweek transfer planner.
Model 3 — League prize-EV / variance-posture model.

MODEL 2: TRANSFER PLANNER
-------------------------
The right question is never "is player B better than player A this week?" but
"does swapping A for B gain enough xPts across the HORIZON to justify the
transfer and any hit?" — and, for Dennis, the horizon that matters is the
MONTH (FPL's official phase), because that's what the prize pays on.

Key rules encoded:
 - A hit costs 4 points, charged once, against the whole horizon's gain.
 - A free transfer has an option value: spending it now means not having it
   later. Rolling banks it (max 5).
 - Compare complete routes, not single slots.

MODEL 3: PRIZE-EV / VARIANCE POSTURE
------------------------------------
Unique to The Regulars. Given:
  - current monthly standings
  - gameweeks left in the month
  - the 50/30/20 split of a $160 pot, with ties pooling and splitting evenly
...it answers: how much variance should Dennis take THIS week, in dollars?

Chasing from outside the money -> maximise ceiling (Frontier "maverick" logic)
Sitting inside the money       -> maximise floor (protect the paying position)
"""

import itertools
import statistics


# =========================================================
# MODEL 2 — MULTI-GAMEWEEK TRANSFER PLANNER
# =========================================================

def evaluate_transfer(out_player_xpts_by_gw, in_player_xpts_by_gw,
                      hit_cost=0, ft_option_value=1.0):
    """
    out_/in_player_xpts_by_gw: list of xPts per gameweek over the horizon.
    hit_cost: 4 per extra transfer beyond free ones, else 0.
    ft_option_value: implied value of keeping a free transfer in the bank
                     (empirically ~1-1.5 pts; set 0 if transfers are abundant).

    Returns net expected gain over the horizon. Positive = worth doing.
    """
    gain = sum(in_player_xpts_by_gw) - sum(out_player_xpts_by_gw)
    net = gain - hit_cost - ft_option_value
    return {
        'raw_gain': round(gain, 2),
        'hit_cost': hit_cost,
        'ft_option_value': ft_option_value,
        'net_gain': round(net, 2),
        'verdict': 'DO IT' if net > 0 else ('MARGINAL' if net > -1 else 'HOLD'),
        'gw_by_gw': [round(i - o, 2) for i, o in
                     zip(in_player_xpts_by_gw, out_player_xpts_by_gw)],
    }


def plan_horizon(squad_xpts_by_gw, candidates_by_gw, free_transfers,
                 horizon_gws, max_hits=1):
    """
    Greedy search over single- and double-transfer routes across the horizon.
    squad_xpts_by_gw:    {player_name: [xpts per gw]}
    candidates_by_gw:    {player_name: [xpts per gw]} for possible incomings
    Returns ranked routes by net gain.
    """
    routes = []
    for out_name, out_xpts in squad_xpts_by_gw.items():
        for in_name, in_xpts in candidates_by_gw.items():
            hit = 0 if free_transfers >= 1 else 4
            ev = evaluate_transfer(out_xpts[:horizon_gws],
                                   in_xpts[:horizon_gws],
                                   hit_cost=hit,
                                   ft_option_value=1.0 if free_transfers <= 1 else 0.5)
            routes.append({'out': out_name, 'in': in_name, **ev})
    routes.sort(key=lambda r: -r['net_gain'])
    return routes


# =========================================================
# MODEL 3 — PRIZE-EV / VARIANCE POSTURE
# =========================================================

POT = 160.0
SPLIT = [0.50, 0.30, 0.20]   # 1st / 2nd / 3rd


def prize_for_positions(positions_occupied):
    """
    Ties pool the prize money for every position the tied group spans,
    then split it evenly. positions_occupied: list of 0-indexed places.
    """
    pooled = sum(SPLIT[p] for p in positions_occupied if p < len(SPLIT))
    return POT * pooled / len(positions_occupied)


def settle_month(scores_by_manager):
    """
    scores_by_manager: {name: monthly_points}
    Returns {name: prize_won} applying the tie-pooling rule.
    """
    ordered = sorted(scores_by_manager.items(), key=lambda kv: -kv[1])
    payouts = {n: 0.0 for n in scores_by_manager}
    place = 0
    i = 0
    while i < len(ordered) and place < len(SPLIT):
        score = ordered[i][1]
        tied = [n for n, s in ordered if s == score]
        positions = list(range(place, place + len(tied)))
        if place < len(SPLIT):
            share = prize_for_positions(positions)
            for n in tied:
                payouts[n] = share
        place += len(tied)
        i += len(tied)
    return payouts


def variance_posture(my_current, rivals_current, gws_left,
                     my_mean_per_gw=55, sd_per_gw=18, n_sims=20000, seed=7):
    """
    Simulates the rest of the month to estimate:
      - P(finishing in the money) under a NEUTRAL strategy
      - the marginal $ value of adding variance vs adding mean

    Returns guidance on whether to chase ceiling or protect floor.
    """
    import random
    rng = random.Random(seed)

    def sim(extra_mean=0.0, extra_sd=0.0):
        total_prize = 0.0
        for _ in range(n_sims):
            scores = {'ME': my_current + sum(
                rng.gauss(my_mean_per_gw + extra_mean, sd_per_gw + extra_sd)
                for _ in range(gws_left))}
            for name, cur in rivals_current.items():
                scores[name] = cur + sum(
                    rng.gauss(my_mean_per_gw, sd_per_gw) for _ in range(gws_left))
            total_prize += settle_month(scores)['ME']
        return total_prize / n_sims

    baseline = sim()
    more_variance = sim(extra_sd=8)      # maverick picks: same mean, fatter tails
    more_mean = sim(extra_mean=3)        # safe upgrade: +3 pts/gw, same spread

    return {
        'expected_prize_neutral': round(baseline, 2),
        'expected_prize_high_variance': round(more_variance, 2),
        'expected_prize_higher_mean': round(more_mean, 2),
        'variance_is_worth': round(more_variance - baseline, 2),
        'mean_is_worth': round(more_mean - baseline, 2),
        'recommended_posture': (
            'CHASE CEILING (differentials/maverick picks)'
            if more_variance - baseline > more_mean - baseline
            else 'PROTECT FLOOR (safe, high-mean picks)'),
    }
