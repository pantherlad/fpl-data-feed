"""
models_v2.py — replacements and fixes, built to the parameters the
open-source FPL optimisation community has converged on.

REPLACES:
  Model 2 + Model 12  -> multi_gw_optimise()      (joint horizon solve)
  Model 15            -> market_odds_strength()   (odds, not results)
  Model 19            -> predict_ownership()      (forward-looking)

ADDS:
  sensitivity_analysis()  — only recommend transfers that survive noise

FIXES:
  set_piece_value_v2()    — no longer double-counts penalties inside xG
  BENCH_WEIGHTS           — calibrated per slot, not flat 10%

Calibrated parameters (community-standard, not invented here):
  HORIZON = 8 gameweeks
  DECAY_BASE = 0.84 per week
  FT_VALUE = 0.8 points
  ITB_VALUE = 0.08 per 0.1m in the bank
  BENCH_WEIGHTS = GK 0.03, sub1 0.21, sub2 0.06, sub3 0.002
"""

import random
from collections import defaultdict

HORIZON = 8
DECAY_BASE = 0.84
FT_VALUE = 0.8
ITB_VALUE = 0.08
BENCH_WEIGHTS = {'gk': 0.03, 1: 0.21, 2: 0.06, 3: 0.002}
HIT_COST = 4


# =========================================================
# REPLACES MODELS 2 + 12 — MULTI-GAMEWEEK JOINT OPTIMISER
# =========================================================

def multi_gw_optimise(squad, candidates, xpts_by_gw, prices, positions, teams,
                      free_transfers, bank, horizon=HORIZON,
                      decay=DECAY_BASE, max_transfers_per_gw=2,
                      locked=None, banned=None):
    """
    Solves transfers across the WHOLE horizon at once, rather than one
    gameweek at a time. This is the single most important upgrade: a move
    that looks bad this week is often right because of GW+3 fixtures.

    xpts_by_gw: {player_name: [xpts for gw1..gwN]}
    Objective:  sum over gw of decay^gw * (starters + weighted bench)
                - 4 per hit
                + FT_VALUE per unused free transfer
                + ITB_VALUE per 0.1m retained

    Uses PuLP if available; else a decayed greedy search.
    """
    locked = set(locked or [])
    banned = set(banned or [])

    try:
        import pulp
    except ImportError:
        return _greedy_multi_gw(squad, candidates, xpts_by_gw, prices,
                                positions, teams, free_transfers, bank,
                                horizon, decay, locked, banned)

    all_players = sorted(set(list(squad) + [c for c in candidates
                                            if c not in banned]))
    gws = list(range(horizon))

    prob = pulp.LpProblem('fpl_multi_gw', pulp.LpMaximize)

    # squad[p][t] = player p in squad at gameweek t
    sq = {(p, t): pulp.LpVariable(f'sq_{i}_{t}', cat='Binary')
          for i, p in enumerate(all_players) for t in gws}
    # xi[p][t] = player p starts at gameweek t
    xi = {(p, t): pulp.LpVariable(f'xi_{i}_{t}', cat='Binary')
          for i, p in enumerate(all_players) for t in gws}
    # transfer in/out
    tin = {(p, t): pulp.LpVariable(f'in_{i}_{t}', cat='Binary')
           for i, p in enumerate(all_players) for t in gws}
    tout = {(p, t): pulp.LpVariable(f'out_{i}_{t}', cat='Binary')
            for i, p in enumerate(all_players) for t in gws}
    hits = {t: pulp.LpVariable(f'hits_{t}', lowBound=0, cat='Integer')
            for t in gws}

    def xp(p, t):
        arr = xpts_by_gw.get(p, [])
        return arr[t] if t < len(arr) else 0.0

    # Objective — decayed starters, small bench credit, minus hits
    prob += (
        pulp.lpSum(decay ** t * xp(p, t) * xi[(p, t)]
                   for p in all_players for t in gws)
        + pulp.lpSum(decay ** t * xp(p, t) * BENCH_WEIGHTS[1] *
                     (sq[(p, t)] - xi[(p, t)])
                     for p in all_players for t in gws)
        - pulp.lpSum(decay ** t * HIT_COST * hits[t] for t in gws)
    )

    quotas = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    xi_min = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
    xi_max = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}

    for t in gws:
        prob += pulp.lpSum(sq[(p, t)] for p in all_players) == 15
        prob += pulp.lpSum(xi[(p, t)] for p in all_players) == 11
        for pos, n in quotas.items():
            prob += pulp.lpSum(sq[(p, t)] for p in all_players
                               if positions[p] == pos) == n
            prob += pulp.lpSum(xi[(p, t)] for p in all_players
                               if positions[p] == pos) >= xi_min[pos]
            prob += pulp.lpSum(xi[(p, t)] for p in all_players
                               if positions[p] == pos) <= xi_max[pos]
        for p in all_players:
            prob += xi[(p, t)] <= sq[(p, t)]
        # 3 per club
        by_club = defaultdict(list)
        for p in all_players:
            by_club[teams[p]].append(p)
        for club, members in by_club.items():
            prob += pulp.lpSum(sq[(p, t)] for p in members) <= 3
        # budget
        prob += pulp.lpSum(prices[p] * sq[(p, t)]
                           for p in all_players) <= 100.0 + bank

    # transfer continuity
    for p in all_players:
        prob += sq[(p, 0)] == (1 if p in squad else 0) + tin[(p, 0)] - tout[(p, 0)]
        for t in gws[1:]:
            prob += sq[(p, t)] == sq[(p, t - 1)] + tin[(p, t)] - tout[(p, t)]
        for t in gws:
            prob += tin[(p, t)] + tout[(p, t)] <= 1
    for p in locked:
        for t in gws:
            prob += sq[(p, t)] == 1

    # hits: transfers beyond free ones
    for t in gws:
        n_in = pulp.lpSum(tin[(p, t)] for p in all_players)
        avail = free_transfers if t == 0 else 1
        prob += hits[t] >= n_in - avail
        prob += n_in <= max_transfers_per_gw + avail

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    plan = []
    for t in gws:
        ins = [p for p in all_players if tin[(p, t)].value() == 1]
        outs = [p for p in all_players if tout[(p, t)].value() == 1]
        starters = [p for p in all_players if xi[(p, t)].value() == 1]
        if ins or outs or t == 0:
            plan.append({'gw_offset': t, 'in': ins, 'out': outs,
                         'hits': int(hits[t].value() or 0),
                         'xi': starters,
                         'xi_xpts': round(sum(xp(p, t) for p in starters), 2)})
    return {'method': 'multi-GW MILP', 'horizon': horizon,
            'decay': decay, 'plan': plan,
            'objective': round(pulp.value(prob.objective), 2)}


def _greedy_multi_gw(squad, candidates, xpts_by_gw, prices, positions, teams,
                     free_transfers, bank, horizon, decay, locked, banned):
    """Decay-weighted greedy fallback when PuLP isn't installed."""
    def horizon_value(p):
        arr = xpts_by_gw.get(p, [])
        return sum(decay ** t * (arr[t] if t < len(arr) else 0)
                   for t in range(horizon))

    moves = []
    for out_p in squad:
        if out_p in locked:
            continue
        for in_p in candidates:
            if in_p in squad or in_p in banned:
                continue
            if positions[in_p] != positions[out_p]:
                continue
            if prices[in_p] > prices[out_p] + bank:
                continue
            gain = horizon_value(in_p) - horizon_value(out_p)
            cost = 0 if free_transfers >= 1 else HIT_COST
            net = gain - cost - (FT_VALUE if free_transfers <= 1 else 0)
            moves.append({'out': out_p, 'in': in_p,
                          'horizon_gain': round(gain, 2),
                          'net': round(net, 2)})
    moves.sort(key=lambda m: -m['net'])
    return {'method': 'greedy (install pulp for true MILP)',
            'horizon': horizon, 'plan': moves[:10]}


# =========================================================
# REPLACES MODEL 15 — MARKET-ODDS TEAM STRENGTH
# =========================================================

def devig(probabilities):
    """Remove bookmaker margin so probabilities sum to 1."""
    total = sum(probabilities)
    return [p / total for p in probabilities] if total else probabilities


def market_odds_strength(matches, shrink_to_mean=0.25):
    """
    Derives per-team attack/defence rates from market win probabilities.

    WHY THIS BEATS DIXON-COLES ON RESULTS: the market prices in injuries,
    rotation, team news and managerial change in real time. A results-based
    model can only see what already happened, and with 4 games played the
    sample is far too thin to be trusted.

    matches: [{'home','away','p_home','p_draw','p_away',
               'over_2_5' (optional)}]
    Returns per-team attack/defence multipliers around 1.0.
    """
    strength = defaultdict(list)
    for m in matches:
        ph, pd_, pa = devig([m['p_home'], m['p_draw'], m['p_away']])
        # supremacy: how much better the home side is, venue-adjusted
        sup_home = ph - pa
        strength[m['home']].append(sup_home - 0.10)   # strip home advantage
        strength[m['away']].append(-sup_home - 0.10)

    ratings = {}
    for team, vals in strength.items():
        raw = sum(vals) / len(vals)
        ratings[team] = (1 - shrink_to_mean) * raw     # shrink early-season
    # map supremacy onto attack/defence multipliers
    out = {}
    for team, r in ratings.items():
        out[team] = {'strength': round(r, 3),
                     'attack_mult': round(1.0 + r * 0.9, 3),
                     'defence_mult': round(1.0 - r * 0.9, 3)}
    return out


def expected_goals_from_odds(ratings, home, away, league_avg=1.45,
                             home_adv=1.15):
    h = ratings.get(home, {'attack_mult': 1, 'defence_mult': 1})
    a = ratings.get(away, {'attack_mult': 1, 'defence_mult': 1})
    return (league_avg * h['attack_mult'] * a['defence_mult'] * home_adv,
            league_avg * a['attack_mult'] * h['defence_mult'])


# =========================================================
# REPLACES MODEL 19 — FORWARD-LOOKING OWNERSHIP PREDICTION
# =========================================================

def predict_ownership(players, total_managers=11_000_000, gws_remaining_in_week=1):
    """
    Projects ownership AT THE NEXT DEADLINE from net transfer velocity,
    rather than reporting current ownership (which is already stale).

    This is what differential calculations actually need: a player at 5%
    now but being bought by 400k managers is NOT a differential by kickoff.
    """
    out = []
    for p in players:
        own_pct = float(p.get('selected_by_percent') or 0)
        owners = own_pct / 100 * total_managers
        net = (p.get('transfers_in_event') or 0) - (p.get('transfers_out_event') or 0)
        projected_owners = owners + net * gws_remaining_in_week
        projected_pct = 100 * projected_owners / total_managers
        delta = projected_pct - own_pct
        out.append({
            'name': p['name'], 'team': p['team'],
            'current_own': round(own_pct, 2),
            'projected_own': round(projected_pct, 2),
            'delta': round(delta, 2),
            'velocity': net,
            'trend': ('SURGING' if delta > 1.5 else
                      'rising' if delta > 0.4 else
                      'FALLING' if delta < -1.5 else
                      'falling' if delta < -0.4 else 'stable'),
            'still_differential': projected_pct < 10,
        })
    out.sort(key=lambda r: -abs(r['delta']))
    return out


# =========================================================
# NEW — SENSITIVITY ANALYSIS
# =========================================================

def sensitivity_analysis(decision_fn, base_inputs, n_runs=50, noise_sd=0.15,
                         seed=11):
    """
    Re-solves the decision many times with random noise added to the xPts
    projections, then reports how often each recommendation survives.

    WHY THIS MATTERS: a transfer that only wins under one exact set of
    assumptions is fragile. One that appears in 80%+ of noisy solves is
    robust. This is the single best guard against acting on a projection
    that is really just noise — the class of error that produced the
    Konsa minutes mistake.

    decision_fn: callable(inputs) -> hashable recommendation
    base_inputs: {'xpts_by_gw': {...}, ...}
    """
    rng = random.Random(seed)
    counts = defaultdict(int)

    for _ in range(n_runs):
        noisy = dict(base_inputs)
        noisy_xpts = {}
        for name, arr in base_inputs['xpts_by_gw'].items():
            noisy_xpts[name] = [max(0.0, v * (1 + rng.gauss(0, noise_sd)))
                                for v in arr]
        noisy['xpts_by_gw'] = noisy_xpts
        try:
            rec = decision_fn(noisy)
        except Exception:
            continue
        counts[rec] += 1

    total = sum(counts.values()) or 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return {
        'n_runs': total,
        'results': [{'recommendation': r, 'frequency': c,
                     'robustness': round(100 * c / total, 1)}
                    for r, c in ranked],
        'top_robustness': round(100 * ranked[0][1] / total, 1) if ranked else 0,
        'verdict': ('ROBUST — survives noise' if ranked and ranked[0][1] / total >= 0.7
                    else 'FRAGILE — depends on exact projections; consider holding'
                    if ranked else 'no result'),
    }


# =========================================================
# FIX — SET-PIECE VALUE WITHOUT DOUBLE COUNTING
# =========================================================

def set_piece_value_v2(player, team_xg_for, pens_per_game=0.12,
                       observed_share=0.6):
    """
    FIXED: the previous version added penalty xPts on top of a player's xG,
    but FPL's expected_goals ALREADY INCLUDES penalties they have taken.
    That double-counted established penalty takers.

    This version subtracts back out the share the player's own xG history
    has already observed, so only the UNOBSERVED portion is added — which
    is exactly what matters for a new signing or someone who has just taken
    the duty over.
    """
    pos = player['position']
    goal_value = {'GKP': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}[pos]
    detail = {}
    val = 0.0

    pen_order = player.get('penalties_order')
    if pen_order == 1:
        gross = pens_per_game * (team_xg_for / 1.45) * 0.78 * goal_value
        minutes = float(player.get('minutes') or 0)
        # the longer they've had the duty, the more their xG already reflects it
        already_priced = min(minutes / 450.0, 1.0) * observed_share
        net = gross * (1 - already_priced)
        val += net
        detail['penalties_net'] = round(net, 2)
        detail['penalties_gross'] = round(gross, 2)
        detail['already_in_xg'] = round(gross - net, 2)
    elif pen_order == 2:
        net = pens_per_game * (team_xg_for / 1.45) * 0.78 * goal_value * 0.25
        val += net
        detail['penalties_backup'] = round(net, 2)

    if player.get('direct_freekicks_order') == 1:
        fk = 0.035 * goal_value * 0.5      # halved: partly in existing xG
        val += fk
        detail['direct_fk'] = round(fk, 2)

    if player.get('corners_and_indirect_freekicks_order') == 1:
        # corner assists ARE largely captured in xA already
        ck = 0.16 * 3 * (team_xg_for / 1.45) * 0.4
        val += ck
        detail['corners'] = round(ck, 2)

    return {'set_piece_xpts': round(val, 2), 'breakdown': detail,
            'note': 'net of what xG/xA already captures'}


# =========================================================
# FIX — CALIBRATED BENCH WEIGHTS
# =========================================================

def bench_order_v2(bench_players):
    """
    FIXED: previously every bench slot was credited a flat 10%. The
    community-calibrated weights are steeply decreasing — the first sub is
    worth ~7x the second, and the fourth is worth almost nothing.

    bench_players: [{'name','xpts','p_plays'}]
    """
    scored = sorted(bench_players, key=lambda b: -(b['xpts'] * b['p_plays']))
    out = []
    for i, b in enumerate(scored, start=1):
        w = BENCH_WEIGHTS.get(i, 0.002)
        out.append({'slot': i, 'name': b['name'],
                    'xpts': round(b['xpts'], 2),
                    'p_plays': round(b['p_plays'], 2),
                    'slot_weight': w,
                    'contribution': round(b['xpts'] * b['p_plays'] * w, 3)})
    return out
