"""
Models 12-21 — built from the academic literature.

Key research findings these implement:
 - MILP/integer programming is the state of the art for squad selection
   (one published approach placed top 4% worldwide retrospectively).
 - Position-specific models consistently outperform one general model.
 - Robust optimisation (planning for uncertainty) beats deterministic.
 - Dixon-Coles style team ratings with time decay beat crude FDR.
"""

import math
from collections import defaultdict


# =========================================================
# 12. MILP SQUAD OPTIMISER
# =========================================================

def optimise_squad(players, budget=100.0, horizon_xpts_key='xpts_horizon',
                   formation_min=(1, 3, 3, 1), formation_max=(1, 5, 5, 3),
                   max_per_club=3, locked_in=None, excluded=None):
    """
    Selects the optimal 15 (and best XI) under FPL's constraints:
      - 2 GKP, 5 DEF, 5 MID, 3 FWD in the squad
      - max 3 per club
      - total price <= budget
      - starting XI must satisfy formation limits
    Maximises total projected points over the horizon.

    Uses PuLP if available (true MILP); otherwise falls back to a greedy
    value-ranked heuristic, which is near-optimal for this problem size.
    """
    locked_in = set(locked_in or [])
    excluded = set(excluded or [])
    pool = [p for p in players if p['name'] not in excluded]

    try:
        import pulp
        return _milp_solve(pool, budget, horizon_xpts_key, max_per_club, locked_in)
    except ImportError:
        return _greedy_solve(pool, budget, horizon_xpts_key, max_per_club, locked_in)


def _milp_solve(pool, budget, key, max_per_club, locked_in):
    import pulp
    prob = pulp.LpProblem('fpl_squad', pulp.LpMaximize)
    x = {p['name']: pulp.LpVariable(f"s_{i}", cat='Binary')
         for i, p in enumerate(pool)}          # in squad
    y = {p['name']: pulp.LpVariable(f"x_{i}", cat='Binary')
         for i, p in enumerate(pool)}          # in starting XI

    # objective: starters full weight, bench at 10% (they rarely play)
    prob += pulp.lpSum(p[key] * (0.10 * x[p['name']] + 0.90 * y[p['name']])
                       for p in pool)

    prob += pulp.lpSum(x[p['name']] for p in pool) == 15
    prob += pulp.lpSum(y[p['name']] for p in pool) == 11
    prob += pulp.lpSum(p['price'] * x[p['name']] for p in pool) <= budget

    for p in pool:
        prob += y[p['name']] <= x[p['name']]          # can't start if not in squad
    for name in locked_in:
        if name in x:
            prob += x[name] == 1

    quotas = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for pos, n in quotas.items():
        prob += pulp.lpSum(x[p['name']] for p in pool if p['position'] == pos) == n

    xi_min = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
    xi_max = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for pos in quotas:
        prob += pulp.lpSum(y[p['name']] for p in pool if p['position'] == pos) >= xi_min[pos]
        prob += pulp.lpSum(y[p['name']] for p in pool if p['position'] == pos) <= xi_max[pos]

    clubs = defaultdict(list)
    for p in pool:
        clubs[p['team']].append(p['name'])
    for team, names in clubs.items():
        prob += pulp.lpSum(x[n] for n in names) <= max_per_club

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    squad = [p for p in pool if x[p['name']].value() == 1]
    xi = [p for p in pool if y[p['name']].value() == 1]
    return {'method': 'MILP (optimal)', 'squad': squad, 'starting_xi': xi,
            'total_cost': round(sum(p['price'] for p in squad), 1),
            'projected': round(sum(p[key] for p in xi), 2)}


def _greedy_solve(pool, budget, key, max_per_club, locked_in):
    quotas = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    ranked = sorted(pool, key=lambda p: -(p[key] / max(p['price'], 0.1)))
    squad, spend = [], 0.0
    club_count = defaultdict(int)
    filled = defaultdict(int)
    for name in locked_in:
        p = next((q for q in pool if q['name'] == name), None)
        if p:
            squad.append(p); spend += p['price']
            club_count[p['team']] += 1; filled[p['position']] += 1
    for p in ranked:
        if p in squad:
            continue
        pos = p['position']
        if filled[pos] >= quotas[pos] or club_count[p['team']] >= max_per_club:
            continue
        if spend + p['price'] > budget:
            continue
        squad.append(p); spend += p['price']
        club_count[p['team']] += 1; filled[pos] += 1
        if len(squad) == 15:
            break
    xi = _best_xi(squad, key)
    return {'method': 'greedy heuristic (install pulp for true MILP)',
            'squad': squad, 'starting_xi': xi,
            'total_cost': round(spend, 1),
            'projected': round(sum(p[key] for p in xi), 2)}


def _best_xi(squad, key):
    """Best legal XI from a 15: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD."""
    by = defaultdict(list)
    for p in squad:
        by[p['position']].append(p)
    for pos in by:
        by[pos].sort(key=lambda p: -p[key])
    best, best_pts = None, -1
    for ndef in range(3, 6):
        for nmid in range(2, 6):
            nfwd = 10 - ndef - nmid
            if not (1 <= nfwd <= 3):
                continue
            if len(by['DEF']) < ndef or len(by['MID']) < nmid or len(by['FWD']) < nfwd:
                continue
            xi = by['GKP'][:1] + by['DEF'][:ndef] + by['MID'][:nmid] + by['FWD'][:nfwd]
            pts = sum(p[key] for p in xi)
            if pts > best_pts:
                best, best_pts = xi, pts
    return best or []


# =========================================================
# 13. ROBUST / REGRET OPTIMISER
# =========================================================

def robust_choice(options, scenarios, risk_posture='neutral'):
    """
    options:   {option_name: {scenario_name: points}}
    Picks by posture rather than raw mean:
      'chase'   -> maximax (best-case), for when behind in a month
      'protect' -> maximin (worst-case), for defending a paying position
      'neutral' -> minimise maximum regret
    """
    names = list(options)
    results = {}
    for n in names:
        vals = [options[n][s] for s in scenarios]
        results[n] = {'mean': sum(vals) / len(vals),
                      'best': max(vals), 'worst': min(vals)}
    # regret
    for s in scenarios:
        best_s = max(options[n][s] for n in names)
        for n in names:
            results[n].setdefault('max_regret', 0)
            results[n]['max_regret'] = max(results[n]['max_regret'],
                                           best_s - options[n][s])
    if risk_posture == 'chase':
        pick = max(names, key=lambda n: results[n]['best'])
    elif risk_posture == 'protect':
        pick = max(names, key=lambda n: results[n]['worst'])
    else:
        pick = min(names, key=lambda n: results[n]['max_regret'])
    return {'recommended': pick, 'posture': risk_posture, 'detail': results}


# =========================================================
# 14. POSITION-SPECIFIC xPTS WEIGHTS
# =========================================================

# Research finding: one model across all positions systematically misprices
# defenders vs forwards. These are position-specific priors/weights applied
# inside the xPts model rather than a single shared assumption.
POSITION_PARAMS = {
    'GKP': {'base_xg90': 0.00, 'base_xa90': 0.00, 'cs_weight': 4,
            'defcon_threshold': None, 'save_rate_per_goal': 2.6,
            'bonus_bias': 0.9, 'minutes_stability': 0.95},
    'DEF': {'base_xg90': 0.06, 'base_xa90': 0.08, 'cs_weight': 4,
            'defcon_threshold': 10, 'save_rate_per_goal': 0,
            'bonus_bias': 1.0, 'minutes_stability': 0.88},
    'MID': {'base_xg90': 0.18, 'base_xa90': 0.16, 'cs_weight': 1,
            'defcon_threshold': 12, 'save_rate_per_goal': 0,
            'bonus_bias': 1.1, 'minutes_stability': 0.80},
    'FWD': {'base_xg90': 0.38, 'base_xa90': 0.13, 'cs_weight': 0,
            'defcon_threshold': 12, 'save_rate_per_goal': 0,
            'bonus_bias': 1.2, 'minutes_stability': 0.78},
}


# =========================================================
# 15. DIXON-COLES TEAM RATINGS (time-decayed)
# =========================================================

def dixon_coles_ratings(results, decay=0.005, iterations=60):
    """
    results: list of {'home','away','hg','ag','days_ago'}
    Estimates per-team attack and defence strengths with exponential time
    decay, plus home advantage — a proper replacement for FPL's 1-5 FDR.
    Fitted by simple iterative scaling (lightweight alternative to full MLE).
    """
    teams = sorted({r['home'] for r in results} | {r['away'] for r in results})
    atk = {t: 1.0 for t in teams}
    dfn = {t: 1.0 for t in teams}
    home_adv = 1.15

    def w(r):
        return math.exp(-decay * r.get('days_ago', 0))

    for _ in range(iterations):
        goals_for = defaultdict(float); exp_for = defaultdict(float)
        goals_ag = defaultdict(float); exp_ag = defaultdict(float)
        for r in results:
            ww = w(r)
            h, a = r['home'], r['away']
            lam_h = atk[h] * dfn[a] * home_adv
            lam_a = atk[a] * dfn[h]
            goals_for[h] += ww * r['hg']; exp_for[h] += ww * lam_h
            goals_for[a] += ww * r['ag']; exp_for[a] += ww * lam_a
            goals_ag[a] += ww * r['hg']; exp_ag[a] += ww * lam_h
            goals_ag[h] += ww * r['ag']; exp_ag[h] += ww * lam_a
        for t in teams:
            if exp_for[t] > 0:
                atk[t] *= (goals_for[t] / exp_for[t]) ** 0.5
            if exp_ag[t] > 0:
                dfn[t] *= (goals_ag[t] / exp_ag[t]) ** 0.5
        m = sum(atk.values()) / len(atk)
        atk = {t: v / m for t, v in atk.items()}
        dfn = {t: v / m for t, v in dfn.items()}
    return {'attack': atk, 'defence': dfn, 'home_advantage': home_adv}


def expected_goals_dc(ratings, home_team, away_team):
    a, d, ha = ratings['attack'], ratings['defence'], ratings['home_advantage']
    return (a[home_team] * d[away_team] * ha, a[away_team] * d[home_team])


# =========================================================
# 16. SET-PIECE / PENALTY VALUE MODEL
# =========================================================

def set_piece_value(player, team_xg_for, pens_per_game=0.12):
    """
    Quantifies the xPts a player gets purely from set-piece duty — one of the
    most reliably mispriced edges in FPL.
    """
    pos = player['position']
    goal_value = {'GKP': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}[pos]
    val = 0.0
    detail = {}

    if player.get('penalties_order') == 1:
        # P(team wins pen) x P(scored) x goal value
        pen_xp = pens_per_game * (team_xg_for / 1.45) * 0.78 * goal_value
        val += pen_xp; detail['penalties'] = round(pen_xp, 2)
    elif player.get('penalties_order') == 2:
        pen_xp = pens_per_game * (team_xg_for / 1.45) * 0.78 * goal_value * 0.25
        val += pen_xp; detail['penalties_backup'] = round(pen_xp, 2)

    if player.get('direct_freekicks_order') == 1:
        fk = 0.035 * goal_value
        val += fk; detail['direct_fk'] = round(fk, 2)

    if player.get('corners_and_indirect_freekicks_order') == 1:
        # corners mostly generate assists
        ck = 0.16 * 3 * (team_xg_for / 1.45)
        val += ck; detail['corners'] = round(ck, 2)

    return {'set_piece_xpts': round(val, 2), 'breakdown': detail}


# =========================================================
# 17. TACTICAL / ROLE-CHANGE DETECTOR
# =========================================================

def detect_role_change(gw_history, recent_n=3, hard_recent_n=2):
    """
    gw_history: list of per-GW dicts for ONE player with keys
                gw, minutes, starts, position, penalties_order, xg, xa
    Flags step-changes in role — minutes, set-piece duty, or attacking output
    per 90 — which historical averages hide.
    """
    if len(gw_history) < recent_n + 1:
        return {'role_change': False, 'reason': 'insufficient history'}

    recent = gw_history[-recent_n:]
    earlier = gw_history[:-recent_n]

    r_mins = sum(g['minutes'] for g in recent) / len(recent)
    e_mins = sum(g['minutes'] for g in earlier) / len(earlier)

    # HARD recency check: if the last `hard_recent_n` games were ALL full
    # starts (>=75 min), treat as nailed regardless of the wider window.
    # A single substitute cameo right before a role change (e.g. the game
    # a player broke into the side) shouldn't dilute a now-clear pattern.
    hard_recent = gw_history[-hard_recent_n:]
    just_became_nailed = (len(hard_recent) == hard_recent_n and
                          all(g['minutes'] >= 75 for g in hard_recent))

    flags = []
    if r_mins - e_mins >= 25:
        flags.append(f'minutes UP ({e_mins:.0f} -> {r_mins:.0f})')
    if e_mins - r_mins >= 25:
        flags.append(f'minutes DOWN ({e_mins:.0f} -> {r_mins:.0f})')

    r_pen = recent[-1].get('penalties_order')
    e_pen = earlier[-1].get('penalties_order')
    if r_pen != e_pen:
        flags.append(f'penalty order {e_pen} -> {r_pen}')

    if recent[-1].get('position') != earlier[-1].get('position'):
        flags.append(f"position {earlier[-1].get('position')} -> {recent[-1].get('position')}")

    if just_became_nailed and not flags:
        flags.append(f'last {hard_recent_n} games both 75+ mins — now nailed')

    return {'role_change': bool(flags), 'flags': flags,
            'recent_mins_avg': round(r_mins, 1), 'earlier_mins_avg': round(e_mins, 1),
            'just_became_nailed': just_became_nailed,
            'guidance': ('Weight RECENT games heavily — this is a step change, '
                         'not noise.' if flags else 'No role change; use season data.')}


# =========================================================
# 18. INJURY-RETURN / MINUTES-RAMP MODEL
# =========================================================

def injury_ramp_discount(gws_since_return, typical_ramp=3):
    """
    Returning players are systematically overvalued for their first 2-3 weeks:
    managers price in pre-injury output while the player gets 45-70 mins.
    Returns a multiplier to apply to expected minutes.
    """
    if gws_since_return is None:
        return 1.0
    if gws_since_return <= 0:
        return 0.45
    if gws_since_return >= typical_ramp:
        return 1.0
    return 0.45 + (0.55 * gws_since_return / typical_ramp)


# =========================================================
# 19. NEWS-SENTIMENT SCORER
# =========================================================

NEGATIVE_CUES = ['doubt', 'knock', 'assess', 'scan', 'limped', 'strain', 'rested',
                 'illness', 'ill', 'fatigue', 'managed', 'substituted early',
                 'not risk', 'late test', 'suspended', 'ban']
POSITIVE_CUES = ['back in training', 'available', 'fit', 'returned', 'no issue',
                 'full training', 'cleared', 'starts']


def score_news(text):
    """
    Lightweight sentiment scorer for FPL 'news' fields and press-conference
    quotes. Research (transformer sentiment paper) found press-conference
    language carries signal beyond the stats — this is a simple, inspectable
    version of that idea, used to nudge expected minutes.
    """
    if not text:
        return {'score': 0.0, 'signal': 'none', 'minutes_multiplier': 1.0}
    t = text.lower()
    neg = sum(1 for c in NEGATIVE_CUES if c in t)
    pos = sum(1 for c in POSITIVE_CUES if c in t)
    score = pos - neg
    if score <= -2:
        mult, sig = 0.45, 'strong negative'
    elif score == -1:
        mult, sig = 0.75, 'negative'
    elif score >= 2:
        mult, sig = 1.05, 'positive'
    elif score == 1:
        mult, sig = 1.0, 'mild positive'
    else:
        mult, sig = 0.9, 'ambiguous'
    return {'score': score, 'signal': sig, 'minutes_multiplier': mult}


# =========================================================
# 20. OPPONENT-ADJUSTED xG
# =========================================================

def opponent_adjusted_xg(player_match_xg, opponent_defence_rating,
                         league_avg_defence=1.0):
    """
    2.0 xG against Arsenal is worth far more than 2.0 against Coventry.
    Normalises raw xG by the quality of the defence faced, so the resulting
    rate is comparable across fixtures.
    """
    if opponent_defence_rating <= 0:
        return player_match_xg
    return player_match_xg * (opponent_defence_rating / league_avg_defence)


def adjusted_xg_series(matches, defence_ratings):
    """matches: [{'xg':.., 'opponent':..}] -> opponent-adjusted average."""
    vals = [opponent_adjusted_xg(m['xg'], defence_ratings.get(m['opponent'], 1.0))
            for m in matches]
    return {'raw_avg': round(sum(m['xg'] for m in matches) / len(matches), 3),
            'adjusted_avg': round(sum(vals) / len(vals), 3),
            'n': len(matches)} if matches else {}


# =========================================================
# 21. SEASON-LONG CHIP SEQUENCING
# =========================================================

def sequence_chips(chip_window_scores, chips_available, phase_targets=None,
                   min_gap=2):
    """
    Optimises the ORDER and SPACING of all remaining chips jointly, rather
    than picking each one's best week independently (which often collides).

    phase_targets: optional {month_name: weight} to bias chips toward months
                   Dennis actually has a chance of winning.
    """
    phase_targets = phase_targets or {}
    key_for = {'bench_boost': 'bench_boost_score',
               'triple_captain': 'triple_captain_score',
               'free_hit': 'free_hit_score'}

    # score every (chip, gw) pair, weighted by how much that month matters
    cand = []
    for chip in chips_available:
        k = key_for.get(chip)
        if not k:
            continue
        for row in chip_window_scores:
            weight = phase_targets.get(row['month'], 1.0)
            cand.append({'chip': chip, 'gw': row['gw'], 'month': row['month'],
                         'score': row[k] * weight})
    cand.sort(key=lambda c: -c['score'])

    chosen, used_gws, used_chips = [], set(), set()
    for c in cand:
        if c['chip'] in used_chips:
            continue
        if any(abs(c['gw'] - g) < min_gap for g in used_gws):
            continue
        chosen.append(c)
        used_gws.add(c['gw']); used_chips.add(c['chip'])
    chosen.sort(key=lambda c: c['gw'])
    return chosen
