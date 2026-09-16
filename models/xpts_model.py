"""
xPts model — built to the methodology described by GoalIQAI / FPL Review / FPL Estimator.

Core formula:
  xP = appearance + goals + assists + clean sheet + saves + DEFCON + bonus - deductions

Key principles followed (and the mistakes they avoid):
 - Minutes come FIRST: everything else is conditional on playing time.
   Appearance modelled as three states (no appearance / <60 / 60+), not one average.
 - Threshold scoring uses PROBABILITY OF CROSSING, not average actions x rate.
   (A mid averaging 8 DEFCON actions does NOT get 8/12 of 2 points.)
 - Clean sheets conditional on reaching 60 mins, derived from Poisson on match xGC,
   not from FDR directly.
 - Position-specific goal values (GK 10 / DEF 6 / MID 5 / FWD 4).
 - Goals-conceded deduction (-1 per 2) for GK/DEF.
 - Card deductions weighted by actual season card rate.
"""

import json
import math


# ---------- helpers ----------

def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_at_least(k, lam):
    return 1.0 - sum(poisson_pmf(i, lam) for i in range(k))


def f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------- 1. Expected minutes ----------

def minutes_profile(player, rotation_risk=0.0):
    """
    Returns (p_no_appearance, p_under_60, p_60_plus).
    Derived from starts/appearances ratio and minutes per appearance.
    rotation_risk: manual override 0..1 (e.g. European game 3 days prior).
    """
    mins = f(player.get('minutes'))
    starts = f(player.get('starts'))
    cop = player.get('chance_of_playing_next_round')

    # Availability haircut from FPL's own flag
    avail = 1.0 if cop is None else f(cop) / 100.0
    if player.get('status') != 'a' and cop is None:
        avail = 0.5

    gws_played = max(starts, 1)
    mins_per_start = mins / gws_played if gws_played else 0

    if starts >= 4 and mins_per_start >= 80:
        base = (0.03, 0.07, 0.90)      # nailed
    elif starts >= 3 and mins_per_start >= 70:
        base = (0.07, 0.15, 0.78)      # near-nailed
    elif starts >= 2 and mins_per_start >= 55:
        base = (0.15, 0.30, 0.55)      # rotation risk
    else:
        base = (0.40, 0.35, 0.25)      # fringe

    p_none, p_sub, p_full = base
    # apply rotation risk + availability: shift mass toward not playing
    shift = (1 - avail) + rotation_risk * (1 - rotation_risk * 0.3)
    shift = min(shift, 0.9)
    p_none = p_none + shift * (p_sub + p_full)
    p_sub = p_sub * (1 - shift)
    p_full = p_full * (1 - shift)
    return p_none, p_sub, p_full


# ---------- 2. Match-level team expectations ----------

def match_expectations(team_xg90, opp_xgc90, is_home, league_avg=1.45):
    """
    Expected goals FOR this team in this match, blending the team's own
    attacking rate with the opponent's leakiness, plus venue adjustment.
    """
    home_mult = 1.15 if is_home else 0.90
    lam = math.sqrt(max(team_xg90, 0.2) * max(opp_xgc90, 0.2)) * home_mult
    # regress lightly toward league average (small-sample protection, 4 GWs in)
    lam = 0.85 * lam + 0.15 * league_avg
    return max(lam, 0.25)


# ---------- 3. Player xPts ----------

def player_xpts(player, team_xg_for, team_xg_against, rotation_risk=0.0,
                pen_share_bonus=True, verbose=False):
    pos = player['position']
    p_none, p_sub, p_full = minutes_profile(player, rotation_risk)
    p_plays = p_sub + p_full

    # --- appearance ---
    xp_app = p_sub * 1 + p_full * 2

    # --- attacking returns ---
    # per-90 rates scaled by the fixture's attacking environment.
    # Regress toward a position baseline: 4 GWs is a small sample, and an
    # unusually hot rate (e.g. a full-back on 0.41 xG/90) is mostly noise.
    xg90_raw = f(player.get('expected_goals_per_90'))
    xa90_raw = f(player.get('expected_assists_per_90'))

    pos_base_xg = {'GKP': 0.0, 'DEF': 0.06, 'MID': 0.18, 'FWD': 0.38}[pos]
    pos_base_xa = {'GKP': 0.0, 'DEF': 0.08, 'MID': 0.16, 'FWD': 0.13}[pos]

    mins_played = f(player.get('minutes'))
    # weight own rate vs positional prior; ~450 mins before own rate dominates
    w = min(mins_played / 450.0, 1.0) * 0.7
    xg90 = w * xg90_raw + (1 - w) * pos_base_xg
    xa90 = w * xa90_raw + (1 - w) * pos_base_xa

    # scale player's baseline rate by how good this fixture is relative to
    # their season-average attacking environment (~1.45 goals/gm baseline)
    fixture_attack_mult = team_xg_for / 1.45

    # minutes-weighted share of a full match
    mins_factor = p_full * 1.0 + p_sub * 0.45

    goal_value = {'GKP': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}[pos]
    exp_goals = xg90 * fixture_attack_mult * mins_factor
    exp_assists = xa90 * fixture_attack_mult * mins_factor

    # penalty duty uplift (first-choice taker gets a real bump)
    if pen_share_bonus and player.get('penalties_order') == 1:
        exp_goals += 0.08 * fixture_attack_mult

    xp_goals = exp_goals * goal_value
    xp_assists = exp_assists * 3

    # --- clean sheet (conditional on 60+ mins) ---
    p_cs = math.exp(-team_xg_against)
    cs_value = {'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}[pos]
    xp_cs = p_full * p_cs * cs_value

    # --- goals conceded deduction (GK/DEF only): -1 per 2 conceded ---
    xp_conceded = 0.0
    if pos in ('GKP', 'DEF'):
        for k in range(2, 9):
            xp_conceded -= poisson_pmf(k, team_xg_against) * p_full * (k // 2)

    # --- saves (GK only): 1 pt per 3 saves ---
    xp_saves = 0.0
    if pos == 'GKP':
        exp_saves = team_xg_against * 2.6          # ~2.6 saves per goal faced
        for mult in (3, 6, 9):
            xp_saves += poisson_at_least(mult, exp_saves) * p_full * 1

    # --- DEFCON: probability of CROSSING threshold, not linear ---
    dc90 = f(player.get('defensive_contribution_per_90'))
    threshold = 10 if pos == 'DEF' else 12
    if dc90 > 0:
        p_cross = poisson_at_least(threshold, dc90)
        xp_defcon = p_cross * 2 * p_full
    else:
        xp_defcon = 0.0

    # --- bonus (scaled by this fixture vs their season average) ---
    bonus_total = f(player.get('bonus'))
    apps = max(f(player.get('starts')), 1)
    bonus_rate = bonus_total / apps
    xp_bonus = bonus_rate * mins_factor * min(fixture_attack_mult, 1.5)

    # --- deductions: cards ---
    yc = f(player.get('yellow_cards'))
    rc = f(player.get('red_cards'))
    xp_cards = -(yc / apps) * p_plays * 1 - (rc / apps) * p_plays * 3

    total = (xp_app + xp_goals + xp_assists + xp_cs + xp_conceded +
             xp_saves + xp_defcon + xp_bonus + xp_cards)

    breakdown = {
        'appearance': round(xp_app, 2),
        'goals': round(xp_goals, 2),
        'assists': round(xp_assists, 2),
        'clean_sheet': round(xp_cs, 2),
        'conceded': round(xp_conceded, 2),
        'saves': round(xp_saves, 2),
        'defcon': round(xp_defcon, 2),
        'bonus': round(xp_bonus, 2),
        'cards': round(xp_cards, 2),
        'P(60+)': round(p_full, 2),
        'P(CS)': round(p_cs, 2),
        'xPts': round(total, 2),
    }
    if verbose:
        print(player['name'], breakdown)
    return total, breakdown
