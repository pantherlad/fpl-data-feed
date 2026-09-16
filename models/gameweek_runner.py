"""
gameweek_runner.py — ties every model together into ONE briefing.

This is the file to run each gameweek. It:
  1. Loads the live feed from GitHub
  2. Derives team attack/defence ratings
  3. Computes xPts + distributions for every squad player
  4. Picks the optimal legal XI, bench order, captain and vice
  5. Checks correlation risk, role changes, price risk
  6. Reads the monthly standing and sets the variance posture
  7. Prints a single briefing

Run:  python3 gameweek_runner.py <GW>
"""

import json
import sys
import urllib.request
from collections import defaultdict

sys.path.insert(0, '.')
from xpts_model import player_xpts, match_expectations, minutes_profile
from model_distribution import simulate_player
from model_planner_and_prize import variance_posture, settle_month
from models_4_to_11 import (rival_captain_tendencies, correlation_risk,
                            price_change_risk, detect_fixture_swings,
                            league_effective_ownership, score_chip_windows)
from models_12_to_21 import detect_role_change
from models_v2 import (set_piece_value_v2, bench_order_v2, predict_ownership,
                       sensitivity_analysis, multi_gw_optimise)

FEED = 'https://raw.githubusercontent.com/pantherlad/fpl-data-feed/main/fpl_feed.json'


def load_feed(url=FEED):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode())


def team_rates(feed):
    """Derive per-team attacking and defensive rates per 90 from player data."""
    games = max(feed['current_event'], 1)
    xg = defaultdict(float)
    xgc = defaultdict(list)
    for p in feed['all_players']:
        xg[p['team']] += float(p.get('expected_goals') or 0)
        if p['position'] in ('DEF', 'GKP') and float(p.get('minutes') or 0) >= 270:
            mins = float(p['minutes'])
            xgc[p['team']].append(float(p.get('expected_goals_conceded') or 0) / mins * 90)
    return {t: (xg[t] / games,
                sum(xgc[t]) / len(xgc[t]) if xgc.get(t) else 1.45)
            for t in xg}


def gw_fixtures(feed, gw):
    out = {}
    for team, fixtures in feed['team_fixture_run'].items():
        for f in fixtures:
            if f['gw'] == gw:
                out[team] = f
    return out


def build_briefing(feed, gw, my_name='Dennis Roy'):
    rates = team_rates(feed)
    fixtures = gw_fixtures(feed, gw)
    players = {(p['name'], p['team']): p for p in feed['all_players']}
    per_gw = feed.get('per_gw_history', {})

    rows = []
    for sp in feed['squad']:
        key = (sp['name'], sp['team'])
        full = players.get(key)
        fx = fixtures.get(sp['team'])
        if not full or not fx:
            continue

        opp = fx['opponent']
        home = fx['venue'] == 'H'
        my_atk, my_def = rates.get(sp['team'], (1.45, 1.45))
        op_atk, op_def = rates.get(opp, (1.45, 1.45))

        xg_for = match_expectations(my_atk, op_def, home)
        xg_against = match_expectations(op_atk, my_def, not home)

        # role-change check: weight recent minutes when a step change is found
        hist = per_gw.get(sp['name'])
        role = None
        if isinstance(hist, list) and len(hist) >= 4:
            role = detect_role_change(hist)

        # news sentiment nudges expected minutes
        news = {'minutes_multiplier': 1.0}
        if full.get('news'):
            news = {'minutes_multiplier': 0.6}  # any news flag = caution

        rotation_risk = 0.0
        if fx.get('days_rest') is not None and fx['days_rest'] <= 3:
            rotation_risk = 0.2
        if news['minutes_multiplier'] < 1.0:
            rotation_risk = max(rotation_risk, 1 - news['minutes_multiplier'])

        # If a role change shows minutes trending UP, treat as nailed
        if role and role['role_change'] and role['recent_mins_avg'] >= 80:
            full = dict(full)
            full['minutes'] = max(float(full.get('minutes') or 0), 340)
            full['starts'] = max(int(full.get('starts') or 0), 4)

        xpts, bd = player_xpts(full, xg_for, xg_against, rotation_risk)
        sp_val = set_piece_value_v2(full, xg_for)

        p_none, p_sub, p_full = minutes_profile(full, rotation_risk)
        dist = simulate_player({
            'position': full['position'], 'p_none': p_none, 'p_sub': p_sub,
            'p_full': p_full,
            'exp_goals': bd['goals'] / {'GKP': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}[full['position']],
            'exp_assists': bd['assists'] / 3,
            'p_cs': bd['P(CS)'], 'team_xg_against': xg_against,
            'dc_per90': float(full.get('defensive_contribution_per_90') or 0),
            'defcon_threshold': 10 if full['position'] == 'DEF' else 12,
            'bonus_rate': 0, 'p_yellow': 0.12, 'is_gk': full['position'] == 'GKP',
        }, n_sims=4000)

        rows.append({
            'name': sp['name'], 'pos': full['position'], 'team': sp['team'],
            'fix': f"{opp}({fx['venue']})", 'diff': fx['difficulty'],
            'xpts': round(xpts, 2), 'floor': dist['floor_p20'],
            'ceiling': dist['ceiling_p90'], 'p_haul': dist['p_haul'],
            'p_blank': dist['p_blank'], 'p60': round(p_full, 2),
            'setpiece': sp_val['set_piece_xpts'],
            'role_flag': role['flags'] if role and role['role_change'] else [],
            'news': full.get('news', ''),
        })

    rows.sort(key=lambda r: -r['xpts'])
    return rows, rates, fixtures


def best_xi(rows):
    """Optimal legal XI: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD (max 11)."""
    by = defaultdict(list)
    for r in rows:
        by[r['pos']].append(r)
    for k in by:
        by[k].sort(key=lambda r: -r['xpts'])
    best, best_pts = None, -1
    for nd in range(3, 6):
        for nm in range(2, 6):
            nf = 10 - nd - nm
            if not (1 <= nf <= 3):
                continue
            if len(by['DEF']) < nd or len(by['MID']) < nm or len(by['FWD']) < nf:
                continue
            xi = by['GKP'][:1] + by['DEF'][:nd] + by['MID'][:nm] + by['FWD'][:nf]
            pts = sum(r['xpts'] for r in xi)
            if pts > best_pts:
                best, best_pts = xi, pts
    return best, round(best_pts, 2)


def monthly_context(feed, my_name='Dennis Roy'):
    phases = feed['official_monthly_phases']
    cur = feed['current_event']
    # Skip FPL's "Overall" phase (spans GW1-38) — we want the calendar month.
    phase = next((p for p in phases
                  if p['start_event'] <= cur <= p['stop_event']
                  and p['name'].lower() != 'overall'
                  and (p['stop_event'] - p['start_event']) < 20), None)
    if not phase:
        return None
    lg = feed['the_regulars_league']
    lo, hi = phase['start_event'] - 1, phase['stop_event']
    scores = {m['name']: sum(m['gwPoints'][lo:hi]) for m in lg['managers']}
    gws_left = phase['stop_event'] - cur
    me = scores.get(my_name, 0)
    rivals = {k: v for k, v in scores.items() if k != my_name}
    posture = variance_posture(me, rivals, max(gws_left, 1))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {'phase': phase['name'], 'gws_left': gws_left, 'my_score': me,
            'standings': ranked, 'posture': posture,
            'my_rank': [i for i, (n, _) in enumerate(ranked, 1) if n == my_name][0]}


def main():
    gw = int(sys.argv[1]) if len(sys.argv) > 1 else None
    feed = load_feed()
    gw = gw or (feed['current_event'] + 1)
    print(f"Feed generated: {feed['generated_at_sgt']}")
    print(f"Free transfers: {feed.get('free_transfers') or feed.get('free_transfers_estimate')}  Bank: £{feed['bank']}m")
    print(f"=== GAMEWEEK {gw} BRIEFING ===\n")

    rows, rates, fixtures = build_briefing(feed, gw)
    print(f"{'Player':<14}{'Pos':<5}{'Fix':<9}{'xPts':>6}{'Floor':>7}{'Ceil':>6}"
          f"{'pHaul':>7}{'P60':>6}{'SetP':>6}")
    for r in rows:
        print(f"{r['name']:<14}{r['pos']:<5}{r['fix']:<9}{r['xpts']:>6.2f}"
              f"{r['floor']:>7}{r['ceiling']:>6}{r['p_haul']:>7.2f}"
              f"{r['p60']:>6.2f}{r['setpiece']:>6.2f}")
        if r['role_flag']:
            print(f"    ROLE CHANGE: {r['role_flag']}")
        if r['news']:
            print(f"    NEWS: {r['news']}")

    xi, total = best_xi(rows)
    bench = [r for r in rows if r not in xi]
    bench_ranked = bench_order_v2([{'name': b['name'], 'xpts': b['xpts'],
                                    'p_plays': b['p60']} for b in bench])
    order = {b['name']: b['slot'] for b in bench_ranked}
    bench.sort(key=lambda r: order.get(r['name'], 9))
    print(f"\nOPTIMAL XI (total xPts {total}):")
    for r in sorted(xi, key=lambda r: ['GKP', 'DEF', 'MID', 'FWD'].index(r['pos'])):
        print(f"   {r['pos']:<4}{r['name']:<14}{r['fix']}")
    print("BENCH (ordered by likelihood x value):")
    for r in bench:
        print(f"   {r['pos']:<4}{r['name']:<14}{r['fix']}")

    ctx = monthly_context(feed)
    if ctx:
        print(f"\n=== {ctx['phase'].upper()} PRIZE CONTEXT ===")
        print(f"You: {ctx['my_score']} pts, rank {ctx['my_rank']}/16, "
              f"{ctx['gws_left']} GW(s) left")
        for i, (n, s) in enumerate(ctx['standings'][:4], 1):
            print(f"   {i}. {n}: {s}")
        p = ctx['posture']
        print(f"Posture: {p['recommended_posture']}")
        print(f"   variance worth ${p['variance_is_worth']} vs "
              f"mean worth ${p['mean_is_worth']}")

        if 'CEILING' in p['recommended_posture']:
            cands = sorted(rows, key=lambda r: -r['p_haul'])[:3]
            print("   Highest-ceiling captain options:")
            for c in cands:
                print(f"      {c['name']}: ceiling {c['ceiling']}, "
                      f"P(haul) {c['p_haul']}")
        else:
            cands = sorted(rows, key=lambda r: -r['floor'])[:3]
            print("   Safest captain options:")
            for c in cands:
                print(f"      {c['name']}: floor {c['floor']}, "
                      f"P(blank) {c['p_blank']}")

    cr = correlation_risk(feed['squad'],
                          {t: [f] for t, f in fixtures.items()})
    print(f"\nCorrelation risk: {cr['assessment']} ({cr['concentration']})")
    for c in cr['same_match_clashes']:
        print(f"   CLASH {c['teams']}: {c['players']}")

    pr = price_change_risk(
        [p for p in feed['all_players']
         if p['name'] in {s['name'] for s in feed['squad']}])
    if pr:
        print("\nPrice watch:")
        for x in pr[:5]:
            print(f"   {x['name']}: {x['flag']} ({x['momentum']})")


if __name__ == '__main__':
    main()
