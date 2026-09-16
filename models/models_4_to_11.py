"""
Models 4-11 for The Regulars FPL consultancy.

 4. Chip-timing optimiser
 5. Rival-tendency model (league-specific)
 6. Price-change predictor
 7. Squad-correlation / stacking risk
 8. League effective ownership (EO)
 9. Bench-order optimiser
10. Fixture-swing detector
11. Decision log / calibration tracker
"""

import json
import math
from collections import defaultdict, Counter


# =========================================================
# 4. CHIP-TIMING OPTIMISER
# =========================================================

def score_chip_windows(team_fixture_run, squad_teams, phases, chips_available,
                       start_gw, end_gw):
    """
    Scores each future gameweek for each chip type.

    Bench Boost  -> wants ALL 15 playing good fixtures (bench included)
    Triple Cap   -> wants one standout premium fixture (lowest difficulty, home)
    Free Hit     -> wants a gameweek where YOUR squad is bad but others are good
                    (blank GWs, or a week your teams all have difficulty 4-5)
    Wildcard     -> wants a sustained fixture swing, not one week

    Returns per-gameweek scores plus the month each GW belongs to, so chip use
    can be aimed at a month Dennis can actually win.
    """
    gw_month = {}
    for ph in phases:
        for gw in range(ph['start_event'], ph['stop_event'] + 1):
            gw_month[gw] = ph['name']

    results = []
    for gw in range(start_gw, end_gw + 1):
        diffs = []
        for t in squad_teams:
            fx = [f for f in team_fixture_run.get(t, []) if f['gw'] == gw]
            if not fx:
                diffs.append(None)          # BLANK gameweek for this team
            else:
                for f in fx:
                    diffs.append(f['difficulty'])
        played = [d for d in diffs if d is not None]
        blanks = sum(1 for d in diffs if d is None)
        doubles = len([d for d in diffs if d is not None]) - len(squad_teams) + blanks

        avg_diff = sum(played) / len(played) if played else 5
        easy_count = sum(1 for d in played if d <= 2)

        results.append({
            'gw': gw,
            'month': gw_month.get(gw, '?'),
            'avg_difficulty': round(avg_diff, 2),
            'easy_fixtures': easy_count,
            'blanks_in_squad': blanks,
            'doubles_in_squad': max(doubles, 0),
            # Bench boost: everyone playing, fixtures easy
            'bench_boost_score': round((5 - avg_diff) * 2 + easy_count - blanks * 3, 2),
            # Triple captain: one elite fixture matters most
            'triple_captain_score': round((5 - min(played) if played else 0) * 3
                                          + (2 if doubles > 0 else 0), 2),
            # Free hit: YOUR squad is bad this week
            'free_hit_score': round(avg_diff * 2 + blanks * 3, 2),
            # Wildcard: needs a sustained swing (scored separately below)
        })
    return results


def wildcard_windows(chip_window_scores, lookahead=5):
    """
    Wildcard wants a SUSTAINED improvement, so score each GW by the average
    fixture quality of the NEXT `lookahead` gameweeks after wildcarding there.
    """
    out = []
    for i, row in enumerate(chip_window_scores):
        window = chip_window_scores[i:i + lookahead]
        if len(window) < lookahead:
            break
        avg = sum(w['avg_difficulty'] for w in window) / len(window)
        out.append({'wildcard_at_gw': row['gw'], 'month': row['month'],
                    'next_%d_avg_difficulty' % lookahead: round(avg, 2),
                    'score': round((5 - avg) * 10, 2)})
    out.sort(key=lambda r: -r['score'])
    return out


# =========================================================
# 5. RIVAL-TENDENCY MODEL
# =========================================================

def rival_captain_tendencies(league_managers):
    """
    Builds a per-rival profile of captaincy behaviour from history.
    Does NOT reveal current-GW picks (impossible before deadline) — it
    estimates the PROBABILITY each rival captains a given player, which
    is what a genuine differential calculation needs.
    """
    profiles = {}
    for m in league_managers:
        caps = [c['name'] for c in m.get('captainByGw', []) if c.get('name')]
        if not caps:
            continue
        counts = Counter(caps)
        total = len(caps)
        # "template-ness": how concentrated their picks are
        concentration = max(counts.values()) / total
        profiles[m['name']] = {
            'captain_counts': dict(counts),
            'most_captained': counts.most_common(1)[0][0],
            'loyalty': round(concentration, 2),   # 1.0 = always same player
            'distinct_captains': len(counts),
            'n_gameweeks': total,
            'predictability': ('HIGH' if concentration >= 0.75 else
                               'MEDIUM' if concentration >= 0.5 else 'LOW'),
        }
    return profiles


def estimate_league_captain_distribution(profiles, candidate_players):
    """
    Estimates how many of the 16 rivals are likely to captain each candidate,
    blending each rival's own history with overall league popularity.
    """
    est = defaultdict(float)
    for name, prof in profiles.items():
        counts = prof['captain_counts']
        total = prof['n_gameweeks']
        for cand in candidate_players:
            # probability this rival captains cand = their historical rate,
            # smoothed toward uniform over candidates
            hist = counts.get(cand, 0) / total if total else 0
            smoothed = 0.7 * hist + 0.3 * (1.0 / max(len(candidate_players), 1))
            est[cand] += smoothed
    return {k: round(v, 2) for k, v in sorted(est.items(), key=lambda kv: -kv[1])}


# =========================================================
# 6. PRICE-CHANGE PREDICTOR
# =========================================================

def price_change_risk(players, total_managers=11_000_000):
    """
    FPL doesn't publish its exact price algorithm, but net transfers as a
    share of ownership is the accepted proxy. Flags likely risers/fallers.
    """
    out = []
    for p in players:
        own_pct = float(p.get('selected_by_percent') or 0)
        owners = max(own_pct / 100 * total_managers, 1)
        net = (p.get('transfers_in_event') or 0) - (p.get('transfers_out_event') or 0)
        momentum = net / owners
        if momentum > 0.08:
            flag = 'LIKELY RISE'
        elif momentum < -0.08:
            flag = 'LIKELY FALL'
        elif momentum > 0.04:
            flag = 'watch (rise)'
        elif momentum < -0.04:
            flag = 'watch (fall)'
        else:
            flag = ''
        if flag:
            out.append({'name': p['name'], 'team': p['team'], 'price': p['price'],
                        'net_transfers': net, 'momentum': round(momentum, 3),
                        'flag': flag})
    out.sort(key=lambda r: -abs(r['momentum']))
    return out


# =========================================================
# 7. SQUAD-CORRELATION / STACKING RISK
# =========================================================

def correlation_risk(squad, gw_fixtures):
    """
    FPL Review's framing: risk is proportional to exposure x correlation.
    Players from the same club share one match outcome; players on OPPOSITE
    sides of the same match are negatively correlated for clean sheets.
    """
    by_team = defaultdict(list)
    for p in squad:
        if p.get('multiplier', 1) > 0:
            by_team[p['team']].append(p['name'])

    stacks = {t: names for t, names in by_team.items() if len(names) >= 2}

    # detect same-match clashes within the squad
    clashes = []
    teams = list(by_team)
    for t in teams:
        fx = gw_fixtures.get(t, [])
        for f in fx:
            if f['opponent'] in by_team:
                pair = tuple(sorted([t, f['opponent']]))
                if pair not in [c['teams'] for c in clashes]:
                    clashes.append({'teams': pair,
                                    'players': by_team[t] + by_team[f['opponent']]})

    total_starters = sum(len(v) for v in by_team.values())
    largest = max((len(v) for v in by_team.values()), default=0)
    concentration = largest / total_starters if total_starters else 0

    return {
        'stacks': stacks,
        'same_match_clashes': clashes,
        'largest_stack': largest,
        'concentration': round(concentration, 2),
        'assessment': ('HIGH — one bad team week sinks the GW' if concentration >= 0.36
                       else 'MODERATE' if concentration >= 0.27 else 'LOW'),
    }


# =========================================================
# 8. LEAGUE EFFECTIVE OWNERSHIP
# =========================================================

def league_effective_ownership(league_managers, gw, my_name):
    """
    Rank movement depends on RIVALS' exposure, not global FPL ownership.
    A 70%-owned player may be in only 3 of 16 Regulars squads — that's a
    differential HERE even though it's template globally.

    EO = ownership% + captaincy% (captained players count double).
    Only computable for PAST gameweeks (picks hidden until deadline passes).
    """
    caps = Counter()
    n = 0
    for m in league_managers:
        row = [c for c in m.get('captainByGw', []) if c.get('gw') == gw]
        if row and row[0].get('name'):
            caps[row[0]['name']] += 1
            n += 1
    if not n:
        return {}
    return {
        'gameweek': gw,
        'managers_counted': n,
        'captain_split': {k: {'count': v, 'pct_of_league': round(100 * v / n, 1)}
                          for k, v in caps.most_common()},
        'my_captain': next((c['name'] for m in league_managers if m['name'] == my_name
                            for c in m.get('captainByGw', []) if c.get('gw') == gw), None),
    }


# =========================================================
# 9. BENCH-ORDER OPTIMISER
# =========================================================

def optimise_bench(bench_players_ctx):
    """
    Bench value = P(a starter fails to play) x P(this sub plays) x their xPts.
    So the right order weights BOTH the sub's own xPts AND their own
    likelihood of actually playing — a high-xPts sub who might be rotated
    is worth less than a nailed one with slightly lower xPts.
    """
    scored = []
    for b in bench_players_ctx:
        p_plays = b['p_sub'] + b['p_full']
        scored.append({
            'name': b['name'],
            'xpts': round(b['xpts'], 2),
            'p_plays': round(p_plays, 2),
            'bench_value': round(b['xpts'] * p_plays, 2),
        })
    scored.sort(key=lambda r: -r['bench_value'])
    return scored


# =========================================================
# 10. FIXTURE-SWING DETECTOR
# =========================================================

def detect_fixture_swings(team_fixture_run, window=4, threshold=0.8):
    """
    Flags where each team's fixture run meaningfully improves or worsens, so
    moves can be planned 2-3 GWs EARLY (before prices rise) rather than
    reactively. Compares rolling `window` averages.
    """
    swings = []
    for team, fixtures in team_fixture_run.items():
        fixtures = sorted(fixtures, key=lambda f: f['gw'])
        for i in range(len(fixtures) - window * 2 + 1):
            cur = fixtures[i:i + window]
            nxt = fixtures[i + window:i + window * 2]
            if len(nxt) < window:
                break
            cur_avg = sum(f['difficulty'] for f in cur) / window
            nxt_avg = sum(f['difficulty'] for f in nxt) / window
            delta = cur_avg - nxt_avg
            if abs(delta) >= threshold:
                swings.append({
                    'team': team,
                    'swing_at_gw': nxt[0]['gw'],
                    'from_avg': round(cur_avg, 2),
                    'to_avg': round(nxt_avg, 2),
                    'delta': round(delta, 2),
                    'direction': 'IMPROVES' if delta > 0 else 'WORSENS',
                })
    swings.sort(key=lambda s: (s['swing_at_gw'], -abs(s['delta'])))
    return swings


# =========================================================
# 11. DECISION LOG / CALIBRATION TRACKER
# =========================================================

class DecisionLog:
    """
    Records every recommendation WITH its pre-deadline reasoning, so at season
    end we can separate GOOD PROCESS from GOOD LUCK.

    GoalIQAI's discipline: a captain returning 24 points does not prove the
    captaincy was right, and a blank does not prove it was wrong. What matters
    is whether the xPts edge was real at the time.
    """

    def __init__(self, path='/mnt/user-data/outputs/fpl_decision_log.json'):
        self.path = path
        try:
            with open(path) as f:
                self.entries = json.load(f)
        except Exception:
            self.entries = []

    def log(self, gw, decision_type, decision, reasoning, xpts_edge=None,
            alternatives=None):
        self.entries.append({
            'gw': gw,
            'type': decision_type,           # captain / transfer / chip / lineup
            'decision': decision,
            'reasoning': reasoning,
            'xpts_edge_at_time': xpts_edge,
            'alternatives_considered': alternatives or [],
            'actual_outcome': None,          # filled in after the GW
        })
        self._save()

    def record_outcome(self, gw, decision_type, actual_points):
        for e in self.entries:
            if e['gw'] == gw and e['type'] == decision_type:
                e['actual_outcome'] = actual_points
        self._save()

    def calibration_report(self):
        """
        Were decisions with a bigger projected edge actually better?
        That's the test of process quality, not the raw hit rate.
        """
        scored = [e for e in self.entries
                  if e['actual_outcome'] is not None and e['xpts_edge_at_time']]
        if len(scored) < 5:
            return {'note': 'Not enough logged outcomes yet for calibration.'}
        edges = [e['xpts_edge_at_time'] for e in scored]
        outs = [e['actual_outcome'] for e in scored]
        mean_e, mean_o = sum(edges) / len(edges), sum(outs) / len(outs)
        cov = sum((e - mean_e) * (o - mean_o) for e, o in zip(edges, outs))
        var_e = sum((e - mean_e) ** 2 for e in edges)
        slope = cov / var_e if var_e else 0
        return {
            'n_decisions': len(scored),
            'mean_projected_edge': round(mean_e, 2),
            'mean_actual_outcome': round(mean_o, 2),
            'edge_to_outcome_slope': round(slope, 3),
            'interpretation': ('Process looks sound — bigger projected edges '
                              'produced better outcomes.' if slope > 0.3 else
                              'No clear signal yet; expect noise over small samples.'),
        }

    def _save(self):
        try:
            with open(self.path, 'w') as f:
                json.dump(self.entries, f, indent=2)
        except Exception:
            pass
