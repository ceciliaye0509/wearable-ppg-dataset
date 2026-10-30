"""Estimate human-assessed alert validity with participant sampling weights."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from build_audit import DEFAULT_OUT, read_rows
from serve_audit import read_ratings


def weighted_proportion(sample, strata, values):
    total = sum(int(s['population_minutes']) for s in strata)
    estimate, variance = 0., 0.
    for s in strata:
        population, n = int(s['population_minutes']), int(s['sampled_minutes'])
        if not population:
            continue
        ys = [float(values[row['id']]) for row in sample if row['participant'] == s['participant']]
        if len(ys) != n:
            raise ValueError('The full probability sample must be completed before estimating prevalence')
        mean = sum(ys) / n
        weight = population / total
        estimate += weight * mean
        if n > 1 and n < population:
            sample_variance = sum((y - mean) ** 2 for y in ys) / (n - 1)
            variance += weight ** 2 * (1 - n / population) * sample_variance / n
    margin = 1.96 * math.sqrt(variance)
    return dict(proportion=estimate, approximate_design_based_95_percent_interval=[max(0., estimate - margin), min(1., estimate + margin)])


def summarize(sample, strata, ratings):
    all_sample = sample
    sample = [row for row in sample if row.get('sampling_stratum', 'old_ambiguous') == 'old_ambiguous']
    complete = {row['id']: ratings[row['id']] for row in sample if row['id'] in ratings and ratings[row['id']].get('status') == 'complete'}
    controls = [row for row in all_sample if row.get('sampling_stratum', 'old_ambiguous') != 'old_ambiguous']
    report = dict(planned_minutes=len(all_sample), primary_planned_minutes=len(sample), primary_completed_minutes=len(complete),
                  safety_planned_minutes=len(controls), safety_completed_minutes=sum(r['id'] in ratings and ratings[r['id']].get('status') == 'complete' for r in controls),
                  completed_minutes=len(complete) + sum(r['id'] in ratings and ratings[r['id']].get('status') == 'complete' for r in controls),
                  population_old_ambiguous_minutes=sum(int(s['population_minutes']) for s in strata),
                  per_participant=[dict(participant=s['participant'], planned=s['sampled_minutes'],
                                        completed=sum(row['participant'] == s['participant'] and row['id'] in complete for row in sample)) for s in strata],
                  population_estimates=None)
    completed_ratings = [ratings[row['id']] for row in all_sample
                         if ratings.get(row['id'], {}).get('status') == 'complete']
    def shown(value):
        return value is True or str(value).lower() == 'true'
    report['review_assistance'] = dict(
        assisted_completed_minutes=sum(shown(r.get('assistance_shown')) for r in completed_ratings),
        unaided_or_legacy_completed_minutes=sum(not shown(r.get('assistance_shown')) for r in completed_ratings),
        note='Algorithm-assisted review is not blinded independent detector validation. Legacy missing assistance metadata is not proof of blinding.')
    if len(complete) != len(sample):
        report['note'] = 'Human review is incomplete. No population error/readability rate is inferred from the selectively completed subset.'
        return report
    distributions = {}
    for field, options in [('qrs_readability', ['readable', 'partially_readable', 'unreadable', 'unsure']),
                           ('direction_determinability', ['yes', 'no', 'unsure'])]:
        distributions[field] = {option: weighted_proportion(sample, strata, {key: answer[field] == option for key, answer in complete.items()}) for option in options}
    report['population_estimates'] = distributions
    weight = {row['id']: float(row['inverse_probability_weight']) for row in sample}
    human_yes = sum(weight[key] for key, a in complete.items() if a['direction_determinability'] == 'yes')
    candidate_both = sum(weight[row['id']] for row in sample if complete[row['id']]['direction_determinability'] == 'yes' and row['new_raw_polarity'] in {'positive', 'negative'})
    agreement = sum(weight[row['id']] for row in sample if complete[row['id']]['direction_determinability'] == 'yes' and row['new_raw_polarity'] == complete[row['id']]['direction'])
    report['candidate_comparison'] = dict(
        coverage_given_human_determinate=candidate_both / human_yes if human_yes else None,
        agreement_when_both_determinate=agreement / candidate_both if candidate_both else None,
        note='Weighted assessment only within the old ambiguous population; not an all-recording detector accuracy claim. Algorithm-assisted judgments are not blinded independent validation.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--ratings-csv', type=Path)
    args = parser.parse_args()
    sample = read_rows(args.audit_dir / 'sampling_manifest.csv')
    protocol = json.loads((args.audit_dir / 'audit_protocol.json').read_text())
    if args.ratings_csv:
        rows = read_rows(args.ratings_csv)
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate audit IDs in ratings export')
        ratings = {r['id']: r for r in rows}
    else:
        database = args.audit_dir / 'human_audit.sqlite3'
        ratings = read_ratings(database) if database.is_file() else {}
    if set(ratings) - {r['id'] for r in sample}:
        raise ValueError('Ratings do not belong to this audit sample')
    print(json.dumps(summarize(sample, protocol['strata'], ratings), indent=2))


if __name__ == '__main__':
    main()
