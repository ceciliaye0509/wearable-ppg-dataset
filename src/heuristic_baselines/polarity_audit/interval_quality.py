"""Frozen interval agreement rule; no web service."""
def aggregate_quarters(quarters):
    """Predeclared 45/60-second agreement, with any explicit opposite veto."""
    signs = [q['direction'] for q in quarters if q['direction'] in {'positive', 'negative'}]
    if len(set(signs)) > 1:
        return 'ambiguous'
    return signs[0] if len(signs) >= 3 else 'ambiguous'
