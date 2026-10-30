"""Experimental ECG QRS direction, independent of the frozen v1/v2 methods.

This estimates the dominant QRS *amplitude* direction, not electrode placement,
clinical electrical axis, QRS readability, or sample-accurate R-peak labels.
ECG papers motivate separating detection, baseline removal, morphology and
short-window quality. The numerical rules below are project adaptations, not
published/validated SOTA polarity rules. No waveform is modified here.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt

PARAMETERS = {
    "method": "qrs_direction_v3_development",
    "output_policy": "minute_primary_direction_with_separate_cross_checks_and_explicit_conflict_veto",
    "direction_definition": "dominant_QRS_amplitude_not_signed_area",
    "bands_hz": [[5., 25.], [8., 35.]],
    "integration_seconds": .080,
    "minimum_spacing_seconds": .250,
    "alignment_radius_seconds": .080,
    "template_radius_seconds": .120,
    "qrs_radius_seconds": .080,
    "tile_seconds": 15.,
    "minimum_tile_beats": 4,
    "minimum_template_correlation": .70,
    "minimum_tile_sign_agreement": .80,
    "minimum_dominance": .05,
    "minimum_supported_time_fraction": .75,
    "minimum_supported_tiles": 2,
    "severe_raw_p01_p99_span_uv": 5000.,
}


def _dominance(values, axis=None):
    up = np.maximum(0., np.max(values, axis=axis))
    down = np.maximum(0., -np.min(values, axis=axis))
    return (up-down) / np.maximum(up+down, 1e-12)


def _correlations(matrix, template):
    a = matrix - matrix.mean(axis=1, keepdims=True)
    b = template - template.mean()
    denom = np.linalg.norm(a, axis=1)*np.linalg.norm(b)
    return np.divide(a@b, denom, out=np.zeros(len(a)), where=denom > 1e-12)


def _candidates(filtered, fs):
    """Polarity invariant detection and steepest-edge alignment.

    Align on squared slope rather than raw absolute extrema: a small R/S
    amplitude change must not shift the template from the R wave to the S wave.
    These are diagnostic anchors, not output R-peak ground truth.
    """
    derivative = np.diff(filtered, prepend=filtered[0])
    width = max(3, round(PARAMETERS['integration_seconds']*fs))
    energy = np.convolve(derivative**2, np.ones(width)/width, 'same')
    median = float(np.median(energy))
    noise = float(np.median(np.abs(energy-median))*1.4826)
    scale = float(np.quantile(energy, .95))
    if scale <= 1e-12:
        return np.array([], dtype=int)
    seeds, _ = find_peaks(energy, distance=max(1, round(.250*fs)),
                         height=max(median+5*noise, .15*scale),
                         prominence=max(4*noise, .10*scale))
    radius, half = max(2, round(.080*fs)), max(3, round(.120*fs))
    anchors = []
    for seed in seeds:
        if seed < half+radius or seed+half+radius >= len(filtered):
            continue
        anchor = int(seed-radius+np.argmax(derivative[seed-radius:seed+radius+1]**2))
        if anchors and anchor-anchors[-1] < round(.200*fs):
            continue
        anchors.append(anchor)
    return np.asarray(anchors, dtype=int)


def _extract(matrix_signal, anchors, fs):
    half, radius = max(3, round(.120*fs)), max(2, round(.080*fs))
    matrix = np.asarray([matrix_signal[i-half:i+half+1] for i in anchors])
    # Symmetric shoulders remove residual filter offset without reintroducing
    # slow drift into the QRS extrema. Odd operations preserve sign equivariance.
    baseline = np.median(np.c_[matrix[:, :half-radius], matrix[:, half+radius+1:]], axis=1)
    matrix = matrix-baseline[:, None]
    amplitude = np.max(np.abs(matrix), axis=1)
    normalized = matrix/np.maximum(amplitude[:, None], 1e-12)
    dom = _dominance(matrix[:, half-radius:half+radius+1], axis=1)
    return matrix, normalized, dom


def measure_direction_variant(signal, times_ms, fs_hz, *, diagnostics=False,
                              aggregate='tiles', bands='both'):
    """Return a candidate direction plus independently inspectable components.

    `aggregate='minute'` and `bands='primary'` are predeclared ablations only.
    No participant-specific thresholds or human-choice lookup is used.
    A tile needs stable morphology and >=80% directional agreement in each
    selected band; a minute needs support spanning >=75% of elapsed time, and
    no confident opposite tile. Hence a locally unclear tile can be retained
    without treating every small QRS as a failed vote for the whole minute.
    """
    if aggregate not in {'tiles', 'minute'} or bands not in {'both', 'primary'}:
        raise ValueError('Unknown direction diagnostic configuration')
    x, t = np.asarray(signal, float), np.asarray(times_ms, float)
    if x.ndim != 1 or x.shape != t.shape:
        raise ValueError('Signal and timestamps must be equal-length 1D arrays')
    result = dict(raw_polarity='low_signal', reason='insufficient_signal',
                  event_count=0, beat_indices=[], tiles=[], supported_fraction=0.,
                  positive_tiles=0, negative_tiles=0,
                  raw_p01_p99_span_uv=float('nan'))
    if (not np.isfinite(fs_hz) or not 60 <= fs_hz <= 1000
            or len(x) < max(100, round(10*fs_hz))
            or not np.isfinite(x).all() or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)):
        return result
    span = float(np.quantile(x, .99)-np.quantile(x, .01))
    result['raw_p01_p99_span_uv'] = span
    if span > PARAMETERS['severe_raw_p01_p99_span_uv']:
        result.update(raw_polarity='severe_artifact', reason='severe_amplitude_gate')
        return result
    channels = [sosfiltfilt(butter(3, [lo, min(hi, .45*fs_hz)], btype='bandpass',
                                  fs=fs_hz, output='sos'), x)
                for lo, hi in PARAMETERS['bands_hz']]
    anchors = _candidates(channels[0], fs_hz)
    result.update(event_count=len(anchors), beat_indices=anchors.tolist())
    duration = (t[-1]-t[0])/1000 + 1/fs_hz
    if len(anchors) < max(8, round(.20*duration)):
        result['reason'] = 'too_few_qrs_candidates'
        return result
    features = [_extract(y, anchors, fs_hz) for y in channels]
    secs = (t[anchors]-t[0])/1000
    tile_width = PARAMETERS['tile_seconds'] if aggregate == 'tiles' else duration
    # Partition elapsed time evenly: sub-sample timestamp jitter must not
    # manufacture an empty fifth tile or turn 3/4 support into <75% support.
    ntiles = max(1, round(duration/tile_width))
    edges = np.linspace(0., duration, ntiles+1)
    half, radius = max(3, round(.120*fs_hz)), max(2, round(.080*fs_hz))
    for tile_i in range(ntiles):
        start, end = float(edges[tile_i]), float(edges[tile_i+1])
        indices = np.flatnonzero((secs >= start) & (secs < end))
        tile = dict(start_sec=start, end_sec=end, count=len(indices),
                    direction='ambiguous', reason='too_few_candidates', bands=[])
        if len(indices) >= PARAMETERS['minimum_tile_beats']:
            for matrix, normalized, dom in features:
                rows = normalized[indices]
                template = np.median(rows, axis=0)
                correlation = float(np.median(_correlations(rows, template)))
                template_dom = float(_dominance(template[half-radius:half+radius+1]))
                # Retain continuous weak-but-consistent evidence. An explicit
                # dead zone prevents an almost equiphasic template deciding sign.
                sign = int(np.sign(template_dom)) if abs(template_dom) >= .05 else 0
                agreement = float(np.mean(sign*dom[indices] > 0)) if sign else 0.
                median_dom = float(np.median(dom[indices]))
                valid = (sign != 0 and sign*median_dom >= .05
                         and correlation >= .70 and agreement >= .80)
                tile['bands'].append(dict(template_dominance=template_dom,
                                          median_beat_dominance=median_dom,
                                          template_correlation=correlation,
                                          sign_agreement=agreement,
                                          direction=('positive' if sign > 0 else 'negative') if valid else 'ambiguous'))
            selected = tile['bands'] if bands == 'both' else tile['bands'][:1]
            signs = {b['direction'] for b in selected}
            if len(signs) == 1 and 'ambiguous' not in signs:
                tile.update(direction=selected[0]['direction'], reason='consistent_qrs_template')
            elif len(signs-{'ambiguous'}) > 1:
                tile['reason'] = 'filter_direction_conflict'
            else:
                tile['reason'] = 'weak_or_inconsistent_qrs_template'
        result['tiles'].append(tile)
    good = [b for b in result['tiles'] if b['direction'] in {'positive', 'negative'}]
    result['positive_tiles'] = sum(b['direction'] == 'positive' for b in good)
    result['negative_tiles'] = sum(b['direction'] == 'negative' for b in good)
    result['supported_fraction'] = sum(b['end_sec']-b['start_sec'] for b in good)/duration
    result.update(raw_polarity='ambiguous', reason='insufficient_consistent_time_support')
    if result['positive_tiles'] and result['negative_tiles']:
        result['reason'] = 'opposite_directions_across_tiles'
    elif any(b['reason'] == 'filter_direction_conflict' for b in result['tiles']):
        result['reason'] = 'filter_direction_conflict'
    elif (len(good) >= (2 if aggregate == 'tiles' else 1)
          and result['supported_fraction'] >= .75-1e-9):
        result.update(raw_polarity=good[0]['direction'], reason='consistent_direction_across_time_and_filters')
    if diagnostics:
        baseline = median_filter(median_filter(x, size=max(3, round(.200*fs_hz)|1), mode='nearest'),
                                 size=max(3, round(.600*fs_hz)|1), mode='nearest')
        result['diagnostics'] = dict(
            relative_beat_seconds=secs.tolist(),
            per_beat_dominance=[f[2].tolist() for f in features],
            templates=[np.median(f[1], axis=0).tolist() for f in features],
            template_seconds=(np.arange(-half,half+1)/fs_hz).tolist(),
            baseline_corrected_raw=x-baseline,
            primary_filtered=channels[0], confirmation_filtered=channels[1])
    return result


def measure_direction_v3(signal, times_ms, fs_hz, *, diagnostics=False):
    """Separate a main direction from the completeness of its cross-checks.

    Failure to reach the stricter diagnostic's support threshold does not by
    itself erase a clear minute-level QRS template. Actual opposite confident
    tiles/bands do veto the minute-level direction. This policy is developmental
    and is NOT an authorization to flip any recording.
    """
    primary = measure_direction_variant(signal,times_ms,fs_hz,diagnostics=diagnostics,
                                        aggregate='minute',bands='primary')
    check = measure_direction_variant(signal,times_ms,fs_hz,aggregate='tiles',bands='both')
    result = dict(primary)
    result.update(primary_direction=primary['raw_polarity'],
                  primary_evidence=primary['tiles'], tiles=check['tiles'],
                  cross_check_direction=check['raw_polarity'],
                  cross_check_reason=check['reason'],cross_check_status='not_applicable',
                  supported_fraction=check['supported_fraction'],
                  positive_tiles=check['positive_tiles'],negative_tiles=check['negative_tiles'])
    if primary['raw_polarity'] in {'positive','negative'}:
        if (check['reason'] in {'opposite_directions_across_tiles','filter_direction_conflict'}
                or check['raw_polarity'] in {'positive','negative'} and check['raw_polarity'] != primary['raw_polarity']):
            result.update(raw_polarity='ambiguous',reason='explicit_cross_check_direction_conflict',cross_check_status='conflict')
        elif check['raw_polarity'] == primary['raw_polarity']:
            result.update(reason='minute_direction_and_cross_checks_agree',cross_check_status='supported')
        else:
            result.update(reason='minute_direction_with_limited_cross_check_support',cross_check_status='limited_support')
    return result
