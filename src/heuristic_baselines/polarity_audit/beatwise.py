"""Candidate beat-wise polarity measurement; not a QRS-readability label.

Keep the historical implementation in correct_ecg_polarity.py unchanged so
previous manifests and ongoing human annotations remain reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

PARAMETERS = {
    "method": "beatwise_qrs_v2_candidate",
    "bandpass_hz": [5.0, 25.0],
    "energy_integration_seconds": 0.080,
    "minimum_candidate_spacing_seconds": 0.250,
    "qrs_search_radius_seconds": 0.080,
    "minimum_raw_qrs_dominance": 0.20,
    "minimum_confident_beat_fraction": 0.70,
    "minimum_same_direction_fraction": 0.90,
    "minimum_template_correlation": 0.70,
    "severe_raw_p01_p99_span_uv": 5000.0,
}


def measure_beatwise(signal, times_ms, fs_hz):
    """One derivative-energy QRS candidate gets at most one polarity vote.

    Votes use raw, locally baseline-referenced QRS extrema, not the sign of
    every peak in ABS(bandpassed ECG). Unbalanced directions must also agree
    with a robust beat template. This does not establish clinical normality,
    electrode placement, or reliable beat timing in every part of a minute.
    """
    x, t = np.asarray(signal, dtype=float), np.asarray(times_ms, dtype=float)
    result = dict(raw_polarity="low_signal", polarity_score=float("nan"),
                  event_count=0, positive_votes=0, negative_votes=0,
                  uncertain_beats=0, confident_beat_fraction=0.0,
                  sign_agreement=float("nan"), template_correlation=float("nan"),
                  template_dominance=float("nan"), template_direction="undetermined",
                  raw_p01_p99_span_uv=float("nan"), q005_uv=float("nan"),
                  q995_uv=float("nan"), reason="insufficient_signal", beat_indices=[])
    if x.ndim != 1 or t.shape != x.shape:
        raise ValueError("Signal and timestamps must be equal-length one-dimensional arrays")
    if (len(x) < max(100, int(10 * fs_hz)) or not 60 <= fs_hz <= 1000
            or not np.isfinite(x).all() or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)):
        return result
    span = float(np.quantile(x, .99) - np.quantile(x, .01))
    result["raw_p01_p99_span_uv"] = span
    if span > PARAMETERS["severe_raw_p01_p99_span_uv"]:
        result.update(raw_polarity="severe_artifact", reason="severe_amplitude_gate")
        return result
    try:
        filtered = sosfiltfilt(butter(3, [5, min(25, .45 * fs_hz)],
                                     btype="bandpass", fs=fs_hz, output="sos"), x)
    except (ValueError, FloatingPointError):
        return result
    result["q005_uv"], result["q995_uv"] = map(float, np.quantile(filtered, [.005, .995]))
    derivative = np.diff(filtered, prepend=filtered[0])
    width = max(3, int(round(.080 * fs_hz)))
    energy = np.convolve(derivative ** 2, np.ones(width) / width, mode="same")
    median = float(np.median(energy))
    noise = float(np.median(np.abs(energy - median)) * 1.4826)
    scale = float(np.quantile(energy, .95))
    if scale <= 1e-12:
        return result
    seeds, _ = find_peaks(energy, distance=max(1, int(round(.250 * fs_hz))),
                         height=max(median + 5 * noise, .15 * scale),
                         prominence=max(4 * noise, .10 * scale))
    radius = max(2, int(round(.080 * fs_hz)))
    outer = max(radius + 1, int(round(.200 * fs_hz)))
    half_template = max(radius, int(round(.120 * fs_hz)))
    votes, anchors, templates = [], [], []
    for seed in seeds:
        if seed < outer or seed + outer >= len(x):
            continue
        shoulders = np.r_[x[seed - outer:seed - radius], x[seed + radius + 1:seed + outer + 1]]
        baseline = float(np.median(shoulders))
        local = x[seed - radius:seed + radius + 1] - baseline
        up, down = max(0., float(local.max())), max(0., float(-local.min()))
        dominance = (up - down) / max(up + down, 1e-12)
        anchor = int(seed - radius + int(np.argmax(np.abs(local))))
        if anchors and (t[anchor] - t[anchors[-1]]) < 200:
            continue
        if anchor < half_template or anchor + half_template >= len(x):
            continue
        template = x[anchor - half_template:anchor + half_template + 1] - baseline
        template /= max(float(np.max(np.abs(template))), 1e-12)
        templates.append(template)
        anchors.append(anchor)
        votes.append(1 if dominance >= .20 else -1 if dominance <= -.20 else 0)
    count = len(votes)
    result.update(event_count=count, beat_indices=anchors)
    required = max(8, int(round(.20 * (t[-1] - t[0]) / 1000)))
    if count < required:
        result["reason"] = "too_few_qrs_candidates"
        return result
    votes = np.asarray(votes)
    positive, negative = int(np.sum(votes == 1)), int(np.sum(votes == -1))
    confident = positive + negative
    fraction = confident / count
    agreement = max(positive, negative) / confident if confident else float("nan")
    score = (positive - negative) / confident if confident else float("nan")
    matrix = np.asarray(templates)
    template = np.median(matrix, axis=0)
    norm = float(np.linalg.norm(template))
    denominator = np.linalg.norm(matrix, axis=1) * norm
    correlations = np.divide(matrix @ template, denominator, out=np.zeros(count), where=denominator > 1e-12)
    correlation = float(np.median(correlations))
    up, down = max(0., float(template.max())), max(0., float(-template.min()))
    template_dominance = (up - down) / max(up + down, 1e-12)
    template_sign = 1 if template_dominance >= .20 else -1 if template_dominance <= -.20 else 0
    result.update(positive_votes=positive, negative_votes=negative,
                  uncertain_beats=count - confident, confident_beat_fraction=fraction,
                  sign_agreement=agreement, polarity_score=score,
                  template_correlation=correlation, template_dominance=template_dominance,
                  template_direction={1: "positive", -1: "negative", 0: "undetermined"}[template_sign],
                  raw_polarity="ambiguous", reason="direction_not_confirmed")
    if fraction < .70:
        result["reason"] = "biphasic_or_uncertain_qrs_extrema"
    elif correlation < .70:
        result["reason"] = "inconsistent_beat_template"
    elif agreement < .90:
        result["reason"] = "mixed_beat_directions"
    elif template_sign and template_sign == (1 if positive > negative else -1):
        result.update(raw_polarity="positive" if template_sign > 0 else "negative",
                      reason="consistent_beatwise_direction")
    else:
        result["reason"] = "template_direction_not_confirmed"
    return result
