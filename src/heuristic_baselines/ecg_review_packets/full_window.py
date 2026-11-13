"""Offline full-window packet validation. No server or annotation storage."""
from __future__ import annotations
import json
import math
try:
    from .peak_assistance import waveform_hash
except ImportError:
    from peak_assistance import waveform_hash
KIND = 'full_window_hrv_validation_v1'
METHOD = 'rodrigues2021_local_qrs_extreme_50ms_full300s'

def is_full_window(clip: dict) -> bool:
    return clip.get('reviewProtocolKind') == KIND

def page_count(clip: dict) -> int:
    return math.ceil(clip['windowDurationSec'] / clip['reviewPageSeconds'])

def normalize_pages(value: object, clip: dict) -> list[int]:
    if not isinstance(value, list) or len(value) > page_count(clip):
        raise ValueError('Reviewed pages must be a list of page numbers')
    if any((type(i) is not int or not 0 <= i < page_count(clip) for i in value)):
        raise ValueError('Reviewed page number is outside this window')
    return sorted(set(value))

def prepare_full_window_supplement(extra: dict, packet_sha256: str) -> None:
    protocol = extra.get('assistanceProtocol', {})
    if extra.get('reviewProtocolKind') != KIND or protocol.get('version') != 1 or protocol.get('method') != METHOD or (not isinstance(protocol.get('generatedAt'), str)):
        raise ValueError('Unrecognized full-window assistance protocol')
    batches = extra.get('reviewBatches', [])
    batch_ids = {b['id'] for b in batches}
    if not batch_ids or len(batch_ids) != len(batches):
        raise ValueError('Missing or duplicate full-window batches')
    for clip in extra['clips']:
        t, x, peaks = (clip.get('t'), clip.get('x'), clip.get('suggested'))
        if not isinstance(t, list) or not isinstance(x, list) or len(t) < 50 or (len(t) != len(x)) or any((type(v) not in (int, float) or not math.isfinite(v) for v in t + x)) or any((a >= b for a, b in zip(t, t[1:]))) or (t[0] != 0) or (not 299.5 <= t[-1] <= 300):
            raise ValueError('Full-window waveform must cover the complete five-minute interval')
        if not is_full_window(clip) or clip.get('windowDurationSec') != 300 or clip.get('reviewPageSeconds') != 5 or (clip.get('annotationScope') != 'reviewer') or (clip.get('blindReview') is not False) or (clip.get('annotationMode') != 'algorithm_assisted') or (clip.get('reviewBatch') not in batch_ids) or (not isinstance(peaks, list)) or any((type(i) is not int or not 0 <= i < len(t) for i in peaks)) or (peaks != sorted(set(peaks))) or (clip.get('restorePeaks') != peaks) or (clip.get('aidAvailable') is not bool(peaks)):
            raise ValueError('Malformed full-window metadata or peak suggestions')
        digest = waveform_hash(clip)
        if clip.get('assistanceWaveformSha256') != digest:
            raise ValueError('Full-window waveform hash mismatch')
        expected_gaps = [[i, i + 1] for i, (a, b) in enumerate(zip(t, t[1:])) if b - a > 0.2501]
        if clip.get('samplingGaps') != expected_gaps:
            raise ValueError('Full-window sampling gaps are missing or inconsistent')
        clip['assistance'] = {'method': METHOD, 'overlaySha256': packet_sha256, 'waveformSha256': digest, 'generatedAt': protocol['generatedAt'], 'status': clip.get('assistanceStatus', 'ok')}
    for batch in batches:
        if sum((c['reviewBatch'] == batch['id'] for c in extra['clips'])) != batch.get('target'):
            raise ValueError('Full-window batch count mismatch')
