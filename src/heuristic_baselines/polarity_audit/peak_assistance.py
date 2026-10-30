"""Non-authoritative QRS peak suggestions; never fill either human assessment."""
import hashlib
from pathlib import Path

import numpy as np

from beatwise import PARAMETERS, measure_beatwise

METHOD = PARAMETERS['method'] + ':raw_abs_qrs_extremum'
SOURCE_SHA256 = hashlib.sha256(Path(__file__).with_name('beatwise.py').read_bytes()).hexdigest()


def suggest_peaks(task):
    t, x = np.asarray(task['t']), np.asarray(task['x'])
    focus = np.flatnonzero((t >= task['focusStartSec']) & (t < task['focusEndSec']))
    result = dict(method=METHOD, source_sha256=SOURCE_SHA256, peak_indices=[],
                  notice='No usable suggestions. Inspect manually; this is not an unreadability decision.')
    if len(focus) < 2:
        return result
    dt = np.median(np.diff(t[focus]))
    if not np.isfinite(dt) or dt <= 0:
        return result
    measured = measure_beatwise(x[focus], t[focus] * 1000, 1 / dt)
    result['peak_indices'] = [int(focus[i]) for i in measured['beat_indices']]
    if result['peak_indices']:
        result['notice'] = ('Candidate QRS extrema, not confirmed R peaks. Check missing/extra beats and placement, '
                            'especially at boundaries and in noisy areas. No direction or readability answer is preselected.')
    return result
