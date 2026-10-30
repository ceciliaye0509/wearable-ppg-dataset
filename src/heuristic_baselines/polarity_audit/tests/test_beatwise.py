import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beatwise import measure_beatwise


def synthetic():
    t = np.arange(0, 60, 1 / 130)
    x = 15 * np.sin(2 * np.pi * .15 * t)
    for center in np.arange(.5, 59.5, 1):
        x += 1000 * np.exp(-.5 * ((t - center) / .012) ** 2)
        x -= 450 * np.exp(-.5 * ((t - center - .035) / .012) ** 2)
        x += 250 * np.exp(-.5 * ((t - center - .30) / .060) ** 2)
    return t * 1000, x


class BeatwiseTests(unittest.TestCase):
    def test_each_period_contributes_one_qrs_not_t_wave_votes(self):
        t, x = synthetic()
        r = measure_beatwise(x, t, 130)
        self.assertEqual(r['raw_polarity'], 'positive', r)
        self.assertTrue(56 <= r['event_count'] <= 61, r['event_count'])
        self.assertEqual(r['negative_votes'], 0)

    def test_sign_inversion_reverses_direction_not_candidate_times(self):
        t, x = synthetic()
        a, b = measure_beatwise(x, t, 130), measure_beatwise(-x, t, 130)
        self.assertEqual(b['raw_polarity'], 'negative')
        self.assertEqual(a['beat_indices'], b['beat_indices'])
        self.assertAlmostEqual(a['polarity_score'], -b['polarity_score'])

    def test_true_direction_change_is_not_forced_to_one_orientation(self):
        t, x = synthetic()
        x[t >= 30000] *= -1
        r = measure_beatwise(x, t, 130)
        self.assertEqual(r['raw_polarity'], 'ambiguous')

    def test_artifact_gate_and_flat_signal_remain_uncorrected(self):
        t, x = synthetic()
        self.assertEqual(measure_beatwise(20 * x, t, 130)['raw_polarity'], 'severe_artifact')
        self.assertEqual(measure_beatwise(np.zeros_like(x), t, 130)['raw_polarity'], 'low_signal')

    def test_non_monotonic_times_are_not_accepted(self):
        t, x = synthetic()
        t[40] = t[39]
        self.assertEqual(measure_beatwise(x, t, 130)['raw_polarity'], 'low_signal')


if __name__ == '__main__':
    unittest.main()
