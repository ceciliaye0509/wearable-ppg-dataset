import sys
from pathlib import Path
import unittest
import gzip
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from polarity_audit.direction_v3 import measure_direction_v3


class DirectionV3SafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs = 130.
        cls.t = np.arange(0, 60, 1/cls.fs)
        cls.x = np.zeros_like(cls.t)
        for beat in np.arange(.5, 60, .8):
            cls.x += (900*np.exp(-((cls.t-beat)/.014)**2)
                      -300*np.exp(-((cls.t-beat-.035)/.014)**2))

    def measure(self, x, **kwargs):
        return measure_direction_v3(x, self.t*1000, self.fs, **kwargs)

    def test_polarity_equivariance_keeps_anchors_and_quality(self):
        pos, neg = self.measure(self.x), self.measure(-self.x)
        self.assertEqual(pos['raw_polarity'], 'positive')
        self.assertEqual(neg['raw_polarity'], 'negative')
        self.assertEqual(pos['beat_indices'], neg['beat_indices'])
        self.assertAlmostEqual(pos['supported_fraction'], neg['supported_fraction'])
        for a, b in zip(pos['tiles'], neg['tiles']):
            for aa, bb in zip(a['bands'], b['bands']):
                self.assertAlmostEqual(aa['template_dominance'], -bb['template_dominance'])
                self.assertAlmostEqual(aa['template_correlation'], bb['template_correlation'])

    def test_slow_large_baseline_and_dc_do_not_change_direction(self):
        drifting = self.x+800*np.sin(2*np.pi*.25*self.t)+10000
        self.assertEqual(self.measure(drifting)['raw_polarity'], 'positive')

    def test_real_direction_change_is_not_hidden_by_majority(self):
        mixed = self.x*np.where(self.t < 45, 1., -1.)
        result = self.measure(mixed)
        self.assertEqual(result['raw_polarity'], 'ambiguous')
        self.assertEqual(result['cross_check_reason'], 'opposite_directions_across_tiles')

    def test_clear_opposite_time_tile_vetoes_a_dominant_beat_count(self):
        x = np.zeros_like(self.t)
        for beat, sign in [(b,1) for b in np.arange(.5,45,.5)]+[(b,-1) for b in np.arange(45.5,60,1.)]:
            x += sign*(900*np.exp(-((self.t-beat)/.014)**2)
                       -300*np.exp(-((self.t-beat-.035)/.014)**2))
        result = self.measure(x)
        self.assertEqual(result['primary_direction'],'positive')
        self.assertEqual(result['raw_polarity'],'ambiguous')
        self.assertEqual(result['cross_check_status'],'conflict')
        self.assertEqual(result['cross_check_reason'],'opposite_directions_across_tiles')

    def test_equiphasic_qrs_is_not_forced_into_a_sign(self):
        x = np.zeros_like(self.t)
        for beat in np.arange(.5, 60, .8):
            x += (700*np.exp(-((self.t-beat+.020)/.010)**2)
                  -700*np.exp(-((self.t-beat-.020)/.010)**2))
        self.assertEqual(self.measure(x)['raw_polarity'], 'ambiguous')

    def test_severe_artifact_flatline_and_noise_abstain(self):
        self.assertEqual(self.measure(4000*np.sin(2*np.pi*self.t))['raw_polarity'], 'severe_artifact')
        self.assertEqual(self.measure(np.zeros_like(self.x))['raw_polarity'], 'low_signal')
        self.assertEqual(self.measure(np.random.default_rng(12).normal(0,100,len(self.x)))['raw_polarity'], 'ambiguous')

    def test_timestamp_jitter_does_not_create_an_empty_fifth_tile(self):
        ms = self.t*1000+np.linspace(0, .4, len(self.t))
        result = measure_direction_v3(self.x, ms, self.fs)
        self.assertEqual(len(result['tiles']), 4)
        self.assertEqual(result['raw_polarity'], 'positive')

    def test_invalid_inputs_are_rejected_without_mutation(self):
        original = self.x.copy()
        result = self.measure(self.x, diagnostics=True)
        self.assertEqual(result['raw_polarity'], 'positive')
        np.testing.assert_array_equal(original, self.x)
        bad = self.t*1000
        bad[20] = bad[19]
        self.assertEqual(measure_direction_v3(self.x,bad,self.fs)['raw_polarity'], 'low_signal')
        with self.assertRaises(ValueError):
            measure_direction_v3(self.x[:-1],self.t*1000,self.fs)

    def test_weak_confirmation_does_not_erase_a_clear_development_example(self):
        packet_path = Path(__file__).resolve().parents[2]/'outputs/polarity_pairwise_disagreements_v1/comparison_packet.json.gz'
        with gzip.open(packet_path,'rt') as f:
            packet = json.load(f)
        # Already seen P4 example, explicitly a development regression rather
        # than fresh direction ground truth. The secondary band is near-balanced.
        task = next(t for t in packet['tasks'] if t['id']=='pair_P4_R002_B00035')
        result = measure_direction_v3(task['x'],np.array(task['t'])*1000,125.)
        inverse = measure_direction_v3(-np.array(task['x']),np.array(task['t'])*1000,125.)
        self.assertEqual(result['raw_polarity'],'positive')
        self.assertEqual(result['cross_check_status'],'limited_support')
        self.assertEqual(result['cross_check_direction'],'ambiguous')
        self.assertEqual(inverse['raw_polarity'],'negative')
        self.assertEqual(inverse['cross_check_status'],'limited_support')


if __name__ == '__main__':
    unittest.main()
