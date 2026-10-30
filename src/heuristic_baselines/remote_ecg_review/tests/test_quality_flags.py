"""Verify screening reasons, clipped intervals, and actual correction state."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_flags import screen_clip


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.times = 100000 + np.arange(11) * 1000
        self.raw = np.linspace(-100, 100, 11)
        self.flips = np.zeros(11, dtype=bool)
        self.clip = {"t": list(range(11)), "restorePeaks": [2, 5, 8]}

    def block(self, raw="positive", decision="stable_positive", start=100000, end=111000, score="1", span="200"):
        return {"start_ms": str(start), "end_ms": str(end), "raw_polarity": raw,
                "decision": decision, "polarity_score": score, "raw_p01_p99_span_uv": span}

    def test_accepted_block_has_no_exclusion_and_does_not_touch_marks(self):
        flags, summary = screen_clip(self.clip, self.times, self.raw, self.flips, [self.block()])
        self.assertEqual(flags, [])
        self.assertFalse(summary["requiresReview"])
        self.assertEqual(self.clip["restorePeaks"], [2, 5, 8])

    def test_minute_artifact_warning_clips_to_overlap(self):
        blocks = [self.block(end=104000), self.block("severe_artifact", "severe_artifact", 104000, 108000, "nan", "6001"), self.block(start=108000)]
        flags, _ = screen_clip(self.clip, self.times, self.raw, self.flips, blocks)
        self.assertEqual(len(flags), 1)
        self.assertEqual((flags[0]["code"], flags[0]["startIndex"], flags[0]["endIndex"]), ("severe_artifact", 4, 7))
        self.assertEqual(flags[0]["rawSpanUv"], 6001)

    def test_bridged_uncertainty_is_flagged_even_if_already_flipped(self):
        flags, summary = screen_clip(self.clip, self.times, self.raw, ~self.flips,
                                     [self.block("ambiguous", "stable_negative", score="-.6")])
        self.assertEqual(flags[0]["code"], "ambiguous_polarity")
        self.assertTrue(flags[0]["bridged"])
        self.assertEqual(flags[0]["signAgreement"], .8)
        self.assertEqual(flags[0]["flippedSampleCount"], 11)
        self.assertEqual(summary["flippedSampleCount"], 11)

    def test_short_run_low_signal_and_unassessed_are_distinct(self):
        blocks = [self.block("negative", "review_short_polarity_run", end=104000),
                  self.block("low_signal", "review_low_signal", 104000, 108000, "nan")]
        flags, _ = screen_clip(self.clip, self.times, self.raw, self.flips, blocks)
        self.assertEqual({f["code"] for f in flags}, {"short_polarity_run", "low_signal", "unassessed_polarity"})

    def test_segment_artifact_can_exist_inside_an_accepted_minute(self):
        flags, summary = screen_clip(self.clip, self.times, np.linspace(-4000, 4000, 11), self.flips, [self.block()])
        self.assertEqual([f["code"] for f in flags], ["severe_segment_artifact"])
        self.assertGreater(summary["rawP01P99SpanUv"], 5000)
        self.assertEqual(self.clip["restorePeaks"], [2, 5, 8])


if __name__ == "__main__":
    unittest.main()
