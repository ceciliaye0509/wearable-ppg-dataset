# Frozen v4.3 model handoff

Start with `../../docs/hrv/WINDOWS_MODEL_HANDOFF.md`.

- `model_comparison.json`: portable M1 specification; not consumed by the legacy runner.
- `folds.json`: colleague participant assignments, with counts recalculated from v4.3.
- `source_window_selection.json`: full-NPZ row indices and original source-window identities.
- `dataset_manifest.json`: frozen data checksums, participant counts, and reference protocol hash.
- `ecg_reference_protocol.json`, `polarity_protocol.json`, `research_joint_decision.json`: frozen research decisions, copied without modification.
- `baseline_primary_scores.csv`, `baseline_freeze_recommendation.json`: current heuristic comparator, not model training scores.
- `source_release_verification.json`, `provenance.json`: validation and source provenance. Historical absolute paths inside provenance records describe the source Mac; they are not runtime paths on Windows.

The 15 participant NPZ files and full annotation sidecars are transferred separately, retaining the entire frozen dataset folder. Do not substitute the older public HF dataset.
