# Beat-wise polarity candidate and independent audit

## Current workspace: fresh direction labels (2026-10-29)

### Independent three-minute flip acceptance

Open `http://127.0.0.1:8767/flip-acceptance`. Both methods were snapshotted into
`analysis/ecg_direction_flip_acceptance_20261029_v1/frozen_sources/` before
scanning all 46,915 source blocks. 35,532 passed the unchanged technical checks;
the shared source in P17 was excluded in full. There were 29 minute-level flip
decision differences. Earlier development/review minutes, their neighboring
contexts, and overlapping selected windows were excluded before sampling.

The frozen blind packet contains eleven new three-minute intervals. The planned
maximum was twelve; non-overlap and a two-interval participant cap limited the
available differing intervals, and no filter was relaxed to fill the quota.
Sampling strata and method predictions stay private. Label every minute in each
interval from the original ECG, using the full three-minute, one-minute and
five-second views. No answer is preselected and no R peaks need editing.

In-progress saves, restore, version conflicts, CSV export and a fresh
`acceptance_labels.sqlite3` database keep these answers separate from previous
annotations. Comparisons stay hidden until all intervals are complete. An
additional flip needs negative human labels and a fully negative three-minute
context; positive, mixed, balanced or unassessable new flips prevent acceptance.
Passing this small targeted check is not a population error estimate. No result
automatically changes production corrections. Packet, prediction and frozen
source checksums are checked when the service starts.

### P17 provenance replacements

The current default workspace is
`analysis/ecg_direction_p17_replacement_20261029_v1/`, prepared by
`build_p17_replacement.py`. Open `http://127.0.0.1:8767/?set=replacement` to
label only four new P17 minutes. The shared P19 source file is excluded in full,
along with every selected P19 interval, waveform duplicates of P19 tasks and
previously available P17 tasks. Four minutes were randomly sampled from 1,383
technically eligible minutes before running any prediction; amplitude and
algorithm agreement are not sampling criteria.

Every one of the original 175 tasks remains available with unchanged waveforms.
All 175 direction labels and their revision history migrate verbatim to a new
packet-bound database; original storage and a cutover backup remain available.
The nine interval reviews retain their existing separate database.

The **corrected main evaluation** has 156 retained originals and four fresh P17
minutes, still 20 participants × 8 minutes. The four affected original P17
records remain historical and never count toward corrected scores. Corrected
results stay hidden until all four replacement labels are complete. The original
main, corrected main, replacement subset, difficult audit and optional checks
are reported separately, never pooled. CSV exports contain `in_corrected_main`,
`superseded_in_corrected_main` and `replaces_id` for explicit membership and
provenance. Earlier labels are never copied onto replacement waveforms.

Old v1, v3 and Rodrigues-assisted predictions are frozen privately before new
labels. The exploratory combined candidate is also evaluated privately for the
four new tasks, without changing any served detector or production flip rule.
Original NPZ files, participant assignments and R-peak marks are not modified.
The prior random workspace below remains the historical source of this repair.

**http://127.0.0.1:8767/** now collects positive/negative QRS direction directly
from the original, unflipped ECG. No method identity, algorithm prediction,
correction proposal or suggested peak is sent to the labeling interface.
Balanced QRS, genuinely mixed direction, unreadable signal and reviewer
uncertainty have separate labels. Complete requires an explicit answer; saved
labels, confidence and notes have version checks and revision history.

The user retired the earlier direction/preference records from this benchmark.
The 36 old direction ratings, 31 paired preferences and 11 three-method choices
are backed up under `analysis/ecg_direction_truth_20261029_v1/archived_previous_records/`.
They are not copied, prefilled or scored. Existing R-peak annotations and production correction intervals stay in
their original stores. The old preference workspace is historical only.

The historical random set is
`/Users/jiqiyu/Desktop/Daily_HRV/analysis/ecg_direction_truth_random20_20261029_v2/`.
It contains **160 random main minutes (20 participants × 8)**, **12 separate
difficult minutes**, and the **3 retained optional checks**. Choose the review
set above the participant list. Every participant has eight selectable minutes
in the main set; progress is shown within the selected set.

The main sample is simple random sampling without replacement within each
participant, seed `20261029`, from complete, technically valid source minutes.
All 160 identities are frozen in `frozen_selection.json` before prediction.
Sampling never uses direction, amplitude, algorithm agreement, or human answers.
Technically invalid/partial minutes and the existing separate tasks are excluded;
amplitude artifacts remain eligible. Missing detector context or detector failure
stays in the sample as an algorithm abstention. Equal allocation to participants
is not duration weighting across all recordings. This is not an unseen external
validation set.

All three method implementations are unchanged, with source checksums saved.
Predictions are frozen privately before labeling. Labels saved during the current
absolute-direction round migrate verbatim, with versions and revision history,
after waveform identity checks; earlier preferences do not migrate. The original
database and a cutover backup remain available.

After completing a review set, the reviewer can explicitly open its comparison.
Each set has its own result gate and scores; **no pooled overall score** is
computed. The table reports correct-and-decided, opposite, abstained, accuracy
among decided cases and coverage, using only newly completed signed labels.
Balanced/mixed labels have a separate signed-output count; unreadable/unsure
are excluded from signed direction accuracy. The API refuses to reveal the
comparison for an unfinished set. The difficult and optional snippets remain
targeted development checks. Isolated random minutes assess minute direction,
not the three-minute automatic-flip policy.

```bash
/Users/jiqiyu/Desktop/Daily_HRV/.venv/bin/python -B \
  src/heuristic_baselines/polarity_audit/serve_direction_truth.py --local-session
# Read-only current comparison, including an explicit zero-label state:
/Users/jiqiyu/Desktop/Daily_HRV/.venv/bin/python -B \
  src/heuristic_baselines/polarity_audit/serve_direction_truth.py --score-only
```

## Targeted three-method follow-up (2026-10-29)

### Current amplitude-gate follow-up

Open `http://127.0.0.1:8767/?set=difficult` for the unchanged twelve difficult
direction minutes. The original 160 completed random labels remain available.
The next-review links on the main page also open the separate interval follow-up
at `http://127.0.0.1:8767/gate-review`.

`direction_gate_review.py` examines all 160 fixed random minutes with three
configurations: original v3, a private diagnostic copy with only the whole-minute
amplitude gate bypassed, and four independent 15-second windows with the original
5,000 µV gate retained. The local candidate needs at least three agreeing signed
windows and no confidently opposite window. Disconnected windows are never
concatenated. The settings are fixed before this diagnostic run; scoring uses
the already-seen labels and is development evidence.

The original v3 gives 105 correct, zero opposite and ten abstentions among 115
signed references. Gate bypass gives 111 correct but also signed outputs on
13/45 manually unreadable minutes. The local-window candidate gives 106 correct,
zero opposite and nine abstentions, with one signed output on an unreadable
minute. It recovers two gated cases but loses one original successful case.
Neither diagnostic has replaced any production direction or flip rule.

A subsequent exploratory fallback preserves every original v3 decision and
consults the local candidate only when original v3 returned `severe_artifact`.
It changes only P2 R019 B00000 and P7 R000 B00129, giving 107/115 correct, zero
opposite, eight abstentions and zero signed outputs on the 45 unreadable
references. This fallback was defined after inspecting the first diagnostic;
`gated_only_fallback_development.json` explicitly records this post hoc status.
It is a development candidate, not independent validation or a deployed rule.

The nine signed-reference minutes rejected by the original amplitude gate are
available for targeted interval review. Label each 5-second interval Readable,
Unreadable or Unsure, then choose the direction of the readable portions.
Amplitude spans are shown as diagnostics, not automatic readability labels.
Candidate directions and earlier direction answers remain hidden. In-progress
save, completion checks, restore, conflict protection and revision history use
the separate `analysis/ecg_direction_gate_review_20261029_v1/interval_reviews.sqlite3`
database. Main direction records, R peaks and source ECG are not changed.
Export these records with the interval page's Export interval labels link.

```bash
/Users/jiqiyu/Desktop/Daily_HRV/.venv/bin/python -B \
  src/heuristic_baselines/polarity_audit/direction_gate_review.py
```

The diagnostic builder refuses existing output directories. The direction server
automatically attaches the generated follow-up on its next restart; 8765 and
8766 are not restarted. The original twelve-minute difficult audit remains
blind to algorithm outputs until its own labels are complete.

Historical preference workflow below: retired from the current benchmark.

The new local workspace at **http://127.0.0.1:8767/** reuses the 36 saved
absolute direction ratings and retains the 31 earlier A/B preferences. A prior
preference is never converted into direction truth or a new three-method answer.
Only unresolved core flip changes or conflicting signed predictions enter the
priority queue. The first packet has **11 priority minutes** and **3 optional
agreement checks**, hidden by default; the previously reviewed P4 block 265 is
retained separately as prior preference evidence and is not asked again.

The three anonymous panels compare old v1, template v3 and Rodrigues-assisted
direction plus their proposed interval action. An original raw waveform stays
visible, all views share scales, and Full minute / 5-second controls synchronize
the panels. A/B/C assignments are stable per task. Pairwise ties and all-method
ties are allowed; no answer is preselected. No peak editing or production sign
correction occurs on this page. New choices have their own packet-bound SQLite
database, conflict protection, revision history and identity-free CSV export.

```bash
/Users/jiqiyu/Desktop/Daily_HRV/.venv/bin/python -B \
  src/heuristic_baselines/polarity_audit/build_targeted_followup.py
/Users/jiqiyu/Desktop/Daily_HRV/.venv/bin/python -B \
  src/heuristic_baselines/polarity_audit/serve_targeted_followup.py --local-session
```

The builder refuses an existing output directory. Its default is
`/Users/jiqiyu/Desktop/Daily_HRV/analysis/ecg_direction_targeted_followup_20261029_v1/`.
The server prints a new access code and uses a separate cookie, leaving 8765
and 8766 running. All old source packets, saved annotations and production
correction intervals are unchanged. Existing-evidence counts are development
agreement, not independent accuracy or a population error-rate estimate.

## ECG direction v3 development (2026-10-28; separate from the website)

`direction_v3.py` keeps QRS direction separate from the completeness of its
cross-checks. Its public `measure_direction_v3` function uses derivative-energy
candidates and steepest-edge alignment, then measures a whole-minute QRS template
in a 5–25 Hz channel. Approximately 15-second templates in 5–25 and 8–35 Hz
provide separate evidence: `supported` or `limited_support`. Weak confirmation
alone does not erase a clear primary direction; explicit opposite confident
tiles/bands do veto it. The existing 5,000 uV raw amplitude gate is retained.
No function flips samples or generates production correction intervals.

ECG-specific literature and the project adaptations/limitations are documented
in the generated report. This is an experimental amplitude-direction convention,
not a clinical lead-reversal diagnosis or a published SOTA polarity algorithm.
The original `beatwise.py`, v1 correction code and served v1/v2 comparison remain
unchanged. Template/slope anchors are not manual R-peak ground truth.

From the repository root, run the isolated development evaluation:

```bash
MPLCONFIGDIR=/private/tmp/hrv_mpl \
  /Users/jiqiyu/Desktop/Daily_HRV/.venv/bin/python -B \
  src/heuristic_baselines/polarity_audit/evaluate_direction_v3.py \
  --preferences /Users/jiqiyu/Downloads/direction_paired_preferences.csv \
  --out-dir /path/to/new-development-output
```

An existing output directory is never replaced. The evaluation checks frozen
v1/v2 reproduction and sign-reversal symmetry on original raw minutes from all
20 participants; it excludes the reserved peak-review five-minute contexts.
Reports include four component ablations, minute/tile/beat diagnostics, code
hashes, a read-only review snapshot, and a Chinese HTML report with figures.
The existing 29 preferences and prior assisted direction reviews are development
evidence, not independent accuracy labels. Stratified controls cannot estimate
whole-dataset error or decisiveness rates. The initial strict prototype is in
`outputs/ecg_direction_v3_development_20261028/`; the direction/support-separated
revision is in `outputs/ecg_direction_v3_development_20261028_r1/`.

## Paired disagreement comparison (added 2026-10-28)

The default page at **http://127.0.0.1:8766/** now opens the paired comparison
when `outputs/polarity_pairwise_disagreements_v1/comparison_packet.json.gz` is
present. The previous 149-task assessment workspace remains at
**http://127.0.0.1:8766/legacy/** with its original database and saved ratings.
Both pages use the same audit login. The 8765 peak workspace is independent and
is not modified by this update.

The new sampling request is **eight randomly drawn one-minute blocks per
participant where the frozen old and new direction categories differ**. Numeric
score changes alone do not qualify. Only complete source blocks of at least
59.5 seconds are eligible; full five-minute contexts reserved for the 40+20
peak-review batches are excluded. When fewer than eight remain, all are used;
zero-eligible participants stay visible as unavailable. There are 127 minutes:
P1/P15 have none; P7 has 7, P11 has 2, P13 has 4, P16/P20 have 5; other
participants have 8. Sampling is without replacement, seed 20261028, with
participant-specific seeds and saved inclusion probabilities.

Review the **same original, unflipped ECG** side by side, with synchronized
full-minute / five-second views, common amplitude scales, and the two methods'
vote locations and frozen direction/interval decisions. A/B assignments are
randomized and balanced within participant, fixed across sessions, and hidden
by the review API. These votes are not verified R-peak annotations. Each
method's vote sign comes from its own internal signal, which can differ from
the sign of the raw sample at the displayed location.

Choose A better / B better / equally reasonable / neither reasonable / cannot
judge, then save in progress or complete. No answer is filled automatically.
Preferences are stored in a **new `paired_preferences.sqlite3`**, with version
conflict checks and revision history; they never overwrite old direction
ratings or peak marks. CSV exports retain A/B choices; the private local packet
maps them to old/new for subsequent analysis. Do not serve the private sampling
manifest or packet as static files.

This is **algorithm-assisted paired preference among disagreements**, not an
independent ground-truth direction accuracy test or an estimate over all ECG.
Do not pool its results into the original old-ambiguous prevalence audit.
No active waveform correction or production label is changed.

To reproduce the already-generated packet, use a *new* output directory (the
builder refuses existing output) and the same code/source versions:

```bash
/Users/jiqiyu/Desktop/Daily_HRV/.venv/bin/python -B \
  src/heuristic_baselines/polarity_audit/build_comparison.py --out-dir /path/to/new-output
```

Start the updated service with the same `serve_audit.py --local-session`
command below. It discovers the installed comparison packet automatically.
The new packet does not replace the frozen 149-task audit packet or protocol.

This version gives each QRS candidate at most one direction vote. Derivative energy identifies a candidate QRS neighborhood; raw, locally baseline-referenced extrema determine its direction. A robust beat template must agree. Direction uncertainty does not establish that QRS timing is unreadable.

The historical `correct_ecg_polarity.py`, all20 v1 correction manifest, active peak-review packet, and existing peak annotations are preserved. New correction intervals are **candidates awaiting validation**. They are not automatically used to flip the current review waveforms or generate final training labels.

## Generate a new version

From the repository root, using the project environment:

```bash
/Users/jiqiyu/Desktop/Daily_HRV/.venv/bin/python -B \
  src/heuristic_baselines/polarity_audit/build_audit.py
```

The default output is `src/heuristic_baselines/outputs/beatwise_polarity_audit_v2/`. An existing non-empty directory is protected; use a new `--out-dir` when intentionally generating another version. The source, parameter settings, random seed, sampling probabilities, and code hashes are recorded in `audit_protocol.json`.

- All 46,915 historical blocks are remeasured, with old/new outcomes in `polarity_blocks_v2.csv` and participant summaries in `participant_comparison.csv`.
- The primary audit is simple random sampling **within each participant's old ambiguous population**, up to eight minutes per participant, seed `20261023`. Rare strata are censuses. The current packet has 114 primary minutes from 18 participants; P1/P15 have no old ambiguous blocks.
- Separate safety controls sample up to two newly uncertain minutes per participant and include every old/new sign reversal: 35 controls in the current packet. They must not be pooled into the primary prevalence estimate.
- Each task contains the original raw minute plus up to 20 seconds of context on either side. Old/new direction predictions and vote counts remain hidden. The original frozen packet is unchanged; the server now derives optional editable QRS timing suggestions at review time.

## Record the two independent assessments

```bash
/Users/jiqiyu/Desktop/Daily_HRV/.venv/bin/python -B \
  src/heuristic_baselines/polarity_audit/serve_audit.py --local-session
```

Open `http://127.0.0.1:8766/` and enter the temporary access code printed by the server. Use the HTTP address, not a local HTML file. The server uses a separate cookie name and a separate `human_audit.sqlite3`, so the existing peak-review workspace stays independent. Progress is shared by clients of this audit host, stale saves are rejected, and previous saved revisions are retained.

Assess only the focus minute (0–60 seconds):

1. **QRS readability:** readable throughout / partially readable / unreadable throughout / unsure. If partial, describe uncertain intervals in Notes.
2. **Direction determinability:** yes / no / unsure. If yes, identify predominantly upward or downward direction in the raw signal as displayed. Readable QRS timing may coexist with balanced or uncertain direction. Direction is not proof of electrode placement or clinical normality.

Save as in progress or complete. Human answers are never filled from automated predictions. Export ratings as CSV. For approved remote access, configure institutional HTTPS/VPN with `ECG_AUDIT_ACCESS_CODE`, `ECG_AUDIT_SESSION_SECRET`, and either TLS certificate/key or an HTTPS reverse proxy (`--secure-cookie`). `--local-session` is restricted to loopback.

### Algorithm-assisted peak editing

The workspace now opens with candidate QRS extrema when no saved peak marks exist. Saved peaks always take precedence. Suggestions use the beat-wise energy detector and raw absolute QRS extrema; they may miss beats, add noise/T-wave events, or choose an inappropriate extremum in biphasic QRS. No direction/readability answer is selected automatically. Severe-amplitude and insufficient-signal gates may produce no suggestions; that is not a human unreadability label.

Use **5 sec →** to open the first five-second view. Click a marker to remove it or click the waveform at the desired time to add a sample-aligned mark. **Restore algorithm suggestions** and **Restore last saved marks** replace only current peak edits, not the two assessments. The latter is unavailable until peak marks have been saved. **Hide peak marks** removes the overlay without deleting marks. Any peak edit clears the explicit **I have checked the current peak marks** checkbox. Complete may be used for the two assessments without claiming complete peak verification.

Ratings retain editable peak sample indices, focus-relative seconds in CSV, algorithm suggestions/method/source hash, whether assistance was shown, origin (algorithm/manual/mixed), and explicit reviewer peak verification. Simply showing a page never stores a completed human answer. Existing saved ratings and revision history remain intact; even an older client saving without peak fields does not delete existing peaks. Packet identity and sampling probabilities are unchanged.

This is now **algorithm-assisted human review**, not a blinded independent test of that detector. Hiding the overlay after seeing it does not erase the assistance flag. Keep this distinction when interpreting candidate agreement. Historical `audit_protocol.json` describes the original packet creation; current review assistance is recorded per saved rating rather than rewriting the frozen packet.

## Analyze completed human review

```bash
/Users/jiqiyu/Desktop/Daily_HRV/.venv/bin/python -B \
  src/heuristic_baselines/polarity_audit/analyze_audit.py
```

This reads the audit database; `--ratings-csv` can read an exported CSV instead. Primary prevalence estimates use inverse inclusion probabilities and separate readability/determinability categories. An approximate stratified finite-population confidence interval is reported. No prevalence estimate is produced until the planned primary sample is complete. Safety controls are counted separately. No final dataset inclusion or algorithm superiority is inferred from automatic outcome changes alone.

`render_previews.py --audit-dir ...` generates raw, prediction-blinded contact sheets for a small first-pass visual check. Any assistant visual assessment must be labelled provisional and must not be stored as a confirmed human rating.
