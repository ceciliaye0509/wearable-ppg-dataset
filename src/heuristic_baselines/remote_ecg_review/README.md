# ECG Peak Review Workspace

## Amplitude-rule full-window review (2026-10-29)

Select **Amplitude-rule review · full 5 minutes** in **Review batch**. This is
an additive eight-window pilot: six single-source-block alert cases and two
controls, selected with a fixed seed before new human annotation. The case/
control assignment and automatic flags are hidden from every reviewer until
that reviewer's first pass is complete. The waveform uses the frozen
2026-10-29 polarity combination, and editable Rodrigues ±50 ms suggestions.

1. Sign in with your usual reviewer name. Choose any available participant.
2. Review all 60 five-second pages, correct peaks and mark unreadable intervals.
   **Cannot reliably read** remains an interval tool; it does not clear peaks.
3. Complete and save the whole first pass. This batch stays on the current
   window instead of automatically advancing.
4. Click **Reveal automatic screening after first-pass review**. The server
   archives that reviewer's exact saved peaks, intervals, pages and version
   before releasing screening flags. Other reviewers remain independently
   masked. Controls are revealed only at the same step.
5. Inspect the violet block intersections, revise if needed and save a second
   pass. The first-pass snapshot remains immutable and is included in exports
   as `first_pass_annotation_json`, with `screening_revealed_at`.

This pilot audits the existing rule; it does not replace ground truth, change
5,000 µV, or declare an entire five-minute window unreadable automatically.
Intervals shade the intersecting source block, not a known artifact duration.
Future HRV scoring must use matching readable continuous runs, report retained
duration, and distinguish partial-window HRV from a complete five-minute label.
The frozen private selection and deployment/backup verification are in
`analysis/ecg_amplitude_fullwindow_review_20261029_side_v1/` at workspace level.

## Complete five-minute HRV validation (2026-10-29)

Select **Full 5-minute HRV validation** in **Review batch**. The initial cohort
contains one fresh, non-overlapping 300-second context for each of the 20
participants. Selection uses existing direction/quality metadata, before new
human annotations or detector scores. Original ECG files, the active polarity
manifest, earlier packets and shared short-clip annotations are preserved.
Rodrigues2021 with the previously validated ±50 ms local raw-QRS-extremum
locator provides editable suggestions on the full context.

1. Sign in with a consistent reviewer name. Full-window peaks, unreadable
   intervals, page confirmations and revision history belong to **that reviewer
   name**. Another name starts a separate review with the same algorithm
   suggestions. Short-clip batches retain their existing shared behavior.
2. Select any participant. Check each of the **60 five-second pages**. Adjacent
   pages show 250 ms of overlapping context to inspect boundary beats. The
   **Full** overview does not permit peak editing; open a page for precision.
3. Edit peaks and mark unreadable intervals as before. Click **Confirm page &
   save →** to save the whole window as in progress and advance to the first
   unconfirmed page. Merely viewing a page does not confirm it. The page grid
   permits jumping or returning to any page. Saved reviews resume at the first
   unconfirmed page, from any device connected to this host.
4. Click **Complete 5-minute window →** after all 60 pages are confirmed.
   **Save as in progress** remains available at any time. Unreadable labels
   never clear peak marks. Restore last saved marks restores saved page
   confirmations as well; clearing peaks or restoring algorithm suggestions
   resets confirmations so the changed window is checked again.

Participant progress in this batch is **your progress**; the window also shows
other reviewers' saved status and page counts, without loading their peak
marks. Different reviewer names isolate records; this is not a separate user
authentication system. Continue using the existing controlled host/access code.

The regular annotation download includes shared legacy annotations plus your
own full-window records. **Export for comparison** provides all reviewers'
full-window records, preserving reviewer IDs, sample indices, page coverage,
unreadable intervals, waveform hashes and sampling-gap metadata. Use that
export after first-pass annotation to resolve differences. Missing-sample gaps
are shaded and drawn with a broken waveform, not a connecting line. Later HRV
scoring must break RR and adjacent RR differences at gaps and unreadable
intervals; this release does not replace production HRV labels or silently
apply IBI correction to human marks. Numerical HRV acceptance thresholds must
be agreed before viewing validation scores.

The builder is `build_full_window_packet.py --output /a/new/local/directory`.
It writes a frozen selection audit and manifest separately from the additive
`supplement_full300_v1.json.gz` installed in the live packet's `supplements/`
directory. Server restart is required after installing a packet. Existing
database initialization adds `full_window_annotations` and
`full_window_revisions` without rewriting legacy tables.

This browser app displays polarity-corrected ECG segments. It includes both the new algorithm-assisted segments and the previously saved manual annotations from the v4 review. Prior marks load as the editable starting point for those older segments; **Restore previous manual marks** returns to that original set, while new segments restore their anonymous algorithm suggestions. Reviewers can add, remove, clear, and restore peak marks, then save progress or complete the segment. Participant status and annotations are shared through one SQLite database on the review host. Changes are stored separately and do not overwrite the source annotation CSV.

## Supplementary review batches (assistance enabled 2026-10-28)

At the user's request, the original 40 independent-validation and 20 difficult
audit clips now have NeuroKit + ±50 ms local raw-QRS-extremum suggestions.
Detection runs in each clip's full original 300-second context on the existing
v1 polarity-corrected signal. No new detector selection or polarity rule is used.
These batches are now labelled **algorithm-assisted**, not independent blind
validation. Sampling identities, waveforms and the original blind packet stay
unchanged; automatic screening judgments and private selection reasons remain
withheld. The following historical blind behavior still applies without the
explicit assistance overlay.

`build_supplement_assistance.py --output /a/new/path.json` prepares the overlay
without modifying annotations or source packets. The service reads
`supplement_peak_assistance.json` next to its base packet on startup. It validates
every waveform hash and suggestion sample index before enabling assistance.
Saved human peaks, including an explicitly saved empty list, always take
precedence. **Restore algorithm suggestions** replaces only the current peak
edits; it does not change unreadable intervals. Reviewers must save to retain
their edits. No suggestions means no usable algorithm output, not a human
unreadability judgment.

Viewing supplement suggestions records an exposure in `algorithm_assistance_views`,
separately from annotations and their revision history. CSV exports append
`algorithm_assistance_seen`, method, overlay hash and first-view timestamp.
`annotation_mode` remains conservatively `algorithm_assisted` for exposed clips,
even if a future service removes the overlay. Viewing a page never marks a
segment complete. Prior annotation rows and revisions are not rewritten.

The **Review batch** selector separates **Independent blind validation** (40 new
20-second clips, two different unused five-minute contexts per participant),
**Difficult / rule-boundary audit** (20 separate new clips), **Existing review**
(235 unchanged clips), and **All review batches**. Participant progress is shown
for the selected batch. Existing saved peaks, intervals, versions and revision
history remain in the same database. Export annotations includes all saved
batches and appends `review_batch` and `annotation_mode` columns.

Blind batches start with no candidate markers. No algorithm suggestions are sent
by the clip API or available through Restore. **Restore last saved marks** still
restores the reviewer's saved peaks and unreadable intervals. Automatic screening
judgments and challenge selection reasons are withheld from the review page;
their private audit is retained locally for later evaluation. Review the entire
20-second segment, using Full and the five-second navigation controls.

`build_supplement_packet.py` prepares a standalone packet and a private selection
audit without modifying the original packet or annotation database. The service
loads additive `*.json.gz` packets from the original packet directory's
`supplements/` subdirectory on startup. Duplicate IDs or a purported blind packet
containing candidate suggestions cause startup to fail. Adding a packet requires
a service reload; use the existing access code/session secret and back up the
SQLite database first. Do not upload the private audit to the browser or a third
party. The revised 40+20 sampling budget is documented separately; algorithm and
exclusion parameters from the frozen protocol are unchanged.

## Unreadable interval controls

Single-click **Cannot reliably read** to select the interval-marking tool. Drag across the waveform or click the start and end points of the unreadable region. The tool also offers **Mark the whole segment unreadable**. Double-click **Cannot reliably read** to cancel selection mode and any unfinished selection. Existing labeled intervals remain until removed using their **Remove ×** buttons. Save using **Save as in progress** or **Complete segment**.

Unreadable intervals and peak marks are saved independently. Marking or cancelling an unreadable interval never clears peaks, including peaks inside the interval. **Clear peak marks** is an explicit peak-edit action, and **Restore last saved marks** restores the peaks and intervals loaded or most recently saved in the current session. Unsaved changes trigger a warning when leaving the page.

The server upgrades the database by adding a range column without modifying existing peak lists. Legacy whole-segment `unresolvable` records are displayed as a full-segment interval. Requests from older pages that try to clear peaks when marking `unresolvable` instead retain the saved peak list. Each subsequent update archives the preceding saved row in `annotation_revisions`.

CSV exports keep all peak rows and add interval rows. `record_type` is `peak`, `unreadable_interval`, or `status`; `peak_in_unreadable_interval` flags retained peaks inside a labeled region. Interval start/end seconds are relative to the segment; absolute Unix millisecond endpoints are included separately.

## Automatic screening flags

The participant list and segment tabs show which segments overlap screening flags. The review panel lists each affected interval, its reason and measurements, and whether the existing correction actually flipped samples there. Violet bands show automatic flags on the waveform. Amber intervals remain the reviewer's own unreadable labels. Screening flags are read-only guidance: they do not clear or filter peak marks, alter manual unreadable intervals, or change a saved review status.

Flags come from the existing `polarity_blocks.csv`: severe minute-level artifact, unclear direction, low signal, a same-direction run shorter than three classified minute blocks, and any interval without a recorded decision. The raw segment's own P01–P99 amplitude span is checked separately against 5,000 µV, because a brief severe artifact can occur inside a minute that passed the broader screen.

The original run uses `abs((positive_votes - negative_votes) / total_votes) >= 0.80`. This is **at least 90% same-direction votes**, rather than 80%. It also allows one ambiguous/low-signal minute at a time to be bridged between matching signs. Such bridged minutes remain flagged and show their actual flip state. This website update displays the existing results and does not change either rule or the waveform correction.

Exports append `quality_flag_codes`, `quality_flags_json` (intervals and evidence), and `peak_in_quality_flag`. All saved peak rows remain present, including those inside flagged intervals.

`build_review_packet.py` includes screening metadata when preparing a new packet. To enrich an existing frozen packet without changing sample indices, run `quality_flags.py`; its default output is `review_data.quality_candidate.json.gz`. It aligns the saved waveform to raw samples, verifies the existing corrected ECG, and checks that all original fields are preserved before writing the candidate. The annotation database is not modified by this step.

## Start the local review workspace

Run these commands from the repository root on the machine that has the ECG data:

```bash
export ECG_REVIEW_ACCESS_CODE='use a private access code with at least 12 characters'
export ECG_REVIEW_SESSION_SECRET='use a random secret with at least 32 characters'
python src/heuristic_baselines/remote_ecg_review/serve_review.py
```

Then open **http://127.0.0.1:8765/** in a browser. Do not open `index.html` directly, and do not use the CSV export URL as the workspace entry point. If the browser shows `file:///api/export.csv`, navigate to the HTTP address above.

If the review packet has not been generated yet, run this first:

```bash
python src/heuristic_baselines/remote_ecg_review/build_review_packet.py
```

The packet is written to `src/heuristic_baselines/outputs/ecg_rpeak_blind_review_remote_v1/`. It contains sensitive physiological data; keep it off Git and do not upload it to a third party without explicit data-handling approval.

## Remote reviewers

For approved off-device access, run the service on the controlled host that stores the review data, reachable through an institutional VPN or other approved private network. Prefer an institutional HTTPS reverse proxy forwarding to the loopback-only app; add `--secure-cookie` when using that proxy. Direct HTTPS is also supported:

```bash
python src/heuristic_baselines/remote_ecg_review/serve_review.py \
  --host 0.0.0.0 --certfile /path/to/fullchain.pem --keyfile /path/to/private-key.pem
```

Do not expose the service to the public internet or use plain HTTP on an untrusted network. The server refuses non-loopback HTTP by default. For a trusted LAN test only, `--allow-insecure-http` is available explicitly. Keep the SQLite database on one persistent host-local disk; do not share a regular network-drive database between multiple servers.

## Review design and limitations

- Uses the latest polarity correction manifest for all 20 participants; raw NPZ files are not modified.
- Imports prior marks from `outputs/assisted_ecg_rpeak_final_validation_v4/rpeak_annotations_1_p6_w01_c1_c2_unreadable.csv`; the older segments are rebuilt from raw NPZ data with the current polarity correction and clearly labeled as a previous review set.
- Samples up to four non-overlapping five-minute windows per participant, with two 20-second review segments per window.
- Avoids windows used in prior manual review sets and excludes windows containing duplicate timestamps.
- Candidate suggestions use the NeuroKit detector followed by local dominant-QRS-extremum refinement within ±50 ms. The reviewer sees suggested peaks but not the detector identity.
- Since candidates are visible, this is **algorithm-assisted human review**, not an independent blinded detector-accuracy evaluation.
- The CSV export provides reviewer ID, review ID, participant, peak timestamps, confidence, status, and notes for downstream analysis.

The participant IDs are coded, but ECG waveforms remain sensitive physiological data. Authorized browsers receive only the segment currently being reviewed; absolute collection timestamps are not returned to the browser. This is a research annotation tool, not a clinical diagnostic product.
