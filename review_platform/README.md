# ECG review on Cloudflare Pages

This is the only maintained browser application. It is a static site: no Python web server, shared SQLite database, npm build, CDN, or task-data API. Waveforms and annotations are read locally and are not uploaded. The retired local review servers are no longer included in this branch.

## Open historical annotations

Enter the **reviewer name** saved in the package (for example `Qiyu`), then drop the supplied **history ZIP** or open its extracted folder. Only that reviewer’s already annotated tasks are listed; unknown reviewers and unannotated tasks do not appear. The reviewer is distinct from the participant (for example P12). Browse **segment duration → Random / Difficult**, then filter annotation category, participant or task ID. Changing a duration/category filter selects a matching waveform; an empty result clears the plot.

Saved human peaks and unreadable intervals load directly from the package. For R-peak tasks, click to add/remove, drag a red marker to move it to a native sample, right-click to delete, or undo the last change. Select **Cannot reliably read** and click the two interval endpoints; peaks inside remain intact. Notes and completion status can be saved. View spans include 1, 2, 5, 10 and 20 seconds and full segment; longer than the source segment still shows the available segment. Native time gaps/reversals are retained, and no snapping, sorting or resampling of waveform data is performed. Direction/preference decisions remain visible with their original semantics rather than becoming R-peak decisions.

Changes save automatically in a separate reviewer/package-scoped IndexedDB session with revision conflict detection. The original ZIP and original annotation rows remain unchanged. **Download progress backup** creates `.ecghistoryprogress`; load the original history package first, then **Restore history backup**. **Download updated package ZIP** produces a directly reopenable history package containing effective updated peaks/masks, notes and statuses in `history_manifest.json`, revision history in `edits/progress.json`, and original rows/manifests for provenance. Only this reviewer’s annotated tasks appear in its manifest. Original source files for provenance may remain in the ZIP. An edited package has its own manifest hash/session identity. History backups are distinct from legacy `.ecgprogress` backups; neither replaces the other’s format.

The supplied 2026-11-13 history package contains 697 existing tasks and 697 bound waveforms (696 complete, one explicitly in progress). It excludes the active 36-window review and does not read its app, data or progress. The existing verified P3/W04/C1 correction replaces the preceding saved mark. Only latest effective tables are included; superseded raw-export archives are not duplicated inside this package. The source databases, original archived ZIP and frozen GT are unchanged.

Sampling groups follow the earlier organized archive. Some older records defaulted to `Random` despite insufficient sampling evidence; the viewer explicitly displays `uncertain_default_random`. This category is not proof of simple random sampling or easy signal quality. Thirty-three minute-level views remain provenance of their parent three-minute tasks, not additional independent tasks.

**Private history ZIPs are supplied separately and must not be committed or uploaded to Pages.**

## Existing reviewer tasks and progress

Reviewer ZIPs can also be dropped directly, or opened as extracted folders. Existing `.ecgreview` descriptors, `.ecgprogress` backups, task IDs, native coordinates, IndexedDB name and export schema remain compatible. Use the same website origin, reviewer and original descriptor to resume work. A grouped reviewer ZIP can use `browse_index.json`: descriptor bytes and their hash remain unchanged, so existing progress identity is preserved. `Unclassified` is shown when no source-established group exists.

Changing Pages domain or browser changes local storage; export progress before switching. Do not clear browser data to refresh the site. Reload twice if an old service worker serves the preceding UI.

## Deploy

For Cloudflare Git integration: repository branch **hrv-followup**, root directory **review_platform**, no build command, output directory **dist**. Alternatively upload a ZIP containing only the contents of `dist/` using Pages Direct Upload. No annotation package, research mapping, database or token belongs in the deployment.

The existing production origin is `https://daily-hrv-review.pages.dev/`; changing GitHub source alone does not update a Direct Upload deployment. A new deployment must publish the updated static assets at that same origin to preserve browser progress.

See [Cloudflare Static HTML documentation](https://developers.cloudflare.com/pages/framework-guides/deploy-anything/).

## Validate / rebuild

```sh
node --test tests/*.test.cjs
node tests/verify_zip.cjs /path/to/reviewer-package.zip
```

`tests/verify_package.cjs` verifies extracted conventional reviewer folders. Archive loading supports stored and DEFLATE ZIP entries, verifies entry CRCs and waveform SHA-256, bounds expanded size, and rejects unsafe paths. The browser's native `DecompressionStream` is used; no external decompressor is fetched.

To organize an existing reviewer ZIP without changing its descriptor or waveform bytes:

```sh
python3 scripts/organize_package.py --source original.zip --output grouped.zip \
  --groups-csv verified_groups.csv
# CSV: task_id,duration_sec,sampling_group
```

History packages are built from private frozen annotation exports and their exact source waveform packets. `scripts/build_history_package.py --help` lists explicit inputs. It never consults a live review database. Version deduplication retains logical task, reviewer, semantic type and waveform-coordinate identity; explicit newer verified CSV corrections replace only changed matching tasks. A provenance ledger records source hashes, replacements, derived views and waveform-binding failures; no waveform is silently invented or sorted. The research root and original packets are not part of this repository.
