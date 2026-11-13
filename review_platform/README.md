# ECG review on Cloudflare Pages

This is the only maintained browser application. It is a static site: no Python web server, shared SQLite database, npm build, CDN, or task-data API. Waveforms and annotations are read locally and are not uploaded. The retired local review servers are no longer included in this branch.

## Open historical annotations

Open the website and drop the supplied **history ZIP** directly onto the import card. No reviewer name is required for history. Browse **segment duration → Random / Difficult**, then filter R peaks, direction decisions, method preferences, participant, task ID or reviewer.

The viewer is read-only. It displays the original waveform, saved native R-peak positions, unreadable intervals, direction/preference labels, notes, status and original annotation rows. Five-second pages, longer views and a full-segment overview are available. Different reviewers, protocols and coordinate systems remain distinct. Method preferences are not direction ground truth; supplementary peaks from direction review retain their separate coordinate semantics.

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
