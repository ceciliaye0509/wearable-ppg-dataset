"""Build a blinded raw-ECG review packet for independent R-apex annotation.

This deliberately does *not* run an ECG detector or draw any candidate peak.
It is the annotation stage that must precede a valid estimate of detector
location error, missed/extra beats, or the change in RMSSD after re-location.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/daily_hrv_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, find_peaks, sosfiltfilt


WINDOW_SEC = 300.0
DETAIL_SEC = 20.0
DETAIL_CENTERS_SEC = (75.0, 225.0)
SEVERE_ARTIFACT_RANGE_UV = 5_000.0
EXCLUDED_PARTICIPANTS = {"P2", "P5", "P14", "P16", "P17"}


def _candidate_starts(times_ms: np.ndarray) -> np.ndarray:
    """5-min starts contained in uninterrupted raw-ECG runs."""
    gap_ms = 250.0
    breaks = np.flatnonzero(np.diff(times_ms) > gap_ms) + 1
    bounds = np.r_[0, breaks, times_ms.size]
    starts: list[float] = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        first, last = float(times_ms[lo]), float(times_ms[hi - 1])
        latest = last - WINDOW_SEC * 1000.0
        if latest < first:
            continue
        grid = np.arange(first, latest + 1.0, 30_000.0)
        starts.extend(grid.tolist())
        if not grid.size or grid[-1] != latest:
            starts.append(latest)
    return np.asarray(starts, dtype=float)


def _choose_starts(times_ms: np.ndarray, count: int) -> np.ndarray:
    candidates = _candidate_starts(times_ms)
    if candidates.size < count:
        raise ValueError(f"only {candidates.size} complete ECG windows are available")
    return candidates[np.linspace(0, candidates.size - 1, count).round().astype(int)]


def _choose_starts_avoiding(times_ms: np.ndarray, count: int, forbidden_starts_ms: np.ndarray) -> np.ndarray:
    """Choose evenly spread 5-min windows that do not overlap frozen audits."""
    candidates = _candidate_starts(times_ms)
    forbidden = np.asarray(forbidden_starts_ms, dtype=float)
    if forbidden.size:
        candidates = candidates[np.all(np.abs(candidates[:, None] - forbidden[None, :]) >= WINDOW_SEC * 1000.0, axis=1)]
    if candidates.size < count:
        raise ValueError(f"only {candidates.size} non-overlapping complete ECG windows are available")
    return candidates[np.linspace(0, candidates.size - 1, count).round().astype(int)]


def _plot_raw_only(
    signal: np.ndarray,
    times_ms: np.ndarray,
    start_ms: float,
    participant: str,
    review_id: str,
    out_path: Path,
) -> None:
    lo = int(np.searchsorted(times_ms, start_ms, side="left"))
    hi = int(np.searchsorted(times_ms, start_ms + WINDOW_SEC * 1000.0, side="right"))
    x, t = signal[lo:hi], times_ms[lo:hi]
    if x.size < 50:
        raise ValueError(f"{review_id}: empty ECG interval")
    sec = (t - start_ms) / 1000.0

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=(0.8, 1.25, 1.25))
    overview = fig.add_subplot(grid[0])
    overview.plot(sec, x, color="#30363d", linewidth=0.35)
    overview.set(xlim=(0, WINDOW_SEC), ylabel="ECG (µV)")
    overview.set_title(f"Blinded raw-ECG review {review_id} | {participant} | 5-min context")
    overview.grid(alpha=0.2)

    for ax, center in zip((fig.add_subplot(grid[1]), fig.add_subplot(grid[2])), DETAIL_CENTERS_SEC, strict=True):
        left, right = center - DETAIL_SEC / 2, center + DETAIL_SEC / 2
        keep = (sec >= left) & (sec <= right)
        ax.plot(sec[keep], x[keep], color="#30363d", linewidth=0.9)
        ax.set(xlim=(left, right), ylabel="ECG (µV)")
        ax.set_title(f"Annotate every visually resolvable R apex: {left:.0f}–{right:.0f} s")
        ax.grid(alpha=0.25)
    ax.set_xlabel("seconds relative to the 5-min window start")
    fig.text(
        0.01,
        0.002,
        "No algorithmic ECG peaks are shown. Record each R apex as an absolute Unix-ms timestamp in the supplied CSV template; "
        "use confidence=high/medium/low and leave unresolvable beats unannotated.",
        fontsize=9,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _independent_raw_r_candidates(signal: np.ndarray, times_ms: np.ndarray) -> np.ndarray:
    """Propose the dominant raw-QRS timing extreme without an ECG detector.

    This is deliberately a lightweight morphology proposal for an *assisted*
    review pass, not a replacement ground truth.  It chooses the dominant
    QRS polarity in the clip (upward R in a normal-looking lead; downward QS
    extreme in an inverted lead), finds sharp 5--25 Hz morphology events, then
    snaps each proposal to the corresponding raw extreme within +/-60 ms.  It
    does not call NeuroKit or Pan--Tompkins, so it cannot reveal the output
    under evaluation.
    """
    x = np.asarray(signal, dtype=float)
    t = np.asarray(times_ms, dtype=float)
    if x.size < 30:
        return np.empty(0, dtype=np.int64)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not dt.size:
        return np.empty(0, dtype=np.int64)
    fs = float(1000.0 / np.median(dt))
    if not np.isfinite(fs) or fs < 60.0:
        return np.empty(0, dtype=np.int64)
    hi = min(25.0, 0.45 * fs)
    if hi <= 5.0:
        return np.empty(0, dtype=np.int64)
    try:
        filtered = sosfiltfilt(butter(3, [5.0, hi], btype="bandpass", fs=fs, output="sos"), x)
    except ValueError:
        return np.empty(0, dtype=np.int64)
    robust_scale = float(np.median(np.abs(filtered - np.median(filtered)))) * 1.4826
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        return np.empty(0, dtype=np.int64)
    raw_baseline = float(np.median(x))
    upper_excursion = float(np.quantile(x, 0.995) - raw_baseline)
    lower_excursion = float(raw_baseline - np.quantile(x, 0.005))
    polarity = 1.0 if upper_excursion >= lower_excursion else -1.0
    seeds, _ = find_peaks(
        polarity * filtered,
        distance=max(1, int(round(0.35 * fs))),
        prominence=0.70 * robust_scale,
    )
    radius = max(1, int(round(0.060 * fs)))
    candidates = []
    for seed in seeds:
        lo, hi_idx = max(0, int(seed) - radius), min(x.size, int(seed) + radius + 1)
        candidates.append(lo + int(np.argmax(polarity * x[lo:hi_idx])))
    candidates = np.unique(np.asarray(candidates, dtype=np.int64))
    if not candidates.size:
        return candidates

    # Retain only conspicuous excursions in the selected polarity. This is a
    # morphology gate, not a detector score, and keeps the proposals
    # intentionally editable by the reviewer.
    raw_scale = float(np.median(np.abs(x - raw_baseline))) * 1.4826
    heights = polarity * (x[candidates] - raw_baseline)
    directional_heights = heights[heights > 0]
    if not directional_heights.size:
        return np.empty(0, dtype=np.int64)
    height_gate = max(
        2.5 * raw_scale,
        0.25 * float(np.quantile(directional_heights, 0.95)),
    )
    return candidates[heights >= height_gate]


def _emrich2023_raw_apex_candidates(signal: np.ndarray, times_ms: np.ndarray) -> np.ndarray:
    """Return Emrich2023 events snapped to the locally dominant raw QRS apex.

    This is an editable reviewer aid, not a final label.  The raw-apex snap
    makes the visual marker represent the fiducial the reviewer is asked to
    judge, rather than the detector's filtered-signal sample.
    """
    x = np.asarray(signal, dtype=float)
    t = np.asarray(times_ms, dtype=float)
    if x.size < 30 or t.size != x.size:
        return np.empty(0, dtype=float)
    steps = np.diff(t)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if not steps.size:
        return np.empty(0, dtype=float)
    fs = float(1000.0 / np.median(steps))
    try:
        clean = nk.ecg_clean(x, sampling_rate=fs, method="emrich2023")
        _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="emrich2023")
        events = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    except Exception:
        return np.empty(0, dtype=float)
    events = events[(events >= 0) & (events < x.size)]
    if not events.size:
        return np.empty(0, dtype=float)
    baseline = float(np.median(x))
    polarity = 1.0 if np.quantile(x, 0.995) - baseline >= baseline - np.quantile(x, 0.005) else -1.0
    radius = max(1, int(round(0.150 * fs)))
    refined: list[int] = []
    for event in events:
        left, right = max(0, int(event) - radius), min(x.size, int(event) + radius + 1)
        refined.append(left + int(np.argmax(polarity * x[left:right])))
    # The annotation app stores candidate *sample indices*, not timestamps.
    # Returning absolute times here would put every marker outside the canvas.
    return np.unique(np.asarray(refined, dtype=int))


def _has_severe_amplitude_artifact(signal: np.ndarray) -> bool:
    """Flag clips where morphology proposals would be misleading.

    This is a technical signal-quality guard, not a clinical classification.
    A 1st--99th percentile span above 5 mV is far outside the ordinary ECG
    scale in these recordings and, in the selected audit clips, reflects
    electrode/contact or motion artefact rather than stable QRS morphology.
    """
    x = np.asarray(signal, dtype=float)
    if x.size < 30:
        return True
    return bool(np.quantile(x, 0.99) - np.quantile(x, 0.01) > SEVERE_ARTIFACT_RANGE_UV)


def _write_readme(
    out_dir: Path,
    *,
    assisted: bool,
    aid_method: str,
    excerpt_description: str,
    severe_pages_omitted: int = 0,
) -> None:
    heading = "# Assisted QRS-timing review packet" if assisted else "# Blinded R-apex review packet"
    aid_name = "Emrich2023 detector proposals snapped to the local raw-QRS apex" if aid_method == "emrich2023" else "independent raw-morphology candidate marks"
    mode_text = (
        f"The annotation app starts each clip with {aid_name}. "
        "Depending on lead polarity, a mark can sit on an upward R apex or a downward dominant QRS extreme. "
        "Reviewers must delete false candidates, add missed QRS timing fiducials, and move incorrect positions. "
        "Use this packet for efficient human-corrected labels, not as a blinded validation pass."
        if assisted else
        "Each page contains raw Polar ECG only: no cleaned signal, detector output, peak candidate, or RR interval is displayed. "
        f"Two reviewers should independently enter every visually resolvable R apex in {excerpt_description}. "
        "Pages automatically marked as severe raw-ECG artefact are excluded: do not annotate them."
    )
    severe_page_note = (
        f"{severe_pages_omitted} pages meeting the severe raw-ECG artefact rule were removed from the app; "
        "their IDs are retained in `auto_excluded_severe_artifact_clips.csv`."
        if severe_pages_omitted
        else "Pages automatically marked as severe raw-ECG artefact are excluded: do not annotate them."
    )
    text = f"""{heading}

{mode_text}

Use `rpeak_annotations_template.csv` as a long-form table. For each click,
record:

- `reviewer_id`: reviewer-specific identifier;
- `review_id`: value from `review_manifest.csv`;
- `participant` and `window_start_unix_ms`: copied from the manifest;
- `manual_rpeak_abs_unix_ms`: the raw ECG timestamp at the selected QRS timing fiducial (upward R apex or, in an inverted lead, downward dominant QRS extreme);
- `confidence`: `high`, `medium`, or `low`（annotation app 会按页面分别保存）；
- `notes`: optional morphology/artifact note.

{severe_page_note}

Do not use the assisted packet as an independent blinded assessment. A later
reconciliation step will match annotations to the stored Pan--Tompkins output
and calculate location error, missed/extra beat counts, and RMSSD change.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _write_annotation_app(
    out_dir: Path,
    manifest: list[dict[str, object]],
    clips: list[dict[str, object]],
    *,
    assisted: bool,
    aid_method: str = "raw_morphology",
    storage_key: str | None = None,
) -> None:
    """Write a local click-to-annotate page.

    The assisted mode displays independent raw-morphology proposals, while the
    blinded mode remains detector/candidate-free. Both store reviewer choices
    only in browser local storage and make no network calls.
    """
    import json

    payload = json.dumps({"manifest": manifest, "clips": clips}, separators=(",", ":"))
    aid_label = "Emrich2023 → raw-apex" if aid_method == "emrich2023" else "raw-morphology"
    heading = f"Assisted ECG QRS-timing correction ({aid_label})" if assisted else "Blinded raw-ECG R-apex annotation"
    hint = (
        f"Red marks are {aid_label} proposals, not Pan--Tompkins. They follow the dominant QRS direction (upward or downward) in each clip. "
        "Click a red mark to remove it; click an unmarked dominant QRS extreme to add it. "
        "Use Reset candidates to restore a clip's original proposals; export only after reviewing every clip."
        if assisted else
        "Click each visually resolvable R apex. Each click snaps to the nearest raw ECG sample. No detector output is present. "
        "Use the arrow buttons to navigate; export only after finishing an independent pass."
    )
    # v2 starts a clean assisted session: v1 proposed misleading marks in
    # severe-artifact clips. The old browser-local v1 annotations remain
    # untouched rather than being overwritten.
    storage_key = storage_key or (
        f"assisted-ecg-r-apex:{out_dir.name}:v2" if assisted else f"blinded-ecg-r-apex:{out_dir.name}:v1"
    )
    reset_button = '<button id="reset">Reset candidates</button>' if assisted else ""
    click_handler = (
        """const prior=ann[key()]||[];let k=-1,best=Infinity;prior.forEach((v,i)=>{let distance=Math.abs(d.t[v]-d.t[j]);if(distance<best){best=distance;k=i}});if(best<=80){history.push({type:'remove',clip:key(),value:prior[k],at:k});prior.splice(k,1);ann[key()]=prior}else{history.push({type:'add',clip:key(),value:j});ann[key()]=[...prior,j].sort((a,b)=>a-b)};save();draw()"""
        if assisted else
        """history.push({type:'add',clip:key(),value:j});ann[key()]=[...(ann[key()]||[]),j].sort((a,b)=>a-b);save();draw()"""
    )
    page = f"""<!doctype html>
<meta charset="utf-8"><title>{heading}</title>
<style>
body{{font:15px system-ui,sans-serif;margin:20px;color:#17202a;max-width:1180px}}button,input,select{{font:inherit;margin:4px}}canvas{{display:block;width:100%;border:1px solid #85929e;cursor:crosshair}}#meta{{margin:12px 0}}#count{{font-weight:600}}.hint{{color:#566573;max-width:1000px}}
</style>
<h1>{heading}</h1>
<p class="hint">{hint}</p>
<label>Reviewer ID <input id="reviewer" autocomplete="off"></label>
<label>Confidence <select id="confidence"><option selected>high</option><option>medium</option><option>low</option></select></label>
<label>Review status <select id="reviewstatus"><option selected>complete</option><option>partial</option><option>unresolvable</option></select></label>
<button id="prev">← Previous</button><button id="next">Next →</button><button id="zoomprev">← 5 s</button><button id="zoomnext">5 s →</button><button id="zoomfull">Full clip</button><button id="undo">Undo</button><button id="clear">Clear clip</button>{reset_button}<button id="download">Download CSV</button>
<div id="meta"></div><canvas id="plot" width="1100" height="480" aria-label="Raw ECG annotation plot"></canvas><p id="count"></p>
<script>
const DATA={payload}; const pad={{l:64,r:20,t:26,b:54}}, cv=document.querySelector('#plot'),ctx=cv.getContext('2d');
let idx=0, zoomBlock=-1, ann=JSON.parse(localStorage.getItem('{storage_key}')||'{{}}'), confidenceByClip=JSON.parse(localStorage.getItem('{storage_key}:confidence-by-clip')||'{{}}'), statusByClip=JSON.parse(localStorage.getItem('{storage_key}:status-by-clip')||'{{}}'), history=[];
const key=()=>DATA.clips[idx].id; const save=()=>localStorage.setItem('{storage_key}',JSON.stringify(ann)); const saveConfidence=()=>localStorage.setItem('{storage_key}:confidence-by-clip',JSON.stringify(confidenceByClip)); const saveStatus=()=>localStorage.setItem('{storage_key}:status-by-clip',JSON.stringify(statusByClip));
const defaults=d=>d.suggested||[]; if({str(assisted).lower()}){{let changed=false;DATA.clips.forEach(d=>{{if(!Array.isArray(ann[d.id])){{ann[d.id]=[...defaults(d)];changed=true}}}});if(changed)save()}}
const nearest=(arr,x)=>{{let lo=0,hi=arr.length-1;while(lo<hi){{let m=(lo+hi)>>1;if(arr[m]<x)lo=m+1;else hi=m}}return lo&&Math.abs(arr[lo-1]-x)<Math.abs(arr[lo]-x)?lo-1:lo}};
function draw(){{const d=DATA.clips[idx],t=d.t,x=d.x,W=cv.width,H=cv.height,w=W-pad.l-pad.r,h=H-pad.t-pad.b;const fullLeft=t[0],fullRight=t[t.length-1],blockMs=5000,blocks=Math.max(1,Math.ceil((fullRight-fullLeft)/blockMs));if(zoomBlock>=blocks)zoomBlock=blocks-1;const viewLeft=zoomBlock<0?fullLeft:fullLeft+zoomBlock*blockMs,viewRight=zoomBlock<0?fullRight:Math.min(fullRight,viewLeft+blockMs);const first=nearest(t,viewLeft),last=nearest(t,viewRight),visible=x.slice(first,last+1);let mn=Math.min(...visible),mx=Math.max(...visible),p=(mx-mn)*.08||1;mn-=p;mx+=p;document.querySelector('#confidence').value=confidenceByClip[key()]||'high';document.querySelector('#reviewstatus').value=statusByClip[key()]||(d.autoSkip?'unresolvable':'complete');
document.querySelector('#meta').textContent=`${{d.id}} · ${{d.participant}} · ${{d.startSec.toFixed(0)}}–${{d.endSec.toFixed(0)}} s in its 5-min window · ${{zoomBlock<0?'full clip':`5 s zoom ${{zoomBlock+1}}/${{blocks}}`}}${{d.autoSkip ? ' · 严重 raw ECG 伪迹：无需人工标注（自动排除）' : (d.aidStatus==='suppressed_severe_artifact' ? ' · Candidate aid suppressed: severe amplitude artifact' : '')}}`;
ctx.clearRect(0,0,W,H);ctx.strokeStyle='#d5d8dc';ctx.lineWidth=1;ctx.font='12px system-ui';ctx.fillStyle='#17202a';
for(let i=0;i<=4;i++){{let y=pad.t+h*i/4,v=mx-(mx-mn)*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.fillText(v.toFixed(0),6,y+4)}}
for(let i=0;i<=4;i++){{let X=pad.l+w*i/4,v=viewLeft+(viewRight-viewLeft)*i/4;ctx.beginPath();ctx.moveTo(X,pad.t);ctx.lineTo(X,H-pad.b);ctx.stroke();ctx.fillText((v/1000).toFixed(2)+' s',X-15,H-27)}}
const X=j=>pad.l+(t[j]-viewLeft)/(viewRight-viewLeft)*w,Y=j=>pad.t+(mx-x[j])/(mx-mn)*h;ctx.strokeStyle='#273746';ctx.lineWidth=1.2;ctx.beginPath();for(let j=first;j<=last;j++){{j===first?ctx.moveTo(X(j),Y(j)):ctx.lineTo(X(j),Y(j))}}ctx.stroke();
(ann[key()]||[]).filter(j=>j>=first&&j<=last).forEach(j=>{{ctx.fillStyle='#c0392b';ctx.beginPath();ctx.arc(X(j),Y(j),4,0,2*Math.PI);ctx.fill();}});document.querySelector('#count').textContent=d.autoSkip?'该页已自动排除；请直接前往下一页。':`${{(ann[key()]||[]).length}} active QRS-timing marks in this clip`}}
cv.addEventListener('click',e=>{{const r=cv.getBoundingClientRect(),d=DATA.clips[idx];if(d.autoSkip)return;const w=cv.width-pad.l-pad.r,fullLeft=d.t[0],fullRight=d.t[d.t.length-1],viewLeft=zoomBlock<0?fullLeft:fullLeft+zoomBlock*5000,viewRight=zoomBlock<0?fullRight:Math.min(fullRight,viewLeft+5000);let u=(e.clientX-r.left)/r.width*cv.width;u=Math.max(pad.l,Math.min(cv.width-pad.r,u));let ms=viewLeft+(u-pad.l)/w*(viewRight-viewLeft),j=nearest(d.t,ms);{click_handler}}});
document.querySelector('#prev').onclick=()=>{{idx=(idx+DATA.clips.length-1)%DATA.clips.length;zoomBlock=-1;draw()}};document.querySelector('#next').onclick=()=>{{idx=(idx+1)%DATA.clips.length;zoomBlock=-1;draw()}};document.querySelector('#zoomprev').onclick=()=>{{const d=DATA.clips[idx],blocks=Math.max(1,Math.ceil((d.t[d.t.length-1]-d.t[0])/5000));zoomBlock=zoomBlock<0?blocks-1:Math.max(0,zoomBlock-1);draw()}};document.querySelector('#zoomnext').onclick=()=>{{const d=DATA.clips[idx],blocks=Math.max(1,Math.ceil((d.t[d.t.length-1]-d.t[0])/5000));zoomBlock=zoomBlock<0?0:Math.min(blocks-1,zoomBlock+1);draw()}};document.querySelector('#zoomfull').onclick=()=>{{zoomBlock=-1;draw()}};
document.querySelector('#confidence').onchange=()=>{{confidenceByClip[key()]=document.querySelector('#confidence').value;saveConfidence()}};
document.querySelector('#reviewstatus').onchange=()=>{{statusByClip[key()]=document.querySelector('#reviewstatus').value;saveStatus()}};
document.querySelector('#undo').onclick=()=>{{let h=history.pop();if(!h)return;let a=ann[h.clip]||[];if(h.type==='add'){{let i=a.indexOf(h.value);if(i>=0)a.splice(i,1)}}else if(h.type==='remove')a.splice(h.at,0,h.value);else a=[...h.before];ann[h.clip]=a;save();draw()}};document.querySelector('#clear').onclick=()=>{{history.push({{type:'replace',clip:key(),before:[...(ann[key()]||[])]}});ann[key()]=[];save();draw()}};
{("document.querySelector('#reset').onclick=()=>{history.push({type:'replace',clip:key(),before:[...(ann[key()]||[])]});ann[key()]=[...defaults(DATA.clips[idx])];save();draw()};" if assisted else "")}
document.querySelector('#download').onclick=()=>{{let id=document.querySelector('#reviewer').value.trim();if(!id){{alert('Enter Reviewer ID first');return}}let lines=['reviewer_id,review_id,participant,window_start_unix_ms,manual_rpeak_abs_unix_ms,confidence,review_status,notes'];for(const d of DATA.clips){{const status=statusByClip[d.id]||(d.autoSkip?'unresolvable':'complete'),marks=ann[d.id]||[],confidence=confidenceByClip[d.id]||'high';if(d.autoSkip||!marks.length&&status!=='complete'){{lines.push([id,d.reviewId,d.participant,d.windowStart,'',confidence,status,d.autoSkip?'unresolvable_raw_ecg_chunk:'+d.id:''].join(','));continue}}for(const j of marks){{lines.push([id,d.reviewId,d.participant,d.windowStart,d.t[j].toFixed(3),confidence,status,''].join(','));}}}}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')+'\\n'],{{type:'text/csv'}}));a.download='rpeak_annotations_'+id+'.csv';a.click();URL.revokeObjectURL(a.href)}};draw();
</script>"""
    (out_dir / "annotation_app.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("Multisite-PPG/raw_data"))
    # P2 is explicitly excluded from the published rawslots training cohort;
    # do not mix it into a validation of that cohort's ECG label pipeline.
    parser.add_argument("--participants", nargs="+", default=["P1", "P13", "P15"])
    parser.add_argument("--windows-per-participant", type=int, default=4)
    parser.add_argument("--extra-windows-participants", nargs="*", default=[])
    parser.add_argument("--extra-windows-per-participant", type=int, default=0)
    parser.add_argument(
        "--avoid-manifest",
        action="append",
        type=Path,
        default=[],
        help="Existing review manifest whose five-minute windows must not overlap this packet. Repeatable.",
    )
    parser.add_argument(
        "--omit-severe-artifact-pages",
        action="store_true",
        help="remove severe raw-ECG artefact pages from the reviewer app; retain a separate audit CSV",
    )
    parser.add_argument("--detail-seconds", type=float, default=DETAIL_SEC)
    parser.add_argument(
        "--detail-centers-sec",
        default=",".join(str(value) for value in DETAIL_CENTERS_SEC),
        help="comma-separated centres within each 5-minute window",
    )
    parser.add_argument("--mode", choices=("blinded", "assisted"), default="blinded")
    parser.add_argument(
        "--aid-method",
        choices=("raw_morphology", "emrich2023"),
        default="raw_morphology",
        help="candidate marks to show in assisted mode",
    )
    parser.add_argument(
        "--manifest-in",
        type=Path,
        help="reuse these exact frozen review windows instead of selecting new ones",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/blinded_ecg_rpeak_review_v2_training_cohort"))
    args = parser.parse_args()
    if args.windows_per_participant < 1:
        raise ValueError("--windows-per-participant must be positive")
    if args.extra_windows_per_participant < 0:
        raise ValueError("--extra-windows-per-participant cannot be negative")
    if args.detail_seconds <= 0:
        raise ValueError("--detail-seconds must be positive")
    detail_centers = tuple(float(value) for value in args.detail_centers_sec.split(",") if value.strip())
    if not detail_centers:
        raise ValueError("--detail-centers-sec must contain at least one centre")
    half_detail = args.detail_seconds / 2.0
    if any(center - half_detail < 0 or center + half_detail > WINDOW_SEC for center in detail_centers):
        raise ValueError("detail excerpts must remain inside each 5-minute window")
    selected_participants = list(args.participants)
    frozen_manifest = None
    if args.manifest_in:
        frozen_manifest = pd.read_csv(args.manifest_in, dtype={"participant": str, "review_id": str})
        required = {"review_id", "participant", "window_start_unix_ms"}
        if missing := required - set(frozen_manifest.columns):
            raise ValueError(f"{args.manifest_in} is missing columns: {sorted(missing)}")
        selected_participants = list(dict.fromkeys(frozen_manifest.participant.tolist()))
    invalid = set(selected_participants) & EXCLUDED_PARTICIPANTS
    if invalid:
        raise ValueError(f"excluded participants cannot enter the review packet: {sorted(invalid)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    app_clips: list[dict[str, object]] = []
    excluded_clips: list[dict[str, object]] = []
    extra_participants = set(args.extra_windows_participants)
    unknown_extra = extra_participants - set(args.participants)
    if unknown_extra:
        raise ValueError(f"extra-window participants must also be selected: {sorted(unknown_extra)}")
    forbidden_by_participant: dict[str, list[float]] = {}
    for manifest_path in args.avoid_manifest:
        prior = pd.read_csv(manifest_path, dtype={"participant": str})
        required = {"participant", "window_start_unix_ms"}
        if missing := required - set(prior.columns):
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        for participant, start in zip(prior.participant, prior.window_start_unix_ms, strict=True):
            if pd.notna(start):
                forbidden_by_participant.setdefault(str(participant), []).append(float(start))
    for participant in selected_participants:
        path = args.raw_root / participant / f"{participant}_polar_ecg_raw.npz"
        with np.load(path, allow_pickle=True) as z:
            signal = np.asarray(z["ecg_uv"], dtype=float)
            times = np.asarray(z["time_ms"], dtype=float)
        valid = np.isfinite(signal) & np.isfinite(times)
        signal, times = signal[valid], times[valid]
        if times.size < 50:
            raise ValueError(f"{participant}: insufficient finite raw ECG samples")
        if frozen_manifest is not None:
            chosen = frozen_manifest[frozen_manifest.participant == participant]
            starts = chosen.window_start_unix_ms.to_numpy(dtype=float)
            review_ids = chosen.review_id.astype(str).tolist()
        else:
            window_count = args.windows_per_participant + (
                args.extra_windows_per_participant if participant in extra_participants else 0
            )
            starts = _choose_starts_avoiding(
                times, window_count, np.asarray(forbidden_by_participant.get(participant, []), dtype=float)
            )
            review_ids = [f"{participant}_W{rank:02d}" for rank in range(1, len(starts) + 1)]
        for start, review_id in zip(starts, review_ids, strict=True):
            plot_path = args.out_dir / f"{review_id}_raw_only.png"
            _plot_raw_only(signal, times, float(start), participant, review_id, plot_path)
            manifest.append({
                "review_id": review_id,
                "participant": participant,
                "window_start_unix_ms": round(float(start), 3),
                "window_end_unix_ms": round(float(start + WINDOW_SEC * 1000.0), 3),
                "detail_excerpt_seconds": ";".join(f"{center-half_detail:g}-{center+half_detail:g}" for center in detail_centers),
                "raw_ecg_path": str(path.resolve()),
                "raw_only_plot_path": str(plot_path.resolve()),
            })
            lo = int(np.searchsorted(times, start, side="left"))
            hi = int(np.searchsorted(times, start + WINDOW_SEC * 1000.0, side="right"))
            local_times, local_signal = times[lo:hi], signal[lo:hi]
            for clip_number, center in enumerate(detail_centers, start=1):
                left, right = center - half_detail, center + half_detail
                keep = (local_times >= start + left * 1000.0) & (local_times <= start + right * 1000.0)
                clip_times = np.asarray(local_times[keep], dtype=float)
                clip_signal = np.asarray(local_signal[keep], dtype=float)
                clip_id = f"{review_id}_C{clip_number}"
                severe = _has_severe_amplitude_artifact(clip_signal)
                if severe and args.omit_severe_artifact_pages:
                    excluded_clips.append({
                        "clip_id": clip_id,
                        "review_id": review_id,
                        "participant": participant,
                        "start_sec_in_window": left,
                        "end_sec_in_window": right,
                        "reason": "severe_raw_ecg_artifact_1st_to_99th_percentile_span_gt_5000uv",
                    })
                    continue
                app_clips.append({
                    "id": clip_id, "reviewId": review_id, "participant": participant,
                    "windowStart": round(float(start), 3), "startSec": left, "endSec": right,
                    "t": clip_times.round(3).tolist(), "x": clip_signal.round(2).tolist(),
                    "aidStatus": (
                        "suppressed_severe_artifact"
                        if args.mode == "assisted" and severe
                        else "shown"
                    ),
                    "autoSkip": bool(severe),
                    "suggested": (
                        []
                        if args.mode != "assisted" or severe
                        else (
                            _emrich2023_raw_apex_candidates(clip_signal, clip_times).tolist()
                            if args.aid_method == "emrich2023"
                            else _independent_raw_r_candidates(clip_signal, clip_times).tolist()
                        )
                    ),
                })

    pd.DataFrame(manifest).to_csv(args.out_dir / "review_manifest.csv", index=False)
    pd.DataFrame(columns=[
        "reviewer_id", "review_id", "participant", "window_start_unix_ms",
        "manual_rpeak_abs_unix_ms", "confidence", "notes",
    ]).to_csv(args.out_dir / "rpeak_annotations_template.csv", index=False)
    pd.DataFrame(excluded_clips, columns=[
        "clip_id", "review_id", "participant", "start_sec_in_window", "end_sec_in_window", "reason",
    ]).to_csv(args.out_dir / "auto_excluded_severe_artifact_clips.csv", index=False)
    _write_readme(
        args.out_dir,
        assisted=args.mode == "assisted",
        aid_method=args.aid_method,
        excerpt_description=f"the {len(detail_centers)} detailed {args.detail_seconds:g}-s excerpts per window",
        severe_pages_omitted=len(excluded_clips),
    )
    _write_annotation_app(args.out_dir, manifest, app_clips, assisted=args.mode == "assisted", aid_method=args.aid_method)
    print(f"Wrote {len(manifest)} {args.mode} review windows to {args.out_dir}")


if __name__ == "__main__":
    main()
