"""Build a fresh, anonymous A/B QRS-timing adjudication packet.

The packet is intentionally small and targeted: four known v3-versus-Emrich+
ProMAC disagreement clips, twelve fresh P4 clips, twelve fresh P6 clips, and
eight fresh clips from other retained participants.  A/B identities are
randomized per clip in the HTML.  The private mapping is retained locally so
the returned human labels can be used to compare candidates reproducibly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/daily_hrv_matplotlib")

import numpy as np
import pandas as pd

from build_blinded_ecg_rpeak_review import _has_severe_amplitude_artifact
from evaluate_assisted_human_confirmed_rpeaks import (
    deployment_candidates_for_clip,
    detector_indices,
    infer_fs,
    pan_tompkins_indices,
    raw_first_pan_fill,
    refine_to_dominant_raw_extreme,
    refine_to_local_qrs_extreme,
)


EXCLUDED_PARTICIPANTS = {"P2", "P5", "P14", "P16", "P17"}
PRIMARY = "emrich2023_local_qrs_extreme_plus_promac_local_qrs_extreme_fill"
V3 = "deployment_5s_raw_morphology_plus_pan_raw_apex_fill"
CONTEXT_MS = 25_000.0
CLIP_MS = 5_000.0
CLIP_OFFSET_MS = 10_000.0
RANDOM_SEED = 20260925


def _ranges(frame: pd.DataFrame) -> list[tuple[str, float, float]]:
    return [
        (str(row.participant), float(row.window_start_unix_ms), float(row.window_end_unix_ms))
        for _, row in frame.iterrows()
        if str(row.participant) not in EXCLUDED_PARTICIPANTS
    ]


def _clip_bounds(manifest: pd.DataFrame, review_id: str, clip_number: int) -> tuple[float, float, str, str]:
    row = manifest.loc[manifest.review_id.astype(str).eq(review_id)].iloc[0]
    pieces = str(row.detail_excerpt_seconds).split(";")
    left_s, right_s = (float(value) for value in pieces[clip_number - 1].split("-"))
    start = float(row.window_start_unix_ms) + left_s * 1000.0
    return start, float(row.window_start_unix_ms) + right_s * 1000.0, str(row.participant), str(row.raw_ecg_path)


def _fresh_clip_starts(signal: np.ndarray, times: np.ndarray, blocked: list[tuple[float, float]], count: int, *, high_complexity: bool) -> list[float]:
    """Choose reproducible non-overlapping central 5-s clips from 25-s contexts."""
    rng = np.random.default_rng(RANDOM_SEED + (1 if high_complexity else 0))
    starts: list[tuple[float, float]] = []
    first = float(times[0]) + CLIP_OFFSET_MS
    last = float(times[-1]) - (CONTEXT_MS - CLIP_OFFSET_MS - CLIP_MS)
    for clip_start in np.arange(first, last, 30_000.0):
        left, right = float(clip_start), float(clip_start + CLIP_MS)
        context_left, context_right = left - CLIP_OFFSET_MS, left - CLIP_OFFSET_MS + CONTEXT_MS
        if any(context_left < end and context_right > begin for begin, end in blocked):
            continue
        lo, hi = np.searchsorted(times, left, side="left"), np.searchsorted(times, right, side="right")
        segment_t, segment_x = times[lo:hi], signal[lo:hi]
        if segment_t.size < 100 or _has_severe_amplitude_artifact(segment_x):
            continue
        starts.append((left, float(hi - lo)))
    if len(starts) < count:
        raise ValueError(f"only {len(starts)} fresh candidate clips available")
    # Spread selections through the available recording; high-complexity is
    # represented by a reproducible shuffle so it does not silently select
    # only one temporal region without an activity label.
    indices = np.arange(len(starts))
    if high_complexity:
        rng.shuffle(indices)
    else:
        indices = np.linspace(0, len(starts) - 1, max(count * 3, count)).round().astype(int)
    selected = sorted({starts[int(i)][0] for i in indices[: max(count * 3, count)]})
    if len(selected) < count:
        selected = [value for value, _ in starts]
    chosen = np.linspace(0, len(selected) - 1, count).round().astype(int)
    return [float(selected[index]) for index in chosen]


def _load_raw(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as source:
        signal = np.asarray(source["ecg_uv"], dtype=float)
        times = np.asarray(source["time_ms"], dtype=float)
    valid = np.isfinite(signal) & np.isfinite(times)
    return signal[valid], times[valid]


def _candidate_sets(raw: np.ndarray, times: np.ndarray, clip_left: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the evaluator's v3 and local Emrich+ProMAC candidates."""
    context_left, context_right = clip_left - CLIP_OFFSET_MS, clip_left - CLIP_OFFSET_MS + CONTEXT_MS
    clip_right = clip_left + CLIP_MS
    lo, hi = np.searchsorted(times, context_left, side="left"), np.searchsorted(times, context_right, side="right")
    context_x, context_t = raw[lo:hi], times[lo:hi]
    if context_t.size < 100:
        raise ValueError("insufficient 25-s context")
    fs = infer_fs(context_t)
    pan = pan_tompkins_indices(context_x, fs)
    pan_refined = refine_to_dominant_raw_extreme(context_x, context_t, pan, fs)
    v3 = raw_first_pan_fill(
        deployment_candidates_for_clip(raw, times, clip_left, clip_right, chunk_ms=5_000.0),
        pan_refined[(pan_refined >= clip_left) & (pan_refined <= clip_right)],
    )
    local: dict[str, np.ndarray] = {}
    for method in ("emrich2023", "promac"):
        indices = detector_indices(context_x, fs, method)
        values = refine_to_local_qrs_extreme(context_x, context_t, indices, fs)
        local[method] = values[(values >= clip_left) & (values <= clip_right)]
    best = raw_first_pan_fill(local["emrich2023"], local["promac"])
    clip_lo, clip_hi = np.searchsorted(times, clip_left, side="left"), np.searchsorted(times, clip_right, side="right")
    return times[clip_lo:clip_hi], raw[clip_lo:clip_hi], np.asarray(v3, dtype=float), np.asarray(best, dtype=float)


def _nearest_indices(times: np.ndarray, values: np.ndarray) -> list[int]:
    if not len(values):
        return []
    indices = np.searchsorted(times, values, side="left")
    indices = np.clip(indices, 0, len(times) - 1)
    left = np.clip(indices - 1, 0, len(times) - 1)
    indices = np.where(np.abs(times[left] - values) < np.abs(times[indices] - values), left, indices)
    return np.unique(indices.astype(int)).tolist()


def _write_app(out_dir: Path, clips: list[dict[str, object]]) -> None:
    payload = json.dumps({"clips": clips}, separators=(",", ":"))
    page = """<!doctype html><meta charset="utf-8"><title>匿名 ECG QRS timing 裁决</title>
<style>body{font:15px system-ui,sans-serif;margin:20px;color:#17202a;max-width:1220px}button,input,select{font:inherit;margin:4px}canvas{display:block;width:100%;border:1px solid #85929e}.editable{cursor:crosshair}.hint{color:#566573;max-width:1080px}h3{margin:18px 0 5px}#count{font-weight:600}</style>
<h1>匿名 ECG QRS-timing 裁决</h1><p class=hint>每个生理 QRS complex 只保留一个一致的 QRS-timing fiducial。上图只显示 A；下图显示 B 和可编辑的绿色最终标注。A/B 身份未命名且每页随机交换。请只在下图删除绿色点或补点。</p>
<label>Reviewer ID <input id=reviewer autocomplete=off></label><label>Final confidence <select id=confidence><option selected>high</option><option>medium</option><option>low</option></select></label><label>Adjudication <select id=decision><option value=edited>人工修正</option><option value=A>A 更合适</option><option value=B>B 更合适</option><option value=unreadable>不可判读</option></select></label><br><button id=prev>← Previous</button><button id=next>Next →</button><button id=useA>Use A as final</button><button id=useB>Use B as final</button><button id=undo>Undo</button><button id=clear>Clear final marks</button><button id=download>Download CSV</button><div id=meta></div>
<h3>A candidate（蓝三角，仅比较）</h3><canvas id=plotA width=1120 height=360></canvas><h3>B candidate + final marks（橙方块 + 绿色圆圈，可编辑）</h3><canvas id=plotB class=editable width=1120 height=360></canvas><p id=count></p>
<script>
const DATA=__PAYLOAD__,pad={l:64,r:20,t:26,b:48},cvA=document.querySelector('#plotA'),cvB=document.querySelector('#plotB'),store='final-ecg-qrs-adjudication-v2';let idx=0,ann=JSON.parse(localStorage.getItem(store)||'{}'),meta=JSON.parse(localStorage.getItem(store+'-meta')||'{}'),history=[];const key=()=>DATA.clips[idx].id,save=()=>{localStorage.setItem(store,JSON.stringify(ann));localStorage.setItem(store+'-meta',JSON.stringify(meta))};let initial=false;DATA.clips.forEach(d=>{if(!Array.isArray(ann[d.id])){ann[d.id]=[...d.B];meta[d.id]={...(meta[d.id]||{}),decision:'B'};initial=true}});if(initial)save();
const nearest=(a,x)=>{let answer=0,best=Infinity;a.forEach((v,i)=>{let gap=Math.abs(v-x);if(gap<best){best=gap;answer=i}});return answer};
function drawPanel(canvas,marks,color,shape,finalMarks){const d=DATA.clips[idx],t=d.t,x=d.x,ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height,w=W-pad.l-pad.r,h=H-pad.t-pad.b;let lo=Math.min(...x),hi=Math.max(...x),p=(hi-lo)*.08||1;lo-=p;hi+=p;ctx.clearRect(0,0,W,H);ctx.strokeStyle='#d5d8dc';ctx.lineWidth=1;ctx.font='12px system-ui';ctx.fillStyle='#17202a';for(let i=0;i<=4;i++){let y=pad.t+h*i/4,v=hi-(hi-lo)*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.fillText(v.toFixed(0),6,y+4)}for(let i=0;i<=4;i++){let X=pad.l+w*i/4,v=t[0]+(t[t.length-1]-t[0])*i/4;ctx.beginPath();ctx.moveTo(X,pad.t);ctx.lineTo(X,H-pad.b);ctx.stroke();ctx.fillText((v/1000).toFixed(2)+' s',X-15,H-20)}const X=j=>pad.l+(t[j]-t[0])/(t[t.length-1]-t[0])*w,Y=j=>pad.t+(hi-x[j])/(hi-lo)*h;ctx.strokeStyle='#273746';ctx.lineWidth=1.2;ctx.beginPath();x.forEach((_,j)=>j?ctx.lineTo(X(j),Y(j)):ctx.moveTo(X(j),Y(j)));ctx.stroke();marks.forEach(j=>{ctx.fillStyle=color;if(shape==='triangle'){ctx.beginPath();ctx.moveTo(X(j),Y(j)-6);ctx.lineTo(X(j)-5,Y(j)+4);ctx.lineTo(X(j)+5,Y(j)+4);ctx.fill()}else ctx.fillRect(X(j)-4,Y(j)-4,8,8)});finalMarks.forEach(j=>{ctx.strokeStyle='#1e8449';ctx.lineWidth=2;ctx.beginPath();ctx.arc(X(j),Y(j),5,0,2*Math.PI);ctx.stroke()})}
function draw(){const d=DATA.clips[idx],m=meta[key()]||{};document.querySelector('#confidence').value=m.confidence||'high';document.querySelector('#decision').value=m.decision||'edited';document.querySelector('#meta').textContent=(idx+1)+'/'+DATA.clips.length+' · '+d.id+' · '+d.participant+' · raw ECG only; candidate identities concealed';drawPanel(cvA,d.A,'#2471a3','triangle',[]);drawPanel(cvB,d.B,'#d68910','square',ann[key()]||[]);document.querySelector('#count').textContent='上图：A（蓝三角） · 下图：B（橙方块）与最终标注（绿色圆圈） · '+(ann[key()]||[]).length+' final marks'}
cvB.onclick=e=>{const d=DATA.clips[idx],r=cvB.getBoundingClientRect(),u=(e.clientX-r.left)/r.width*cvB.width,w=cvB.width-pad.l-pad.r,px=Math.max(pad.l,Math.min(cvB.width-pad.r,u)),time=d.t[0]+(px-pad.l)/w*(d.t[d.t.length-1]-d.t[0]),j=nearest(d.t,time),marks=ann[key()]||[],old=nearest(marks,j);history.push([key(),[...marks]]);if(marks.length&&Math.abs(marks[old]-j)<=8)marks.splice(old,1);else{marks.push(j);marks.sort((a,b)=>a-b)}ann[key()]=marks;save();draw()};
const use=n=>{const d=DATA.clips[idx];history.push([key(),[...(ann[key()]||[])]]);ann[key()]=[...d[n]].sort((a,b)=>a-b);meta[key()]={...(meta[key()]||{}),decision:n};save();draw()};document.querySelector('#useA').onclick=()=>use('A');document.querySelector('#useB').onclick=()=>use('B');document.querySelector('#prev').onclick=()=>{idx=(idx+DATA.clips.length-1)%DATA.clips.length;draw()};document.querySelector('#next').onclick=()=>{idx=(idx+1)%DATA.clips.length;draw()};document.querySelector('#clear').onclick=()=>{history.push([key(),[...(ann[key()]||[])]]);ann[key()]=[];save();draw()};document.querySelector('#undo').onclick=()=>{const h=history.pop();if(h){ann[h[0]]=h[1];save();draw()}};document.querySelector('#confidence').onchange=()=>{meta[key()]={...(meta[key()]||{}),confidence:document.querySelector('#confidence').value};save()};document.querySelector('#decision').onchange=()=>{meta[key()]={...(meta[key()]||{}),decision:document.querySelector('#decision').value};save()};
document.querySelector('#download').onclick=()=>{const who=document.querySelector('#reviewer').value.trim();if(!who){alert('Enter Reviewer ID first');return}const rows=['reviewer_id,review_id,participant,window_start_unix_ms,manual_rpeak_abs_unix_ms,confidence,notes'];for(const d of DATA.clips){const m=meta[d.id]||{},note='adjudication:'+(m.decision||'edited');if(m.decision==='unreadable'){rows.push([who,d.reviewId,d.participant,d.windowStart,'',m.confidence||'low','unresolvable_raw_ecg_chunk:'+d.id].join(','));continue}for(const j of (ann[d.id]||[]))rows.push([who,d.reviewId,d.participant,d.windowStart,d.t[j].toFixed(3),m.confidence||'high',note].join(','))}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')+'\\n'],{type:'text/csv'}));a.download='rpeak_annotations_'+who+'.csv';a.click();URL.revokeObjectURL(a.href)};draw();
</script>"""
    (out_dir / "annotation_app.html").write_text(page.replace("__PAYLOAD__", payload), encoding="utf-8")
    return
    page = f"""<!doctype html><meta charset=\"utf-8\"><title>匿名 ECG QRS timing 裁决</title>
<style>body{{font:15px system-ui,sans-serif;margin:20px;color:#17202a;max-width:1220px}}button,input,select{{font:inherit;margin:4px}}canvas{{display:block;width:100%;border:1px solid #85929e;cursor:crosshair}}.hint{{color:#566573;max-width:1080px}}#count{{font-weight:600}}</style>
<h1>匿名 ECG QRS-timing 裁决</h1>
<p class=hint>目标：每一个生理 QRS complex 只保留一个一致的 QRS-timing fiducial。A/B 为未命名的候选；三角和方块只用于比较，绿色圆点是你的最终标注。可先按 Use A 或 Use B，再逐点增删。若该页无法可靠判读，选择“不可判读”。</p>
<label>Reviewer ID <input id=reviewer autocomplete=off></label><label>Final confidence <select id=confidence><option selected>high</option><option>medium</option><option>low</option></select></label><label>Adjudication <select id=decision><option value=edited>人工修正</option><option value=A>A 更合适</option><option value=B>B 更合适</option><option value=unreadable>不可判读</option></select></label><br>
<button id=prev>← Previous</button><button id=next>Next →</button><button id=useA>Use A</button><button id=useB>Use B</button><button id=undo>Undo</button><button id=clear>Clear final marks</button><button id=download>Download CSV</button><div id=meta></div><canvas id=plot width=1120 height=500></canvas><p id=count></p>
<script>const DATA={payload},pad={{l:64,r:20,t:26,b:54}},cv=document.querySelector('#plot'),ctx=cv.getContext('2d'),key=()=>DATA.clips[idx].id,store='final-ecg-qrs-adjudication-v1';let idx=0,ann=JSON.parse(localStorage.getItem(store)||'{{}}'),meta=JSON.parse(localStorage.getItem(store+'-meta')||'{{}}'),history=[];const save=()=>{{localStorage.setItem(store,JSON.stringify(ann));localStorage.setItem(store+'-meta',JSON.stringify(meta));}};let initialized=false;DATA.clips.forEach(d=>{{if(!Array.isArray(ann[d.id])){{ann[d.id]=[...d.A];meta[d.id]={{...(meta[d.id]||{{}}),decision:'A'}};initialized=true}}}});if(initialized)save();const nearest=(a,x)=>{{let i=0,b=Infinity;a.forEach((v,j)=>{{let z=Math.abs(v-x);if(z<b){{b=z;i=j}}}});return i}};
function draw(){{let d=DATA.clips[idx],t=d.t,x=d.x,W=cv.width,H=cv.height,w=W-pad.l-pad.r,h=H-pad.t-pad.b,mn=Math.min(...x),mx=Math.max(...x),p=(mx-mn)*.08||1;mn-=p;mx+=p;let m=meta[key()]||{{}};document.querySelector('#confidence').value=m.confidence||'high';document.querySelector('#decision').value=m.decision||'edited';document.querySelector('#meta').textContent=`${{idx+1}}/${{DATA.clips.length}} · ${{d.id}} · ${{d.participant}} · raw ECG only; candidate identities concealed`;ctx.clearRect(0,0,W,H);ctx.strokeStyle='#d5d8dc';ctx.lineWidth=1;ctx.font='12px system-ui';ctx.fillStyle='#17202a';for(let i=0;i<=4;i++){{let y=pad.t+h*i/4,v=mx-(mx-mn)*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.fillText(v.toFixed(0),6,y+4)}}for(let i=0;i<=4;i++){{let X=pad.l+w*i/4,v=t[0]+(t[t.length-1]-t[0])*i/4;ctx.beginPath();ctx.moveTo(X,pad.t);ctx.lineTo(X,H-pad.b);ctx.stroke();ctx.fillText((v/1000).toFixed(2)+' s',X-15,H-27)}}const X=j=>pad.l+(t[j]-t[0])/(t[t.length-1]-t[0])*w,Y=j=>pad.t+(mx-x[j])/(mx-mn)*h;ctx.strokeStyle='#273746';ctx.lineWidth=1.2;ctx.beginPath();x.forEach((_,j)=>j?ctx.lineTo(X(j),Y(j)):ctx.moveTo(X(j),Y(j)));ctx.stroke();d.A.forEach(j=>{{ctx.fillStyle='#2471a3';ctx.beginPath();ctx.moveTo(X(j),Y(j)-6);ctx.lineTo(X(j)-5,Y(j)+4);ctx.lineTo(X(j)+5,Y(j)+4);ctx.fill()}});d.B.forEach(j=>{{ctx.fillStyle='#d68910';ctx.fillRect(X(j)-4,Y(j)-4,8,8)}});(ann[key()]||[]).forEach(j=>{{ctx.strokeStyle='#1e8449';ctx.lineWidth=2;ctx.beginPath();ctx.arc(X(j),Y(j),5,0,2*Math.PI);ctx.stroke()}});document.querySelector('#count').textContent=`A: blue triangle · B: orange square · final: green circle · ${{(ann[key()]||[]).length}} final marks`}}
cv.onclick=e=>{{let d=DATA.clips[idx],r=cv.getBoundingClientRect(),u=(e.clientX-r.left)/r.width*cv.width,w=cv.width-pad.l-pad.r;u=Math.max(pad.l,Math.min(cv.width-pad.r,u));let time=d.t[0]+(u-pad.l)/w*(d.t[d.t.length-1]-d.t[0]),j=nearest(d.t,time),a=ann[key()]||[],k=nearest(a,j);if(a.length&&Math.abs(a[k]-j)<=8){{history.push([key(),[...a]]);a.splice(k,1)}}else{{history.push([key(),[...a]]);a.push(j);a.sort((z,q)=>z-q)}}ann[key()]=a;save();draw()}};
const use=n=>{{let d=DATA.clips[idx];history.push([key(),[...(ann[key()]||[])]]);ann[key()]=[...d[n]].sort((a,b)=>a-b);meta[key()]={{...(meta[key()]||{{}}),decision:n}};save();draw()}};document.querySelector('#useA').onclick=()=>use('A');document.querySelector('#useB').onclick=()=>use('B');document.querySelector('#prev').onclick=()=>{{idx=(idx+DATA.clips.length-1)%DATA.clips.length;draw()}};document.querySelector('#next').onclick=()=>{{idx=(idx+1)%DATA.clips.length;draw()}};document.querySelector('#clear').onclick=()=>{{history.push([key(),[...(ann[key()]||[])]]);ann[key()]=[];save();draw()}};document.querySelector('#undo').onclick=()=>{{let h=history.pop();if(h){{ann[h[0]]=h[1];save();draw()}}}};document.querySelector('#confidence').onchange=()=>{{meta[key()]={{...(meta[key()]||{{}}),confidence:document.querySelector('#confidence').value}};save()}};document.querySelector('#decision').onchange=()=>{{meta[key()]={{...(meta[key()]||{{}}),decision:document.querySelector('#decision').value}};save()}};
document.querySelector('#download').onclick=()=>{{let who=document.querySelector('#reviewer').value.trim();if(!who){{alert('Enter Reviewer ID first');return}}let rows=['reviewer_id,review_id,participant,window_start_unix_ms,manual_rpeak_abs_unix_ms,confidence,notes'];for(let d of DATA.clips){{let m=meta[d.id]||{{}},note='adjudication:'+(m.decision||'edited');if(m.decision==='unreadable'){{rows.push([who,d.reviewId,d.participant,d.windowStart,'',m.confidence||'low','unresolvable_raw_ecg_chunk:'+d.id].join(','));continue}}for(let j of (ann[d.id]||[]))rows.push([who,d.reviewId,d.participant,d.windowStart,d.t[j].toFixed(3),m.confidence||'high',note].join(','));}}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')+'\\n'],{{type:'text/csv'}}));a.download='rpeak_annotations_'+who+'.csv';a.click();URL.revokeObjectURL(a.href)}};draw();</script>"""
    (out_dir / "annotation_app.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("Multisite-PPG/raw_data"))
    parser.add_argument("--first-manifest", type=Path, required=True)
    parser.add_argument("--second-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    first, second = pd.read_csv(args.first_manifest), pd.read_csv(args.second_manifest)
    blocked_by_participant: dict[str, list[tuple[float, float]]] = {}
    for participant, start, end in _ranges(pd.concat([first, second], ignore_index=True)):
        blocked_by_participant.setdefault(participant, []).append((start, end))
    requests: list[dict[str, object]] = []
    for source, manifest, review_id, number in (
        ("known_tradeoff_first", first, "P4_W03", 3), ("known_tradeoff_first", first, "P4_W03", 4),
        ("known_tradeoff_second", second, "P6_W01", 2), ("known_tradeoff_second", second, "P6_W01", 4),
    ):
        left, right, participant, raw_path = _clip_bounds(manifest, review_id, number)
        requests.append({"source": source, "participant": participant, "clip_left": left, "raw_path": raw_path})
    for participant, count, complex_flag in (("P4", 12, True), ("P6", 12, True)):
        raw_path = args.raw_root / participant / f"{participant}_polar_ecg_raw.npz"
        signal, times = _load_raw(str(raw_path))
        for left in _fresh_clip_starts(signal, times, blocked_by_participant.get(participant, []), count, high_complexity=complex_flag):
            requests.append({"source": "fresh_targeted_participant", "participant": participant, "clip_left": left, "raw_path": str(raw_path.resolve())})
    for participant in ("P1", "P3", "P7", "P8", "P9", "P10", "P11", "P12"):
        raw_path = args.raw_root / participant / f"{participant}_polar_ecg_raw.npz"
        signal, times = _load_raw(str(raw_path))
        left = _fresh_clip_starts(signal, times, blocked_by_participant.get(participant, []), 1, high_complexity=False)[0]
        requests.append({"source": "fresh_cross_participant", "participant": participant, "clip_left": left, "raw_path": str(raw_path.resolve())})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(RANDOM_SEED)
    clips: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    for ordinal, request in enumerate(requests, start=1):
        raw_path = str(request["raw_path"])
        if raw_path not in raw_cache:
            raw_cache[raw_path] = _load_raw(raw_path)
        raw, times = raw_cache[raw_path]
        left = float(request["clip_left"])
        clip_t, clip_x, v3, best = _candidate_sets(raw, times, left)
        if _has_severe_amplitude_artifact(clip_x):
            raise ValueError(f"selected a severe-artifact clip unexpectedly: {request}")
        review_id = f"ADJ_{ordinal:02d}_{request['participant']}"
        clip_id = f"{review_id}_C1"
        a_is_v3 = bool(rng.integers(0, 2))
        a, b = (v3, best) if a_is_v3 else (best, v3)
        context_start = left - CLIP_OFFSET_MS
        clips.append({
            "id": clip_id, "reviewId": review_id, "participant": request["participant"], "windowStart": round(context_start, 3),
            "t": clip_t.round(3).tolist(), "x": clip_x.round(2).tolist(), "A": _nearest_indices(clip_t, a), "B": _nearest_indices(clip_t, b),
        })
        manifest_rows.append({
            "review_id": review_id, "participant": request["participant"], "window_start_unix_ms": round(context_start, 3),
            "window_end_unix_ms": round(context_start + CONTEXT_MS, 3), "detail_excerpt_seconds": "10-15",
            "raw_ecg_path": raw_path, "selection_source": request["source"], "clip_id": clip_id,
        })
        private_rows.append({"clip_id": clip_id, "A_method": V3 if a_is_v3 else PRIMARY, "B_method": PRIMARY if a_is_v3 else V3, "v3_count": len(v3), "best_count": len(best)})
    pd.DataFrame(manifest_rows).to_csv(args.out_dir / "review_manifest.csv", index=False)
    pd.DataFrame(private_rows).to_csv(args.out_dir / "private_candidate_identity_map.csv", index=False)
    pd.DataFrame(columns=["reviewer_id", "review_id", "participant", "window_start_unix_ms", "manual_rpeak_abs_unix_ms", "confidence", "notes"]).to_csv(args.out_dir / "rpeak_annotations_template.csv", index=False)
    protocol = {"version": "final_ecg_reference_adjudication_v1", "clips": len(clips), "excluded_participants": sorted(EXCLUDED_PARTICIPANTS), "candidate_A_B_identity": "randomized_per_clip_and_hidden_from_HTML", "candidate_methods": [V3, PRIMARY], "rescue_policy_to_evaluate": "Emrich+ProMAC primary with v3-only long-gap rescue", "fiducial_rule": "one polarity-aware QRS-timing fiducial per physiologic QRS; do not double-label R and S"}
    (args.out_dir / "review_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text("# 最终 ECG reference 裁决包\n\n36 个 5 秒 raw-ECG 片段。A/B 为每页随机交换身份的两个候选；HTML 不显示方法名称。每个生理 QRS 只标一个一致的 QRS-timing fiducial。不可判读时选择该状态，导出的 CSV 会保留排除原因。完成后将 CSV 用作最终策略与受限 v3 rescue 的冻结验证参考。\n", encoding="utf-8")
    _write_app(args.out_dir, clips)
    print(f"Wrote {len(clips)} anonymous adjudication clips to {args.out_dir}")


if __name__ == "__main__":
    main()
