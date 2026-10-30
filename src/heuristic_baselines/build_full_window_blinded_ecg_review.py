"""Build a full-window, detector-blinded raw-ECG R-apex review packet.

This is the first required artifact for rebuilding ECG labels: it creates a
small, complete five-minute manual validation set.  The browser app displays
only 20-second raw-ECG chunks and intentionally includes no detector output,
RR intervals, or candidate marks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_MS = 300_000.0
CHUNK_MS = 20_000.0
# These participants are intentionally valid members of the released rawslots
# training cohort.  In particular, do not include P2 in this validation set.
DEFAULT_PARTICIPANTS = ("P1", "P13", "P15")
EXCLUDED_PARTICIPANTS = {"P2", "P5", "P14", "P16", "P17"}


def _candidate_starts(times_ms: np.ndarray) -> np.ndarray:
    """Return 30-s-spaced, complete 5-min starts from gap-free ECG runs."""
    times = np.asarray(times_ms, dtype=float)
    breaks = np.flatnonzero(np.diff(times) > 250.0) + 1
    bounds = np.r_[0, breaks, len(times)]
    starts: list[float] = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        latest = float(times[hi - 1]) - WINDOW_MS
        if latest < float(times[lo]):
            continue
        starts.extend(np.arange(times[lo], latest + 1.0, 30_000.0).tolist())
    return np.asarray(starts, dtype=float)


def _temporal_stratified_starts(times_ms: np.ndarray, count: int) -> np.ndarray:
    """Choose windows evenly over a recording when activity labels are absent."""
    candidates = _candidate_starts(times_ms)
    if candidates.size < count:
        raise ValueError(f"only {candidates.size} complete 5-min windows are available")
    return candidates[np.linspace(0, candidates.size - 1, count).round().astype(int)]


def _write_readme(out_dir: Path, *, windows: int, chunks: int, chunk_seconds: float) -> None:
    (out_dir / "README.md").write_text(
        f"""# 完整五分钟、盲法 ECG R-apex 审阅包

本包用于建立重建 ECG ground truth 所需的人工验证集。它包含 {windows} 个完整
五分钟窗口、共 {chunks} 个连续的 {chunk_seconds:g} 秒 raw-ECG 页面。页面不显示任何 ECG
detector、candidate peak、cleaned ECG、RR interval 或 RMSSD。

## 审阅规则

1. 两位审阅者必须独立完成所有页面。每人使用不同的 `Reviewer ID`；应用会按
   Reviewer ID 将浏览器内未导出的标注分开保存。
2. 只点击可辨认的 raw-ECG R apex。若导联倒置，以主导 QRS 极值作为一致的
   时间 fiducial；无法分辨的 beat 不要猜测，可在导出的 CSV 的 `notes` 中说明。
3. 点击会捕捉到最近的原始 ECG sample。`Undo` 只撤销当前审阅者最近一次操作；
   `Clear chunk` 只清空当前页面。
4. 整页被瞬态伪迹或接触不良遮蔽时，点击“标记本页不可判读”。该状态会导出为
   一行空时间戳记录，后续验证会自动排除包含该页面的五分钟窗口，不能把它当成
   “这一页没有 R apex”。
5. 页面默认使用稳健纵轴范围，以免巨大瞬态伪迹压扁其余 QRS；超出范围的波形会
   截在边缘。需要检查完整振幅时，可点击“显示全幅”。此切换只影响显示，不改变
   raw sample 或导出的时间戳。
6. 完成后下载 CSV，并将两位审阅者的 CSV 交给
   `analyze_full_window_blinded_ecg_review.py`。在人工共识生成前，不可用
   Pan–Tompkins 或其他 detector 标记影响人工审阅。

## 选窗限制

本机目前有 `P1`、`P13`、`P15` 的有效训练队列 raw ECG。本包为每位
参与者从记录早期和晚期各选一个完整窗口，以提供时间覆盖。原始文件没有可用的
活动日志，因此本包**不是按活动类别分层的抽样**；后续获得活动元数据或更多
参与者 raw ECG 后，应扩展验证集。

## 文件

- `annotation_app.html`：离线点击式标注页面。
- `full_window_review_manifest.csv`：窗口、raw ECG 来源和页面范围。
- `rpeak_annotations_template.csv`：CSV 格式模板。
- `review_protocol.json`：可机器读取的固定审阅协议。
""",
        encoding="utf-8",
    )


def _write_app(out_dir: Path, clips: list[dict[str, object]]) -> None:
    """Write a no-network, raw-only browser annotation page."""
    payload = json.dumps(clips, ensure_ascii=False, separators=(",", ":"))
    page = f"""<!doctype html>
<meta charset="utf-8"><title>完整五分钟盲法 ECG R-apex 标注</title>
<style>
body{{font:15px system-ui,sans-serif;margin:20px;color:#17202a;max-width:1180px}}
button,input{{font:inherit;margin:4px}} canvas{{display:block;width:100%;border:1px solid #85929e;cursor:crosshair}}
#meta{{margin:12px 0}} #count{{font-weight:600}} .hint{{color:#566573;max-width:1000px}}
</style>
<h1>完整五分钟盲法 ECG R-apex 标注</h1>
<p class="hint">仅显示 raw ECG。点击每一个可辨认 R apex；点击会对齐至最近 raw sample。页面没有 detector、候选 peak、RR 或 RMSSD。</p>
<label>Reviewer ID <input id="reviewer" autocomplete="off"></label><button id="load">载入/切换审阅者</button>
<button id="prev">← 上一页</button><button id="next">下一页 →</button><button id="undo">撤销</button><button id="clear">清空本页</button><button id="flag">标记本页不可判读</button><button id="scale">显示全幅</button><button id="download">下载 CSV</button>
<div id="meta"></div><canvas id="plot" width="1100" height="480" aria-label="Raw ECG annotation plot"></canvas><p id="count"></p>
<script>
const CLIPS={payload}, cv=document.querySelector('#plot'),ctx=cv.getContext('2d'),pad={{l:64,r:20,t:26,b:54}},prefix='full-window-blinded-r-apex-v2:';
let idx=0,reviewer='',ann={{}},unresolvable={{}},history=[],robustScale=true;
const key=()=>CLIPS[idx].id;
const stateKey=()=>prefix+reviewer;
const nearest=(a,x)=>{{let lo=0,hi=a.length-1;while(lo<hi){{let m=(lo+hi)>>1;if(a[m]<x)lo=m+1;else hi=m}}return lo&&Math.abs(a[lo-1]-x)<Math.abs(a[lo]-x)?lo-1:lo}};
function save(){{if(reviewer){{localStorage.setItem(stateKey(),JSON.stringify(ann));localStorage.setItem(stateKey()+':unresolvable',JSON.stringify(unresolvable))}}}}
function requireReviewer(){{if(reviewer)return true;alert('请先输入并载入 Reviewer ID');return false}}
function draw(){{const d=CLIPS[idx],t=d.timeRelMs,x=d.ecgUv,W=cv.width,H=cv.height,w=W-pad.l-pad.r,h=H-pad.t-pad.b;let mn=Math.min(...x),mx=Math.max(...x);if(robustScale){{let ordered=[...x].sort((a,b)=>a-b),med=ordered[Math.floor(ordered.length/2)],dev=x.map(v=>Math.abs(v-med)).sort((a,b)=>a-b),mad=dev[Math.floor(dev.length/2)]*1.4826;if(mad>0){{mn=med-20*mad;mx=med+20*mad}}}}let p=(mx-mn)*.08||1;mn-=p;mx+=p;
document.querySelector('#meta').textContent=`${{d.reviewId}} · ${{d.participant}} · 窗口 ${{d.windowNumber}}/${{d.windowsForParticipant}} · 第 ${{d.chunkNumber}}/${{d.chunksPerWindow}} 页（${{d.startSec.toFixed(0)}}–${{d.endSec.toFixed(0)}} s） · 总页 ${{idx+1}}/${{CLIPS.length}} · ${{robustScale?'稳健纵轴（伪迹截断）':'全幅纵轴'}}${{unresolvable[key()]?' · 已标记不可判读':''}}`;
ctx.clearRect(0,0,W,H);ctx.strokeStyle='#d5d8dc';ctx.lineWidth=1;ctx.font='12px system-ui';ctx.fillStyle='#17202a';
for(let i=0;i<=4;i++){{let y=pad.t+h*i/4,v=mx-(mx-mn)*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.fillText(v.toFixed(0),6,y+4)}}
for(let i=0;i<=4;i++){{let X=pad.l+w*i/4,v=(t[0]+(t[t.length-1]-t[0])*i/4)/1000;ctx.beginPath();ctx.moveTo(X,pad.t);ctx.lineTo(X,H-pad.b);ctx.stroke();ctx.fillText(v.toFixed(2)+' s',X-15,H-27)}}
const X=j=>pad.l+(t[j]-t[0])/(t[t.length-1]-t[0])*w,Y=j=>pad.t+(mx-x[j])/(mx-mn)*h;ctx.strokeStyle='#273746';ctx.lineWidth=1.2;ctx.beginPath();x.forEach((_,j)=>j?ctx.lineTo(X(j),Y(j)):ctx.moveTo(X(j),Y(j)));ctx.stroke();
(ann[key()]||[]).forEach(j=>{{ctx.fillStyle='#c0392b';ctx.beginPath();ctx.arc(X(j),Y(j),4,0,2*Math.PI);ctx.fill()}});document.querySelector('#count').textContent=`${{(ann[key()]||[]).length}} 个本页 R-apex 标记${{unresolvable[key()]?' · 本页不可判读':''}}${{reviewer?' · 审阅者：'+reviewer:' · 请先载入 Reviewer ID'}}`;document.querySelector('#flag').textContent=unresolvable[key()]?'取消不可判读标记':'标记本页不可判读'}}
document.querySelector('#load').onclick=()=>{{let id=document.querySelector('#reviewer').value.trim();if(!id){{alert('请输入 Reviewer ID');return}}reviewer=id;ann=JSON.parse(localStorage.getItem(stateKey())||'{{}}');unresolvable=JSON.parse(localStorage.getItem(stateKey()+':unresolvable')||'{{}}');history=[];draw()}};
cv.addEventListener('click',e=>{{if(!requireReviewer())return;const r=cv.getBoundingClientRect(),d=CLIPS[idx],w=cv.width-pad.l-pad.r;let u=(e.clientX-r.left)/r.width*cv.width;u=Math.max(pad.l,Math.min(cv.width-pad.r,u));let rel=d.timeRelMs[0]+(u-pad.l)/w*(d.timeRelMs[d.timeRelMs.length-1]-d.timeRelMs[0]),j=nearest(d.timeRelMs,rel);history.push({{type:'add',clip:key(),value:j}});ann[key()]=[...(ann[key()]||[]),j].sort((a,b)=>a-b);save();draw()}});
document.querySelector('#prev').onclick=()=>{{idx=(idx+CLIPS.length-1)%CLIPS.length;draw()}};document.querySelector('#next').onclick=()=>{{idx=(idx+1)%CLIPS.length;draw()}};
document.querySelector('#undo').onclick=()=>{{if(!requireReviewer())return;let h=history.pop();if(!h)return;let a=ann[h.clip]||[],i=a.indexOf(h.value);if(i>=0)a.splice(i,1);ann[h.clip]=a;save();draw()}};
document.querySelector('#clear').onclick=()=>{{if(!requireReviewer())return;history.push({{type:'replace',clip:key(),before:[...(ann[key()]||[])]}});ann[key()]=[];save();draw()}};
document.querySelector('#flag').onclick=()=>{{if(!requireReviewer())return;unresolvable[key()]=!unresolvable[key()];save();draw()}};
document.querySelector('#scale').onclick=()=>{{robustScale=!robustScale;document.querySelector('#scale').textContent=robustScale?'显示全幅':'使用稳健纵轴';draw()}};
document.querySelector('#download').onclick=()=>{{if(!requireReviewer())return;let lines=['reviewer_id,review_id,participant,window_start_unix_ms,manual_rpeak_abs_unix_ms,confidence,notes'];for(const d of CLIPS){{if(unresolvable[d.id])lines.push([reviewer,d.reviewId,d.participant,d.windowStartUnixMs,'','','unresolvable_raw_ecg_chunk'].join(','));for(const j of (ann[d.id]||[]))lines.push([reviewer,d.reviewId,d.participant,d.windowStartUnixMs,(d.windowStartUnixMs+d.timeRelMs[j]).toFixed(3),'',''].join(','));}}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')+'\\n'],{{type:'text/csv'}}));a.download='full_window_rpeak_annotations_'+reviewer+'.csv';a.click();URL.revokeObjectURL(a.href)}};draw();
</script>"""
    (out_dir / "annotation_app.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("Multisite-PPG/raw_data"))
    parser.add_argument("--participants", nargs="+", default=list(DEFAULT_PARTICIPANTS))
    parser.add_argument("--windows-per-participant", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/full_window_blinded_ecg_review_v2_training_cohort"),
    )
    args = parser.parse_args()
    if args.windows_per_participant < 1:
        raise ValueError("--windows-per-participant must be positive")
    if args.chunk_seconds <= 0 or WINDOW_MS % (args.chunk_seconds * 1000.0) != 0:
        raise ValueError("--chunk-seconds must divide 300 seconds exactly")
    participants = tuple(args.participants)
    invalid = set(participants) & EXCLUDED_PARTICIPANTS
    if invalid:
        raise ValueError(f"excluded participants cannot enter the training-cohort review: {sorted(invalid)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunk_ms = args.chunk_seconds * 1000.0
    chunks_per_window = int(round(WINDOW_MS / chunk_ms))
    manifest: list[dict[str, object]] = []
    clips: list[dict[str, object]] = []
    for participant in participants:
        raw_path = args.raw_root / participant / f"{participant}_polar_ecg_raw.npz"
        with np.load(raw_path, allow_pickle=True) as raw_file:
            signal = np.asarray(raw_file["ecg_uv"], dtype=float)
            times = np.asarray(raw_file["time_ms"], dtype=float)
        starts = _temporal_stratified_starts(times, args.windows_per_participant)
        for rank, start in enumerate(starts, start=1):
            review_id = f"{participant}_FW{rank:02d}"
            manifest.append({
                "review_id": review_id,
                "participant": participant,
                "window_start_unix_ms": round(float(start), 3),
                "window_end_unix_ms": round(float(start + WINDOW_MS), 3),
                "window_seconds": 300.0,
                "chunk_seconds": args.chunk_seconds,
                "chunks_per_window": chunks_per_window,
                "selection": "temporally_stratified_without_activity_metadata",
                "raw_ecg_path": str(raw_path.resolve()),
            })
            lo = int(np.searchsorted(times, start, side="left"))
            hi = int(np.searchsorted(times, start + WINDOW_MS, side="right"))
            local_t = times[lo:hi]
            local_x = signal[lo:hi]
            for chunk_no in range(chunks_per_window):
                left = chunk_no * chunk_ms
                right = (chunk_no + 1) * chunk_ms
                keep = (local_t >= start + left) & (local_t <= start + right)
                clip_t = local_t[keep]
                clip_x = local_x[keep]
                if clip_t.size < 10:
                    raise ValueError(f"{review_id}, chunk {chunk_no + 1}: insufficient raw ECG samples")
                clips.append({
                    "id": f"{review_id}_C{chunk_no + 1:02d}",
                    "reviewId": review_id,
                    "participant": participant,
                    "windowStartUnixMs": round(float(start), 3),
                    "windowNumber": rank,
                    "windowsForParticipant": args.windows_per_participant,
                    "chunkNumber": chunk_no + 1,
                    "chunksPerWindow": chunks_per_window,
                    "startSec": left / 1000.0,
                    "endSec": right / 1000.0,
                    "timeRelMs": (clip_t - start).round(3).tolist(),
                    "ecgUv": clip_x.round(2).tolist(),
                })

    pd.DataFrame(manifest).to_csv(args.out_dir / "full_window_review_manifest.csv", index=False)
    pd.DataFrame(columns=[
        "reviewer_id", "review_id", "participant", "window_start_unix_ms",
        "manual_rpeak_abs_unix_ms", "confidence", "notes",
    ]).to_csv(args.out_dir / "rpeak_annotations_template.csv", index=False)
    protocol = {
        "version": "full_window_blinded_ecg_review_v2",
        "scope": "manual raw-ECG R-apex validation only; no detector markers shown",
        "participants": list(participants),
        "excluded_participants": sorted(EXCLUDED_PARTICIPANTS),
        "windows": len(manifest),
        "window_seconds": 300.0,
        "chunk_seconds": args.chunk_seconds,
        "chunks": len(clips),
        "window_selection": "temporally stratified because no activity log is available locally",
    }
    (args.out_dir / "review_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(args.out_dir, windows=len(manifest), chunks=len(clips), chunk_seconds=args.chunk_seconds)
    _write_app(args.out_dir, clips)
    print(f"Wrote {len(manifest)} full-window review windows and {len(clips)} blinded chunks to {args.out_dir}")


if __name__ == "__main__":
    main()
