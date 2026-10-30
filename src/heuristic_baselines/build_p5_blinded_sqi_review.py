"""Build a blinded manual signal-quality review page for P5 ECG clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _write_app(out_dir: Path, clips: list[dict[str, object]]) -> None:
    payload = json.dumps({"clips": clips}, separators=(",", ":"))
    page = f"""<!doctype html>
<meta charset="utf-8"><title>P5 blinded ECG signal-quality review</title>
<style>
body{{font:15px system-ui,sans-serif;margin:20px;color:#17202a;max-width:1180px}}button,input,select,textarea{{font:inherit;margin:4px}}canvas{{display:block;width:100%;border:1px solid #85929e;cursor:crosshair}}textarea{{vertical-align:middle;width:340px;height:2.2em}}#meta{{margin:12px 0;font-weight:600}}#status{{font-weight:600}}.hint{{color:#566573;max-width:1080px}}
</style>
<h1>P5 blinded raw-ECG signal-quality review</h1>
<p class="hint">Each page is one 20-s raw-ECG excerpt. No detector peaks, automatic quality score, or automatic stratum is shown. For each clip choose: <b>assessable</b> (nearly every beat has a repeatable main QRS timing extreme), <b>partial</b> (only some beats are reliable), or <b>unassessable</b> (no reliable beat timing). For assessable/partial clips, click one dominant QRS timing extreme per visible beat; do not click multiple deflections of the same beat. Leave unassessable clips with no clicks. </p>
<label>Reviewer ID <input id="reviewer" autocomplete="off"></label>
<label>Clip judgement <select id="verdict"><option value="unreviewed">unreviewed</option><option value="assessable">assessable</option><option value="partial">partial</option><option value="unassessable">unassessable</option></select></label>
<label>Note <textarea id="note" placeholder="optional"></textarea></label>
<button id="prev">← Previous</button><button id="next">Next →</button><button id="undo">Undo peak</button><button id="clear">Clear peaks</button><button id="download">Download review CSV</button>
<div id="meta"></div><canvas id="plot" width="1100" height="480" aria-label="Raw ECG quality-review plot"></canvas><p id="status"></p>
<script>
const DATA={payload},STORE='p5-blinded-ecg-sqi-review-v1',pad={{l:64,r:20,t:26,b:54}},cv=document.querySelector('#plot'),ctx=cv.getContext('2d');
let idx=0,state=JSON.parse(localStorage.getItem(STORE)||'{{}}');const key=()=>DATA.clips[idx].id;const save=()=>localStorage.setItem(STORE,JSON.stringify(state));
function entry(){{return state[key()]||(state[key()]={{verdict:'unreviewed',note:'',peaks:[]}})}}
function nearest(a,x){{let lo=0,hi=a.length-1;while(lo<hi){{let m=(lo+hi)>>1;if(a[m]<x)lo=m+1;else hi=m}}return lo&&Math.abs(a[lo-1]-x)<Math.abs(a[lo]-x)?lo-1:lo}}
function draw(){{const d=DATA.clips[idx],e=entry(),t=d.t,x=d.x,W=cv.width,H=cv.height,w=W-pad.l-pad.r,h=H-pad.t-pad.b;let mn=Math.min(...x),mx=Math.max(...x),p=(mx-mn)*.08||1;mn-=p;mx+=p;
document.querySelector('#meta').textContent=`${{d.id}} · P5 · 20-s blinded excerpt`;document.querySelector('#verdict').value=e.verdict;document.querySelector('#note').value=e.note;
ctx.clearRect(0,0,W,H);ctx.strokeStyle='#d5d8dc';ctx.lineWidth=1;ctx.font='12px system-ui';ctx.fillStyle='#17202a';
for(let i=0;i<=4;i++){{let y=pad.t+h*i/4,v=mx-(mx-mn)*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.fillText(v.toFixed(0),6,y+4)}}
for(let i=0;i<=4;i++){{let X=pad.l+w*i/4,v=(t[t.length-1]-t[0])*i/4;ctx.beginPath();ctx.moveTo(X,pad.t);ctx.lineTo(X,H-pad.b);ctx.stroke();ctx.fillText((v/1000).toFixed(1)+' s',X-12,H-27)}}
const X=j=>pad.l+(t[j]-t[0])/(t[t.length-1]-t[0])*w,Y=j=>pad.t+(mx-x[j])/(mx-mn)*h;ctx.strokeStyle='#273746';ctx.lineWidth=1.2;ctx.beginPath();x.forEach((_,j)=>j?ctx.lineTo(X(j),Y(j)):ctx.moveTo(X(j),Y(j)));ctx.stroke();e.peaks.forEach(j=>{{ctx.fillStyle='#c0392b';ctx.beginPath();ctx.arc(X(j),Y(j),4,0,2*Math.PI);ctx.fill()}});
const reviewed=Object.values(state).filter(v=>v.verdict!=='unreviewed').length;document.querySelector('#status').textContent=`${{e.peaks.length}} QRS-timing marks · ${{reviewed}} / ${{DATA.clips.length}} clips judged`}}
cv.addEventListener('click',ev=>{{const r=cv.getBoundingClientRect(),d=DATA.clips[idx],e=entry(),w=cv.width-pad.l-pad.r;let u=(ev.clientX-r.left)/r.width*cv.width;u=Math.max(pad.l,Math.min(cv.width-pad.r,u));let ms=d.t[0]+(u-pad.l)/w*(d.t[d.t.length-1]-d.t[0]),j=nearest(d.t,ms),near=e.peaks.findIndex(v=>Math.abs(d.t[v]-d.t[j])<=80);if(near>=0)e.peaks.splice(near,1);else e.peaks.push(j),e.peaks.sort((a,b)=>a-b);save();draw()}});
document.querySelector('#verdict').onchange=ev=>{{entry().verdict=ev.target.value;save();draw()}};document.querySelector('#note').onchange=ev=>{{entry().note=ev.target.value.trim();save()}};
document.querySelector('#prev').onclick=()=>{{idx=(idx+DATA.clips.length-1)%DATA.clips.length;draw()}};document.querySelector('#next').onclick=()=>{{idx=(idx+1)%DATA.clips.length;draw()}};document.querySelector('#undo').onclick=()=>{{entry().peaks.pop();save();draw()}};document.querySelector('#clear').onclick=()=>{{entry().peaks=[];save();draw()}};
document.querySelector('#download').onclick=()=>{{const reviewer=document.querySelector('#reviewer').value.trim();if(!reviewer){{alert('Enter Reviewer ID first');return}}let lines=['reviewer_id,record_type,manual_review_id,participant,clip_start_unix_ms,quality_judgement,manual_qrs_time_unix_ms,notes'];for(const d of DATA.clips){{const e=state[d.id]||{{verdict:'unreviewed',note:'',peaks:[]}};lines.push([reviewer,'clip_qc',d.id,'P5',d.t[0].toFixed(3),e.verdict,'',JSON.stringify(e.note)].join(','));for(const j of e.peaks)lines.push([reviewer,'qrs_fiducial',d.id,'P5',d.t[0].toFixed(3),e.verdict,d.t[j].toFixed(3),JSON.stringify(e.note)].join(','))}}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')+'\\n'],{{type:'text/csv'}}));a.download='p5_blinded_ecg_sqi_review_'+reviewer+'.csv';a.click();URL.revokeObjectURL(a.href)}};draw();
</script>"""
    (out_dir / "annotation_app.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("Multisite-PPG/raw_data/P5/P5_polar_ecg_raw.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/p5_ecg_signal_quality_v1/p5_manual_blinded_sampling_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/p5_blinded_ecg_sqi_review_v1"))
    args = parser.parse_args()
    sample = pd.read_csv(args.manifest)
    with np.load(args.raw, allow_pickle=True) as raw:
        x, t = np.asarray(raw["ecg_uv"], float), np.asarray(raw["time_ms"], float)
    clips: list[dict[str, object]] = []
    for row in sample.itertuples(index=False):
        keep = (t >= row.start_unix_ms) & (t <= row.end_unix_ms)
        clips.append({"id": row.manual_review_id, "t": t[keep].round(3).tolist(), "x": x[keep].round(2).tolist()})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_app(args.out_dir, clips)
    (args.out_dir / "REVIEWER_INSTRUCTIONS.md").write_text(
        "# P5 blinded ECG signal-quality review\n\n"
        "For every 20-s page choose `assessable`, `partial`, or `unassessable`. "
        "Do not infer quality from peak count: the page is intentionally detector-free. "
        "For assessable/partial clips, mark one dominant QRS timing extreme per visible beat. "
        "For unassessable clips leave zero QRS marks. Download one CSV after all 75 clips are judged.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(clips)} blinded P5 review clips to {args.out_dir}")


if __name__ == "__main__":
    main()
