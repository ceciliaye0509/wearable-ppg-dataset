/* Portable, dependency-free package validation and exchange. No network APIs. */
(function (root) {
  'use strict';
  const enc = new TextEncoder(), dec = new TextDecoder('utf-8', {fatal:true});
  const MAX_META = 25 * 1024 * 1024, MAX_JSON = 10 * 1024 * 1024;
  const fail = message => { throw new Error(message); };
  const obj = x => x && typeof x === 'object' && !Array.isArray(x);
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const id = x => typeof x === 'string' && !['constructor','prototype','__proto__'].includes(x) && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$/.test(x);
  function keys(value, allowed, required=allowed) {
    if (!obj(value) || Object.keys(value).some(k => !allowed.includes(k)) || required.some(k => !(k in value))) fail('Package fields are unsupported or required fields are missing.');
  }
  function text(x, limit=500) { if(typeof x !== 'string' || x.length > limit || /[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(x)) fail('Invalid text field.'); }
  const safePath = x => typeof x === 'string' && /^(?:[A-Za-z0-9_-]+\/)*waveforms\/[A-Za-z0-9_.-]+\.json(?:\.gz)?$/.test(x) && !x.includes('..');
  const u32 = (v, p, n) => v.setUint32(p,n,true), u16 = (v,p,n) => v.setUint16(p,n,true);
  const crcTable = Array.from({length:256}, (_,n) => { for(let i=0;i<8;i++) n = n&1 ? 0xedb88320^(n>>>1) : n>>>1; return n>>>0; });
  function crc32(a) { let c=0xffffffff; for(const b of a)c=crcTable[(c^b)&255]^(c>>>8); return (c^0xffffffff)>>>0; }
  function zip(entries) {
    const pieces=[], central=[]; let offset=0;
    for(const [name,value] of Object.entries(entries)) {
      const nameBytes=enc.encode(name), data=typeof value==='string'?enc.encode(value):new Uint8Array(value), crc=crc32(data);
      const local=new Uint8Array(30+nameBytes.length), v=new DataView(local.buffer);
      u32(v,0,0x04034b50);u16(v,4,20);u16(v,6,0x800);u32(v,14,crc);u32(v,18,data.length);u32(v,22,data.length);u16(v,26,nameBytes.length);local.set(nameBytes,30);
      const cd=new Uint8Array(46+nameBytes.length), cv=new DataView(cd.buffer);
      u32(cv,0,0x02014b50);u16(cv,4,20);u16(cv,6,20);u16(cv,8,0x800);u32(cv,16,crc);u32(cv,20,data.length);u32(cv,24,data.length);u16(cv,28,nameBytes.length);u32(cv,42,offset);cd.set(nameBytes,46);
      pieces.push(local,data);central.push(cd);offset+=local.length+data.length;
    }
    const end=new Uint8Array(22), ev=new DataView(end.buffer), length=central.reduce((a,x)=>a+x.length,0);
    u32(ev,0,0x06054b50);u16(ev,8,central.length);u16(ev,10,central.length);u32(ev,12,length);u32(ev,16,offset);
    const parts=[...pieces,...central,end], result=new Uint8Array(offset+length+22); let i=0;for(const p of parts){result.set(p,i);i+=p.length;}return result;
  }
  function readBook(bytes) {
    const a=new Uint8Array(bytes); if(a.length>MAX_META || a.length<22)fail('Invalid .ecgreview size; maximum 25 MB.');
    const v=new DataView(a.buffer,a.byteOffset,a.byteLength), end=a.length-22;
    if(v.getUint32(end,true)!==0x06054b50 || v.getUint16(end+20,true)!==0 || v.getUint16(end+4,true)!==0 || v.getUint16(end+6,true)!==0 || v.getUint16(end+8,true)!==2 || v.getUint16(end+10,true)!==2)fail('A standard two-file .ecgreview package is required.');
    let p=v.getUint32(end+16,true);const size=v.getUint32(end+12,true), out={}, offsets=[];
    if(p+size!==end)fail('Invalid package directory.');
    for(let i=0;i<2;i++){
      if(p+46>end || v.getUint32(p,true)!==0x02014b50)fail('Package directory is damaged.');
      const flags=v.getUint16(p+8,true),method=v.getUint16(p+10,true),crc=v.getUint32(p+16,true),packed=v.getUint32(p+20,true),length=v.getUint32(p+24,true),nl=v.getUint16(p+28,true),el=v.getUint16(p+30,true),cl=v.getUint16(p+32,true),off=v.getUint32(p+42,true);
      if(p+46+nl+el+cl>end || length>MAX_JSON || packed!==length || method!==0 || (flags&~0x800)!==0)fail('Use the builder’s uncompressed ZIP format; each JSON must be at most 10 MB.');
      if(v.getUint16(p+34,true)!==0)fail('Multi-disk ZIP is not supported.');
      const name=dec.decode(a.subarray(p+46,p+46+nl));
      if(!['manifest.json','items.json'].includes(name) || name in out || off+30>v.getUint32(end+16,true) || v.getUint32(off,true)!==0x04034b50)fail('Invalid package file name or offset.');
      const lnl=v.getUint16(off+26,true),lel=v.getUint16(off+28,true),start=off+30+lnl+lel,stop=start+length;
      if(dec.decode(a.subarray(off+30,off+30+lnl))!==name || v.getUint16(off+6,true)!==flags || v.getUint16(off+8,true)!==0 || v.getUint32(off+14,true)!==crc || v.getUint32(off+18,true)!==length || v.getUint32(off+22,true)!==length || stop>v.getUint32(end+16,true))fail('Inconsistent internal package files.');
      if(offsets.some(([x,y])=>off<y&&stop>x))fail('Internal package files overlap.');offsets.push([off,stop]);
      const data=a.subarray(start,stop);if(crc32(data)!==crc)fail('Package checksum failed.');
      out[name]=JSON.parse(dec.decode(data));p+=46+nl+el+cl;
    }
    if(p!==end)fail('Unexpected package directory content.');
    validatePackage(out['manifest.json'],out['items.json']);return {manifest:out['manifest.json'],items:out['items.json']};
  }
  function validatePackage(m,items) {
    keys(m,['format','version','packageId','title','createdAt','reviewMode','instructions']);
    if(m.format!=='ecg-review' || m.version!==1 || !id(m.packageId) || !['blind','assisted'].includes(m.reviewMode))fail('Invalid package version or review mode.');
    text(m.title,150);text(m.createdAt,40);text(m.instructions,2000);
    if(!Array.isArray(items)||!items.length||items.length>2000)fail('A package must contain 1–2000 tasks.');
    const ids=new Set();let samples=0;
    for(const t of items){
      keys(t,['id','participant','title','mode','file','sha256','fileBytes','sampleCount','durationSec','targets','contextPeaks','initialPeaks','initialPeakSource','instructions','contextPeakSources','durationGroupSec','samplingGroup'],['id','participant','title','mode','file','sha256','fileBytes','sampleCount','durationSec','targets','contextPeaks','initialPeaks','initialPeakSource','instructions']);
      if(!id(t.id)||ids.has(t.id)||!id(t.participant))fail('Duplicate or invalid task ID.');ids.add(t.id);text(t.title,150);text(t.instructions,2000);
      if(t.durationGroupSec!==undefined&&(!finite(t.durationGroupSec)||t.durationGroupSec<=0||t.durationGroupSec>3600))fail('Invalid duration group.');
      if(t.samplingGroup!==undefined&&!['Random','Difficult','Unclassified'].includes(t.samplingGroup))fail('Invalid sampling group.');
      if(t.mode!=='rpeaks')fail('This tool supports R-peak review packages only.');
      if(!safePath(t.file)||!(/^[a-f0-9]{64}$/.test(t.sha256))||!Number.isInteger(t.fileBytes)||t.fileBytes<1||t.fileBytes>MAX_JSON)fail('Invalid waveform reference.');
      if(!Number.isInteger(t.sampleCount)||t.sampleCount<2||t.sampleCount>200000||!finite(t.durationSec)||t.durationSec<=0||t.durationSec>3600)fail('Invalid waveform length.');samples+=t.sampleCount;
      if(samples>10000000)fail('Package too large. Split it into smaller packages.');
      const validIndex=i=>Number.isInteger(i)&&i>=0&&i<t.sampleCount;
      for(const list of [t.contextPeaks,t.initialPeaks])if(!Array.isArray(list)||list.some((x,i)=>!validIndex(x)||(i>0&&x<=list[i-1])))fail('Peaks must be sorted, unique native sample indices.');
      if(!['none','algorithm','prior_human','mixed'].includes(t.initialPeakSource))fail('Invalid initial peak source.');
      if(t.contextPeakSources!==undefined&&(!obj(t.contextPeakSources)||Object.keys(t.contextPeakSources).length))fail('R-peak tasks cannot contain separate context peak sources.');
      if(!Array.isArray(t.targets)||t.targets.length||t.contextPeaks.length)fail('R-peak tasks may contain editable R positions only.');
      if(m.reviewMode==='blind'&&(t.initialPeaks.length||t.initialPeakSource!=='none'))fail('Blind packages cannot contain candidate peaks or reference sources.');
    }return true;
  }
  async function sha256(bytes){if(!root.crypto?.subtle)fail('File verification is unavailable. Use HTTPS in Chrome/Edge or the offline tool.');return Array.from(new Uint8Array(await root.crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');}
  async function boundedBytes(stream,limit){const reader=stream.getReader();let count=0;const parts=[];try{while(true){const {done,value}=await reader.read();if(done)break;count+=value.length;if(count>limit)fail('Decompressed file exceeds the size limit.');parts.push(value);}}catch(e){await reader.cancel().catch(()=>{});throw e;}const a=new Uint8Array(count);let i=0;for(const p of parts){a.set(p,i);i+=p.length;}return a;}
  async function loadWave(file,item){
    if(file.size!==item.fileBytes)fail('Waveform size does not match the package.');
    const raw=new Uint8Array(await file.arrayBuffer());if(await sha256(raw)!==item.sha256)fail('Waveform SHA-256 failed. Obtain the package again.');
    const bytes=item.file.endsWith('.gz')?await boundedBytes(new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip')),MAX_JSON):raw;
    const w=JSON.parse(dec.decode(bytes));keys(w,['t','x']);
    if(!Array.isArray(w.t)||!Array.isArray(w.x)||w.t.length!==item.sampleCount||w.x.length!==w.t.length||w.t.some((t,i)=>!finite(t)||(i>0&&t<=w.t[i-1]))||w.x.some(x=>!finite(x))||Math.abs(w.t[0])>0.00001||Math.abs(w.t.at(-1)-item.durationSec)>0.0001)fail('ECG must contain finite values with strictly increasing native sample times.');
    return w;
  }
  // Empty labels is reserved solely for compatibility with existing v1 R-peak backups.
  function blank(item){return {peaks:[...item.initialPeaks],labels:{},unreadable:[],pages:[],notes:'',status:'in_progress',version:0,updatedAt:null};}
  function normalizeRanges(ranges){const out=[];for(const [a,b]of ranges.map(([a,b])=>[Math.min(a,b),Math.max(a,b)]).sort((a,b)=>a[0]-b[0])){if(a===b)continue;if(out.length&&a<=out.at(-1)[1])out.at(-1)[1]=Math.max(b,out.at(-1)[1]);else out.push([a,b]);}return out;}
  function validateRecord(r,t,complete=r.status==='complete'){
    keys(r,['peaks','labels','unreadable','pages','notes','status','version','updatedAt']);
    const idx=x=>Number.isInteger(x)&&x>=0&&x<t.sampleCount;
    if(!['in_progress','complete'].includes(r.status)||!Number.isInteger(r.version)||r.version<0)fail('Invalid saved revision.');text(r.notes,4000);if(r.updatedAt!==null)text(r.updatedAt,40);
    if(t.mode!=='rpeaks'||!Array.isArray(r.peaks)||r.peaks.some((x,i)=>!idx(x)||(i&&x<=r.peaks[i-1])))fail('Invalid R-peak annotations.');
    if(!Array.isArray(r.unreadable)||r.unreadable.some(a=>!Array.isArray(a)||a.length!==2||!idx(a[0])||!idx(a[1])||a[0]>=a[1]))fail('Invalid unreadable intervals.');
    const count=Math.ceil(t.durationSec/5);if(!Array.isArray(r.pages)||new Set(r.pages).size!==r.pages.length||r.pages.some(x=>!Number.isInteger(x)||x<0||x>=count))fail('Invalid page confirmations.');
    if(!obj(r.labels)||Object.keys(r.labels).length)fail('R-peak backup contains unsupported annotation fields.');
    if(complete){
      if(r.pages.length!==count)fail('Confirm every five-second page before completing the window.');
      if(!r.peaks.length&&!r.unreadable.some(([a,b])=>a===0&&b===t.sampleCount-1))fail('No R peaks. Check the waveform or mark the entire segment unreadable.');
    }return true;
  }
  function validateBackup(b,book,digest){
    keys(b,['format','version','packageHash','packageId','reviewer','exportedAt','records','history']);
    if(b.format!=='ecg-progress'||b.version!==1||b.packageHash!==digest||b.packageId!==book.manifest.packageId||!id(b.reviewer)||!Array.isArray(b.history)||b.history.length>100000)fail('Backup format or package identity does not match.');text(b.exportedAt,40);if(!obj(b.records))fail('Invalid backup records.');
    for(const [key,r]of Object.entries(b.records)){const t=book.items.find(x=>x.id===key);if(!t)fail('Unknown task in backup.');validateRecord(r,t);}
    const versions=new Map();for(const row of b.history){keys(row,['taskId','record']);const t=book.items.find(x=>x.id===row.taskId);if(!t||!b.records[row.taskId])fail('Unknown task in revision history.');validateRecord(row.record,t);if(row.record.version<1||row.record.version<=(versions.get(t.id)||0))fail('Invalid revision history order.');versions.set(t.id,row.record.version);}
    for(const [taskId,r]of Object.entries(b.records)){const last=b.history.filter(x=>x.taskId===taskId).at(-1);if(!last||JSON.stringify(last.record)!==JSON.stringify(r))fail('Current records do not match revision history.');}return true;
  }
  function csv(rows){return '\ufeff'+rows.map(r=>r.map(v=>{let s=v==null?'':String(v);if(/^[\s]*[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';}).join(',')).join('\r\n')+'\r\n';}
  function exportsFor(book,b,waves){
    const h=['package_id','package_sha256','reviewer','task_id','participant','status','version','updated_at'];
    const events=[h.concat(['sample_index','time_sec','unreadable','notes'])],ranges=[h.concat(['start_index','end_index','start_sec','end_sec'])],tasks=[h.concat(['mode','pages_confirmed','pages_required','peak_count','notes'])];
    for(const t of book.items){const r=b.records[t.id],w=waves.get(t.id),base=[book.manifest.packageId,b.packageHash,b.reviewer,t.id,t.participant,r?.status||'not_started',r?.version||0,r?.updatedAt||''];
      tasks.push([...base,t.mode,r?.pages.length||0,Math.ceil(t.durationSec/5),r?.peaks.length||0,r?.notes||'']);
      if(!r)continue;
      for(const i of r.peaks)events.push([...base,i,w.t[i],r.unreadable.some(([a,b])=>a<=i&&i<=b),r.notes]);
      for(const [a,z]of r.unreadable)ranges.push([...base,a,z,w.t[a],w.t[z]]);
    }
    return {'event_reviews.csv':csv(events),'unreadable_intervals.csv':csv(ranges),'task_reviews.csv':csv(tasks),'revision_history.json':JSON.stringify(b,null,2)};
  }
  function togglePeak(peaks,times,index,seconds,tolerance=.055){
    let closest=-1,distance=Infinity;peaks.forEach((p,i)=>{const d=Math.abs(times[p]-seconds);if(d<distance){distance=d;closest=i;}});
    const next=[...peaks];let changedIndex=index;
    if(closest>=0&&distance<=tolerance){changedIndex=next[closest];next.splice(closest,1);}else if(!next.includes(index)){next.push(index);next.sort((a,b)=>a-b);}
    return {peaks:next,changedIndex};
  }
  function durationGroup(t){if(t.durationGroupSec!==undefined)return t.durationGroupSec;const core=String(t.instructions||'').match(/(?:核心|core\s*)(\d+(?:\.\d+)?)\s*(?:秒|seconds?|s\b)/i);return core&&Number(core[1])>0&&Number(core[1])<=t.durationSec?Number(core[1]):Math.round(t.durationSec*10)/10;}
  function samplingGroup(t){return t.samplingGroup||'Unclassified';}
  function matchesGroup(t,duration,sampling,query=''){return (!duration||String(durationGroup(t))===duration)&&(!sampling||samplingGroup(t)===sampling)&&(!query||`${t.participant} ${t.title}`.toLowerCase().includes(query.toLowerCase()));}
  function validateBrowseIndex(index,book,hash){
    keys(index,['format','version','packageHash','items']);
    if(index.format!=='ecg-browse-index'||index.version!==1||index.packageHash!==hash||!Array.isArray(index.items)||index.items.length!==book.items.length)fail('Browse index does not match this task package.');
    const ids=new Set(book.items.map(t=>t.id)),seen=new Set(),paths=new Set();
    for(const row of index.items){keys(row,['id','file','durationGroupSec','samplingGroup']);if(!ids.has(row.id)||seen.has(row.id)||paths.has(row.file)||!safePath(row.file)||!finite(row.durationGroupSec)||row.durationGroupSec<=0||row.durationGroupSec>3600||!['Random','Difficult','Unclassified'].includes(row.samplingGroup))fail('Invalid browse index entry.');seen.add(row.id);paths.add(row.file);}
    return new Map(index.items.map(row=>[row.id,row]));
  }
  root.ECGCore={validateBrowseIndex,durationGroup,samplingGroup,matchesGroup,togglePeak,MAX_META,MAX_JSON,readBook,validatePackage,sha256,loadWave,blank,normalizeRanges,validateRecord,validateBackup,zip,csv,exportsFor,id};
})(typeof window==='undefined'?globalThis:window);
