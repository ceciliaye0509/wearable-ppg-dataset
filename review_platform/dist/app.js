/* Application operations never transmit task files, waveform data or answers. */
'use strict';
const $=s=>document.querySelector(s),C=ECGCore,DB=ECGStore;
const S={book:null,hash:null,files:null,waves:new Map(),reviewer:'',backup:null,revision:0,index:0,record:null,page:0,tool:'edit',rangeStart:null,dirty:false,timer:null,undo:[],saving:Promise.resolve(),busy:false,pendingBackup:null};
const clone=x=>JSON.parse(JSON.stringify(x)),item=()=>S.book.items[S.index],wave=()=>S.waves.get(item().id),sessionKey=()=>S.hash+'/'+S.reviewer;
let toastTimer;
function notice(message,error=false){$('#toast').hidden=false;$('#toast').textContent=message;clearTimeout(toastTimer);toastTimer=setTimeout(()=>{$('#toast').hidden=true;},error?14000:5000);}
function saveStatus(message,error=false){$('#save-status').textContent=message;$('#save-status').classList.toggle('error',error);}
function run(fn){return async(...args)=>{try{await fn(...args);}catch(e){notice(e.message||'Operation failed.',true);console.error(e.name+': '+e.message);}};}
function needReviewer(){const r=$('#reviewer').value.trim();if(!C.id(r))throw Error('Enter a valid reviewer name using letters, numbers, periods, underscores or hyphens.');return r;}
async function openFiles(files,handle=null){
  if(S.busy)throw Error('Validation in progress. Please wait.');S.busy=true;$('#import-status').textContent='Checking the package and waveforms locally…';$('#import-status').className='';
  try{
    const reviewer=needReviewer(),books=files.filter(f=>f.name.endsWith('.ecgreview'));if(books.length!==1)throw Error('The folder must contain exactly one .ecgreview file.');
    const bookFile=books[0];if(bookFile.size>C.MAX_META)throw Error('Package metadata exceeds the 25 MB limit.');
    const bytes=await bookFile.arrayBuffer(),book=C.readBook(bytes),hash=await C.sha256(bytes),prefix=bookFile._relative.slice(0,-bookFile.name.length),map=new Map();
    for(const f of files){if(map.has(f._relative))throw Error('Duplicate paths in the selected folder.');map.set(f._relative,f);}
    let browse=new Map();const indexFile=map.get(prefix+'browse_index.json');if(indexFile){if(indexFile.size>2*1024*1024)throw Error('Browse index is too large.');browse=C.validateBrowseIndex(JSON.parse(await indexFile.text()),book,hash);for(const t of book.items){const row=browse.get(t.id);t.durationGroupSec=row.durationGroupSec;t.samplingGroup=row.samplingGroup;}}
    const refs=book.items.map(t=>map.get(prefix+(browse.get(t.id)?.file||t.file)));if(refs.some(x=>!x))throw Error('Missing waveform files. Select the entire extracted package folder.');
    if(refs.reduce((n,f)=>n+f.size,0)>250*1024*1024)throw Error('Waveform files exceed 250 MB. Ask the team to split the package.');
    const waves=new Map();for(let i=0;i<book.items.length;i++){const t=book.items[i];$('#import-status').textContent=`Checking waveform locally ${i+1}/${book.items.length}…`;waves.set(t.id,await C.loadWave(refs[i],t));}
    let stored=await DB.get('sessions',hash+'/'+reviewer);if(stored)C.validateBackup(stored.backup,book,hash);
    if(S.pendingBackup){const b=S.pendingBackup;C.validateBackup(b,book,hash);if(b.reviewer!==reviewer)throw Error('The backup reviewer name does not match.');const existing=stored?.backup.history||[];if(existing.length>b.history.length||existing.some((r,i)=>JSON.stringify(r)!==JSON.stringify(b.history[i])))throw Error('Local progress is newer or has diverged. Export it before restoring a backup.');const revision=await DB.saveSession(hash+'/'+reviewer,b,stored?.revision||0);stored={revision,backup:b};S.pendingBackup=null;}
    Object.assign(S,{book,hash,files:map,waves,reviewer,revision:stored?.revision||0,backup:stored?.backup||{format:'ecg-progress',version:1,packageHash:hash,packageId:book.manifest.packageId,reviewer,exportedAt:new Date().toISOString(),records:{},history:[]},dirty:false});
    if(handle){try{await DB.put('handles','last-folder',handle);}catch{notice('Folder permission could not be remembered. Select the folder again next time.');}}
    try{localStorage.setItem('ecg-reviewer',reviewer);}catch{}
    $('#home').hidden=true;$('#workspace').hidden=false;$('#principles').hidden=true;$('#import-status').textContent='';
    $('#package-title').textContent=book.manifest.title;$('#package-description').textContent=book.manifest.instructions;
    $('#package-mode').textContent=book.manifest.reviewMode==='blind'?'BLIND REVIEW · No candidate peaks':'ASSISTED REVIEW · Editable suggestions';$('#reviewer-badge').textContent='Reviewer: '+reviewer;
    $('#duration-filter').replaceChildren(new Option('All durations',''));for(const n of [...new Set(book.items.map(C.durationGroup))].sort((a,b)=>a-b))$('#duration-filter').append(new Option(n+' seconds',String(n)));$('#sampling-filter').value='';$('#task-filter').value='';
    const next=book.items.findIndex(t=>S.backup.records[t.id]?.status!=='complete');loadTask(next<0?0:next);renderList();
    try{if(navigator.storage?.persist)await navigator.storage.persist();}catch{}
  }catch(e){$('#import-status').textContent=e.message;$('#import-status').className='error';throw e;}finally{S.busy=false;if(S.book)$('#task-note').disabled=false;}
}
async function folderFiles(handle,prefix=''){const out=[];for await(const [name,h]of handle.entries()){if(h.kind==='directory'){if(!name.startsWith('.'))out.push(...await folderFiles(h,prefix+name+'/'));}else if(name.endsWith('.ecgreview')||/\.json(?:\.gz)?$/.test(name)){const f=await h.getFile();f._relative=prefix+name;out.push(f);}}return out;}
$('#choose-folder').onclick=run(async()=>{needReviewer();if(window.showDirectoryPicker){let h;try{h=await showDirectoryPicker({mode:'read'});}catch(e){if(e.name==='AbortError')return;if(e.name==='SecurityError'){ $('#folder-input').click();return;}throw e;}await openFiles(await folderFiles(h),h);}else $('#folder-input').click();});
$('#folder-input').onchange=run(async e=>{const files=[...e.target.files];for(const f of files)f._relative=f.webkitRelativePath;await openFiles(files);e.target.value='';});
$('#reconnect').onclick=run(async()=>{needReviewer();const h=await DB.get('handles','last-folder');if(!h)throw Error('No previous folder permission. Select the folder again.');if(await h.requestPermission({mode:'read'})!=='granted')throw Error('Read permission was not granted. Select the folder manually.');await openFiles(await folderFiles(h),h);});
function loadTask(i){S.index=i;S.record=clone(S.backup.records[item().id]||C.blank(item()));S.page=0;S.tool='edit';S.rangeStart=null;S.undo=[];S.dirty=false;renderTask();}
async function selectTask(i){await flush();if(S.dirty)throw Error('Progress is not saved. Download a backup before resolving the save error.');loadTask(i);renderList();}
function renderList(){
  const done=S.book.items.filter(t=>S.backup.records[t.id]?.status==='complete').length;
  $('#task-count').textContent=`${done}/${S.book.items.length}`;
  $('#progress-detail').textContent=`${done}/${S.book.items.length} segments complete · Your progress`;
  $('#progress-bar').style.width=`${100*done/S.book.items.length}%`;
  const host=$('#task-list');host.replaceChildren();const q=$('#task-filter').value.trim(),duration=$('#duration-filter').value,sampling=$('#sampling-filter').value;
  const all=S.book.items.map((t,i)=>({t,i})),visible=all.filter(({t})=>C.matchesGroup(t,duration,sampling,q));
  $('#filter-count').textContent=`${visible.length} / ${all.length} segments shown`;
  if(!visible.length){const empty=document.createElement('p');empty.className='micro';empty.textContent='No segments match these filters.';host.append(empty);}
  for(const seconds of [...new Set(visible.map(({t})=>C.durationGroup(t)))].sort((a,b)=>a-b)){
    const h=document.createElement('h3');h.className='duration-heading';h.textContent=seconds+' seconds';host.append(h);
    for(const group of ['Random','Difficult','Unclassified']){
      const grouped=visible.filter(({t})=>C.durationGroup(t)===seconds&&C.samplingGroup(t)===group);if(!grouped.length)continue;
      const sh=document.createElement('h4');sh.className='sampling-heading';sh.textContent=group+' · '+grouped.length;host.append(sh);
      for(const participant of new Set(grouped.map(({t})=>t.participant))){
        const tasks=grouped.filter(({t})=>t.participant===participant),ph=document.createElement('div');ph.className='participant-heading';ph.textContent=participant;host.append(ph);
        for(const {t,i}of tasks){const b=document.createElement('button'),r=S.backup.records[t.id];b.className='task-button'+(i===S.index?' active':'')+(r?.status==='complete'?' done':'');
          const title=document.createElement('strong'),small=document.createElement('small');title.textContent=t.title;small.textContent=`${r?.status==='complete'?'Complete':r?'In progress':'Not reviewed'} · ${S.book.manifest.reviewMode==='blind'?'Blind review':'Assisted review'}`;b.append(title,small);b.onclick=run(()=>selectTask(i));host.append(b);}
      }
    }
  }
}
function setTool(tool){S.tool=tool;S.rangeStart=null;document.querySelectorAll('[data-tool]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.tool===tool)));plot();}
function renderTask(){const t=item(),r=S.record;$('#task-mode').textContent='R-PEAKS · Native sample positions';$('#task-title').textContent=t.title;$('#task-instructions').textContent=t.instructions;$('#task-status').textContent=r.status==='complete'?'Complete':'In progress';$('#page-panel').hidden=false;
  $('#task-note').value=r.notes;$('#undo').disabled=!S.undo.length;$('#confirm-page').disabled=S.page<0;$('#task-note').disabled=S.busy;
  renderMasks();renderPages();setTool(S.tool);if(S.dirty)saveStatus('Changes pending…');else saveStatus(r.updatedAt?`Saved · v${r.version} · ${new Date(r.updatedAt).toLocaleTimeString()}`:'Not yet saved');
}
function checkpoint(){S.undo.push(clone(S.record));if(S.undo.length>60)S.undo.shift();}
function changed(){S.record.status='in_progress';S.dirty=true;clearTimeout(S.timer);S.timer=setTimeout(()=>{flush().catch(e=>saveStatus(e.message,true));},650);$('#task-status').textContent='In progress';$('#undo').disabled=!S.undo.length;saveStatus('Saving on this device…');}
function pageInvalidated(index){const p=Math.min(Math.ceil(item().durationSec/5)-1,Math.floor(wave().t[index]/5));S.record.pages=S.record.pages.filter(x=>x!==p);}
function flush(){clearTimeout(S.timer);S.saving=S.saving.catch(()=>{}).then(async()=>{
    if(!S.dirty||!S.book)return;const r=clone(S.record);C.validateRecord(r,item());r.version=(S.backup.records[item().id]?.version||0)+1;r.updatedAt=new Date().toISOString();
    const b=clone(S.backup);b.records[item().id]=r;b.history.push({taskId:item().id,record:clone(r)});b.exportedAt=new Date().toISOString();
    const revision=await DB.saveSession(sessionKey(),b,S.revision);S.revision=revision;S.backup=b;
    // UI may have changed during the asynchronous IndexedDB transaction.
    const current=clone(S.record);current.version=r.version;current.updatedAt=r.updatedAt;const unchanged=JSON.stringify({...current,version:0,updatedAt:null})===JSON.stringify({...r,version:0,updatedAt:null});
    S.record.version=r.version;S.record.updatedAt=r.updatedAt;S.dirty=!unchanged;saveStatus(S.dirty?'Further changes pending…':`Saved on this device · v${r.version} · ${new Date(r.updatedAt).toLocaleTimeString()}`);renderList();if(S.dirty){clearTimeout(S.timer);S.timer=setTimeout(()=>flush().catch(e=>saveStatus(e.message,true)),300);}
  });return S.saving;}
$('#task-note').oninput=()=>{checkpoint();S.record.notes=$('#task-note').value;changed();};
document.querySelectorAll('[data-tool]').forEach(b=>b.onclick=()=>setTool(b.dataset.tool));
$('#undo').onclick=run(()=>{const old=S.undo.pop();if(!old)return;S.record=old;changed();renderTask();});$('#save').onclick=run(async()=>{await flush();if(!S.dirty)notice('Progress saved in this browser.');});
$('#complete').onclick=run(async()=>{C.validateRecord(S.record,item(),true);S.record.status='complete';S.dirty=true;await flush();if(S.dirty)throw Error('Some changes are still pending.');const i=S.book.items.findIndex((t,j)=>j>S.index&&S.backup.records[t.id]?.status!=='complete');if(i>=0){loadTask(i);renderList();}else{renderTask();notice('Window complete. Continue reviewing or download annotations ZIP.');}});
$('#confirm-page').onclick=run(async()=>{if(S.page<0)throw Error('Open a five-second page first.');checkpoint();if(!S.record.pages.includes(S.page))S.record.pages.push(S.page);S.record.pages.sort((a,b)=>a-b);changed();const pages=Math.ceil(item().durationSec/5);S.page=Array.from({length:pages},(_,i)=>i).find(i=>i>S.page&&!S.record.pages.includes(i))??S.page;renderPages();plot();await flush();});
function renderPages(){$('#page-summary').textContent=`${S.record.pages.length}/${Math.ceil(item().durationSec/5)} pages confirmed`;const host=$('#pages');host.replaceChildren();for(let i=0;i<Math.ceil(item().durationSec/5);i++){const b=document.createElement('button');b.textContent=`${i*5}–${Math.min((i+1)*5,item().durationSec).toFixed(0)}`;b.className=(S.record.pages.includes(i)?'confirmed ':'')+(S.page===i?'current':'');b.setAttribute('aria-label',`${i*5} sec: ${S.record.pages.includes(i)?'Confirmed':'Not confirmed'}`);b.onclick=()=>{S.page=i;S.rangeStart=null;renderPages();plot();};host.append(b);}$('#confirm-page').disabled=S.page<0;}
$('#next-unreviewed').onclick=()=>{S.page=Array.from({length:Math.ceil(item().durationSec/5)},(_,i)=>i).find(i=>!S.record.pages.includes(i))??0;S.rangeStart=null;renderPages();plot();};
$('#view-full').onclick=()=>{S.page=-1;S.rangeStart=null;renderPages();plot();};$('#page-prev').onclick=()=>changePage(-1);$('#page-next').onclick=()=>changePage(1);function changePage(n){const max=Math.ceil(item().durationSec/5)-1;S.page=S.page<0?0:Math.max(0,Math.min(max,S.page+n));S.rangeStart=null;renderPages();plot();}
function renderMasks(){const host=$('#masks');host.replaceChildren();if(!S.record.unreadable.length){host.textContent='No unreadable intervals marked.';}S.record.unreadable.forEach(([a,z],i)=>{const b=document.createElement('button');b.textContent=`Unreadable ${wave().t[a].toFixed(3)}–${wave().t[z].toFixed(3)}s ×`;b.onclick=()=>{checkpoint();S.record.unreadable.splice(i,1);changed();renderMasks();plot();};host.append(b);});}
function bounds(){const d=item().durationSec;return S.page<0?[0,d]:[S.page*5,Math.min(d,S.page*5+5)];}
function nearest(t){const a=wave().t;let l=0,h=a.length-1;while(l<h){const m=(l+h)>>1;if(a[m]<t)l=m+1;else h=m;}return l>0&&Math.abs(a[l-1]-t)<Math.abs(a[l]-t)?l-1:l;}
let plotGeom;
function plot(){if(!S.book)return;const canvas=$('#waveform'),rect=canvas.getBoundingClientRect(),ratio=devicePixelRatio||1,w=rect.width,h=rect.height;if(!w||!h)return;canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,h);
  const [a,z]=bounds(),wv=wave(),i0=nearest(a),i1=nearest(z),left=52,right=w-18,top=25,bottom=h-31;let lo=Infinity,hi=-Infinity;for(let i=i0;i<=i1;i++){lo=Math.min(lo,wv.x[i]);hi=Math.max(hi,wv.x[i]);}const pad=Math.max(50,(hi-lo)*.16);lo-=pad;hi+=pad;const X=t=>left+(t-a)/(z-a)*(right-left),Y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);plotGeom={left,right,top,bottom,a,z};
  ctx.font='10px system-ui';ctx.lineWidth=1;for(let i=0;i<=5;i++){const y=top+(bottom-top)*i/5;ctx.strokeStyle='#e4eae3';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillStyle='#7a8d7e';ctx.fillText(Math.round(hi-(hi-lo)*i/5),4,y+3);}for(let i=0;i<=5;i++){const t=a+(z-a)*i/5,x=X(t);ctx.strokeStyle='#e4eae3';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();ctx.fillStyle='#7a8d7e';ctx.fillText(t.toFixed(1)+'s',x-12,h-10);}
  for(const [p,q]of S.record.unreadable){const x=X(Math.max(a,wv.t[p])),end=X(Math.min(z,wv.t[q]));if(end>=x){ctx.fillStyle='#deb35d35';ctx.fillRect(x,top,end-x,bottom-top);}}
  ctx.strokeStyle='#435357';ctx.lineWidth=1;ctx.beginPath();let drawing=false;for(let i=i0;i<=i1;i++){const x=X(wv.t[i]),y=Y(wv.x[i]);if(!drawing||(i>i0&&wv.t[i]-wv.t[i-1]>.2501)){ctx.moveTo(x,y);drawing=true;}else ctx.lineTo(x,y);}ctx.stroke();
  // Every native sample is retained; drawing does not alter the signal or peak indices.
  for(const p of S.record.peaks){if(wv.t[p]<a||wv.t[p]>z)continue;ctx.fillStyle='#c65d5b';ctx.beginPath();ctx.arc(X(wv.t[p]),Y(wv.x[p]),2.7,0,Math.PI*2);ctx.fill();}
  if(S.rangeStart!==null){ctx.strokeStyle='#a67c3d';ctx.beginPath();ctx.moveTo(X(wv.t[S.rangeStart]),top);ctx.lineTo(X(wv.t[S.rangeStart]),bottom);ctx.stroke();}
  $('#page-position').textContent=S.page<0?'Full window':`${a.toFixed(0)}–${z.toFixed(1)}s`;$('#page-prev').disabled=S.page<=0;$('#page-next').disabled=S.page<0||z>=item().durationSec;$('#confirm-page').disabled=S.page<0;
  $('#marker-count').textContent=`${S.record.peaks.filter(p=>wv.t[p]>=a&&wv.t[p]<=z).length} visible / ${S.record.peaks.length} total markers`;
  $('#plot-hint').textContent=S.tool==='mask'?(S.rangeStart===null?'Click the interval start, then its end. Peak marks are kept.':'Click the interval end. Switch tools to cancel.'):S.tool==='edit'?'Click near a red marker to remove it; click an unmarked QRS peak to add it.':'View only. Choose Edit peaks to add or remove markers.';
}
$('#waveform').onclick=run(e=>{if(!plotGeom)return;const rect=e.currentTarget.getBoundingClientRect(),x=e.clientX-rect.left,g=plotGeom;if(x<g.left||x>g.right)return;if(S.page<0&&S.tool!=='inspect')throw Error('Open a five-second page to edit peaks or unreadable intervals.');const t=g.a+(x-g.left)/(g.right-g.left)*(g.z-g.a),idx=nearest(t),r=S.record;
  if(S.tool==='mask'){if(S.rangeStart===null){S.rangeStart=idx;plot();return;}const a=Math.min(idx,S.rangeStart),z=Math.max(idx,S.rangeStart);if(a===z)throw Error('Interval start and end must differ.');checkpoint();r.unreadable=C.normalizeRanges([...r.unreadable,[a,z]]);for(let p=Math.floor(wave().t[a]/5);p<=Math.floor(wave().t[z]/5);p++)r.pages=r.pages.filter(x=>x!==p);S.rangeStart=null;changed();renderMasks();renderPages();plot();return;}
  if(S.tool==='inspect')return;if(S.page<0)throw Error('Open a five-second page to edit peaks or unreadable intervals.');
  if(S.tool==='edit'){
    checkpoint();const result=C.togglePeak(r.peaks,wave().t,idx,t,Math.max(.055,(g.z-g.a)/(g.right-g.left)*7));
    pageInvalidated(result.changedIndex);r.peaks=result.peaks;
  }changed();renderPages();plot();});
new ResizeObserver(()=>plot()).observe($('#waveform'));
function download(name,bytes,type){const blob=new Blob([bytes],{type}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),60000);}
async function backupSnapshot(){try{await flush();}catch(e){notice('Local save failed. Exporting an emergency backup from memory. Keep this file.',true);}const b=clone(S.backup);if(S.dirty){const r=clone(S.record);r.version=(b.records[item().id]?.version||0)+1;r.updatedAt=new Date().toISOString();b.records[item().id]=r;b.history.push({taskId:item().id,record:r});}b.exportedAt=new Date().toISOString();C.validateBackup(b,S.book,S.hash);return b;}
$('#backup').onclick=run(async()=>{const b=await backupSnapshot();download(`${b.packageId}_${b.reviewer}_progress.ecgprogress`,JSON.stringify(b,null,2),'application/json');});
$('#export').onclick=run(async()=>{const b=await backupSnapshot(),done=S.book.items.filter(t=>b.records[t.id]?.status==='complete').length;if(done<S.book.items.length&&!confirm(`Completed ${done}/${S.book.items.length} segments. Export partial results? Incomplete statuses will be retained.`))return;const entries=C.exportsFor(S.book,b,S.waves);entries['progress.ecgprogress']=JSON.stringify(b);entries['README.txt']='Daily HRV Local Review v1\nAll times are relative to each original waveform file. Sample indices are zero-based.\nUnreadable intervals retain their peaks; exclude them appropriately downstream.\nCheck task_reviews.csv statuses before analysis. No formal dataset was modified.\n';download(`${b.packageId}_${b.reviewer}_results.zip`,C.zip(entries),'application/zip');});
$('#restore').onclick=()=>$('#backup-input').click();$('#choose-backup-home').onclick=()=>$('#backup-input').click();
$('#backup-input').onchange=run(async e=>{const f=e.target.files[0];e.target.value='';if(!f)return;if(f.size>50*1024*1024)throw Error('Backup exceeds the 50 MB limit.');const b=JSON.parse(await f.text());if(!S.book){if(b.format!=='ecg-progress'||b.version!==1||!C.id(b.reviewer))throw Error('Invalid backup format.');S.pendingBackup=b;$('#reviewer').value=b.reviewer;$('#import-status').textContent='Backup read. Select its original package folder to validate and restore.';notice('Select the original package folder to restore the backup.');return;}await flush();C.validateBackup(b,S.book,S.hash);if(b.reviewer!==S.reviewer)throw Error('Reviewer names differ. Return home and use the name from the backup.');const existing=S.backup.history;if(existing.length> b.history.length||existing.some((r,i)=>JSON.stringify(r)!==JSON.stringify(b.history[i])))throw Error('The backup is older or has diverged. Export local results for the team to reconcile.');const revision=await DB.saveSession(sessionKey(),b,S.revision);S.revision=revision;S.backup=b;loadTask(S.index);renderList();notice('Backup restored.');});
$('#back-home').onclick=run(async()=>{await flush();if(S.dirty)throw Error('Download a backup and resolve the save error before switching.');$('#workspace').hidden=true;$('#home').hidden=false;$('#principles').hidden=false;S.book=null;S.waves.clear();S.files=null;S.pendingBackup=null;});
$('#task-filter').oninput=()=>renderList();$('#duration-filter').onchange=()=>renderList();$('#sampling-filter').onchange=()=>renderList();$('#help-open').onclick=()=>$('#help').showModal();$('#help-close').onclick=()=>$('#help').close();
window.addEventListener('beforeunload',e=>{if(S.dirty){e.preventDefault();e.returnValue='';}});
function syntheticWave(seconds){const fs=130,t=[],x=[];const peaks=[];for(let p=.55;p<seconds;p+=.84)peaks.push(Math.round(p*fs));for(let i=0;i<seconds*fs;i++){const sec=i/fs;let y=9*Math.sin(2*Math.PI*.2*sec)+3*Math.sin(i*1.33);for(const p of peaks){const d=(i-p)/fs;if(Math.abs(d)<.5)y+=65*Math.exp(-(((d+.18)/.025)**2))-100*Math.exp(-(((d+.025)/.012)**2))+850*Math.exp(-((d/.014)**2))-160*Math.exp(-(((d-.025)/.013)**2))+180*Math.exp(-(((d-.25)/.065)**2));}t.push(Number(sec.toFixed(6)));x.push(Number(y.toFixed(3)));}return {t,x,peaks};}
$('#demo').onclick=run(async()=>{if(!$('#reviewer').value.trim())$('#reviewer').value='demo_reviewer';const w=syntheticWave(30),raw=new TextEncoder().encode(JSON.stringify({t:w.t,x:w.x})),digest=await C.sha256(raw),base={participant:'DEMO',file:'waveforms/demo.json',sha256:digest,fileBytes:raw.length,sampleCount:w.t.length,durationSec:w.t.at(-1),instructions:'Synthetic ECG for practice only. No participant data or medical labels.',initialPeakSource:'algorithm'};
  const manifest={format:'ecg-review',version:1,packageId:'synthetic_demo_rpeaks_v2',title:'Synthetic ECG · Practice review',createdAt:'2026-11-06T00:00:00Z',reviewMode:'assisted',instructions:'Practice R-peak review using synthetic ECG.'},items=[{...base,id:'demo_rpeaks',title:'01 · R-peak review',mode:'rpeaks',targets:[],contextPeaks:[],initialPeaks:w.peaks}];const book=C.zip({'manifest.json':JSON.stringify(manifest),'items.json':JSON.stringify(items)}),b=new File([book],'demo.ecgreview'),f=new File([raw],'demo.json');b._relative='demo.ecgreview';f._relative='waveforms/demo.json';await openFiles([b,f]);});

async function openZip(file){$('#import-status').textContent='Opening ZIP locally…';try{const files=await ECGArchive.files(file);if(files.some(f=>f.name==='history_manifest.json')){await ECGHistory.open(files);$('#import-status').textContent='';}else await openFiles(files);}catch(e){$('#import-status').textContent=e.message;$('#import-status').className='error';throw e;}}
$('#choose-zip').onclick=e=>{e.stopPropagation();$('#zip-input').click();};
$('#zip-drop').onclick=()=>$('#zip-input').click();
$('#zip-drop').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#zip-input').click();}};
$('#zip-input').onchange=run(async e=>{const f=e.target.files[0];e.target.value='';if(f)await openZip(f);});
$('#zip-drop').ondragover=e=>{e.preventDefault();$('#zip-drop').classList.add('drag-over');};
$('#zip-drop').ondragleave=()=>$('#zip-drop').classList.remove('drag-over');
$('#zip-drop').ondrop=run(async e=>{e.preventDefault();$('#zip-drop').classList.remove('drag-over');if(e.dataTransfer.files.length!==1)throw Error('Drop one package ZIP at a time.');await openZip(e.dataTransfer.files[0]);});

async function boot(){try{const reviewer=localStorage.getItem('ecg-reviewer');if(reviewer)$('#reviewer').value=reviewer;}catch{}try{if(window.showDirectoryPicker&&await DB.get('handles','last-folder'))$('#reconnect').hidden=false;}catch{notice('Browser storage is unavailable. Leave private browsing or try Chrome/Edge.',true);}if(location.protocol==='https:'||location.hostname==='127.0.0.1'||location.hostname==='localhost'){try{if('serviceWorker'in navigator){await navigator.serviceWorker.register('./sw.js');navigator.serviceWorker.addEventListener('message',e=>{if(e.data?.type==='offline-ready')$('#offline-status').textContent='Tool cached · Available offline in this browser';});}}catch{$('#offline-status').textContent='Offline cache unavailable · Use the offline tool package';}}}
boot();
