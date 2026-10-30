"use strict";
const $ = s => document.querySelector(s);
const radios = () => [...document.querySelectorAll('input[name="preference"]')];
const state = {tasks:[],participants:[],participant:null,task:null,version:0,dirty:false,busy:false,signedIn:false,zoom:null,showVotes:true};
async function api(path, options={}) {
  const response=await fetch(path,{credentials:"same-origin",...options});
  const value=await response.json();
  if(!response.ok){const error=new Error(value.error||`Request failed (${response.status})`);error.status=response.status;throw error;}
  return value;
}
function message(text,error=false){$('#message').textContent=text;$('#message').classList.toggle('error',error);}
function dirty(){state.dirty=true;message('Unsaved choice');}
function okayToLeave(){return !state.dirty||confirm('Discard the unsaved choice and notes for this minute?');}
function setBusy(value){state.busy=value;['#save','#complete','#restore','#choices','#confidence','#notes'].forEach(s=>{$(s).disabled=value;});}
async function refresh(){const d=await api('/api/comparison/tasks');state.tasks=d.tasks;state.participants=d.participants;if(state.task)drawTasks();else drawParticipants();}
async function boot(){
  if(!['http:','https:'].includes(location.protocol)){$('#startup-error').hidden=false;$('#startup-error').textContent='Start the audit service and open http://127.0.0.1:8766/; do not open this file directly.';return;}
  try{const me=await api('/api/me');state.signedIn=true;$('#reviewer').textContent=`Reviewer: ${me.reviewer}`;$('#login').close();await refresh();}
  catch(error){if(error.status===401)$('#login').showModal();else{$('#startup-error').hidden=false;$('#startup-error').textContent=error.message;}}
}
function drawParticipants(){
  $('#participants').replaceChildren();
  $('#overall').textContent=`${state.tasks.filter(t=>t.rating?.status==='complete').length}/${state.tasks.length} minutes complete across ${state.participants.length} participants`;
  state.participants.forEach(p=>{
    const rows=state.tasks.filter(t=>t.participant===p.participant),done=rows.filter(t=>t.rating?.status==='complete').length;
    const button=document.createElement('button');button.className=`participant${state.participant===p.participant?' active':''}`;
    const name=document.createElement('strong');name.textContent=p.participant;
    const detail=document.createElement('small');detail.textContent=rows.length?`${done}/${rows.length} complete · ${p.eligible_minutes} eligible`:p.differing_blocks?'No eligible full-minute disagreements':'No differing direction results';
    button.append(name,detail);button.disabled=!rows.length;button.onclick=()=>chooseParticipant(p.participant);$('#participants').append(button);
  });
}
function drawTasks(){
  const rows=state.tasks.filter(t=>t.participant===state.participant),stratum=state.participants.find(p=>p.participant===state.participant);
  $('#participant-title').textContent=state.participant;
  $('#progress').textContent=`${rows.filter(t=>t.rating?.status==='complete').length}/${rows.length} complete · randomly sampled from ${stratum.eligible_minutes} eligible differing minutes`;
  $('#tasks').replaceChildren();rows.forEach((row,i)=>{
    const b=document.createElement('button');b.className=`${row.rating?.status||''}${state.task?.id===row.id?' active':''}`;
    b.textContent=`${row.rating?.status==='complete'?'Complete':row.rating?'In progress':'Not reviewed'} · ${i+1}`;
    b.onclick=()=>{if(!state.busy&&okayToLeave())load(row.id);};$('#tasks').append(b);
  });drawParticipants();
}
async function chooseParticipant(pid){if(state.busy||!okayToLeave())return;const rows=state.tasks.filter(t=>t.participant===pid);if(!rows.length)return;await load((rows.find(t=>t.rating?.status!=='complete')||rows[0]).id);}
function applyRating(rating){
  const r=rating||{};radios().forEach(input=>{input.checked=input.value===(r.choice||'');});
  $('#confidence').value=r.confidence||'medium';$('#notes').value=r.notes||'';state.dirty=false;
  message(r.version?`Saved by ${r.reviewer} · ${r.status==='complete'?'Complete':'In progress'}`:'Not reviewed — choose A, B, equally reasonable, neither, or cannot judge');
}
async function load(id){
  setBusy(true);
  try{const task=await api(`/api/comparison/task/${encodeURIComponent(id)}`);state.task=task;state.participant=task.participant;state.version=task.rating?.version||0;state.zoom=null;state.showVotes=true;applyRating(task.rating);
    $('#empty').hidden=true;$('#panel').hidden=false;$('#task-label').textContent=`${task.id} · compare the whole 0–${task.focusEndSec.toFixed(1)} second minute`;
    drawTasks();drawPlots();
  }catch(error){message(error.message,true);}finally{setBusy(false);}
}
function viewBounds(){const duration=state.task.focusEndSec;return state.zoom===null?[0,duration]:[state.zoom,Math.min(state.zoom+5,duration)];}
function directionText(label){return ({positive:'Predominantly upward',negative:'Predominantly downward',ambiguous:'Direction not determined',low_signal:'Insufficient signal for direction',severe_artifact:'Severe amplitude artifact'})[label]||'Direction not assessed';}
function actionText(decision){if(decision==='stable_negative')return 'Interval action: reverse within a sustained negative interval';if(decision==='stable_positive')return 'Interval action: keep within a sustained positive interval';return 'Interval action: no automatic flip; review required';}
function drawPlots(){
  if(!state.task)return;
  const task=state.task,[left,right]=viewBounds();let low=Infinity,high=-Infinity;
  task.x.forEach((v,i)=>{if(task.t[i]>=left&&task.t[i]<=right){low=Math.min(low,v);high=Math.max(high,v);}});
  if(!Number.isFinite(low)||!Number.isFinite(high))return;
  const margin=(high-low)*.08||1;low-=margin;high+=margin;
  $('#view-label').textContent=state.zoom===null?`Full minute · 0–${right.toFixed(1)} s`:`Detail · ${left.toFixed(1)}–${right.toFixed(1)} s`;
  for(const side of ['A','B']){
    const data=task.sides[side],canvas=$(`#plot-${side}`),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;
    $('#result-'+side).textContent=directionText(data.direction);$('#action-'+side).textContent=actionText(data.decision);
    $('#stats-'+side).textContent=`${data.event_count} candidates · ${data.positive_votes} upward / ${data.negative_votes} downward / ${data.uncertain_beats} uncertain · net score ${data.score===null?'unavailable':data.score.toFixed(3)}`;
    const X=v=>65+(v-left)/Math.max(.001,right-left)*(W-85),Y=v=>18+(high-v)/(high-low)*(H-58);
    ctx.clearRect(0,0,W,H);ctx.font='12px system-ui';ctx.lineWidth=1;
    for(let i=0;i<=4;i++){
      const y=18+(H-58)*i/4,sec=left+(right-left)*i/4;ctx.strokeStyle='#e2ebed';ctx.fillStyle='#687c83';
      ctx.beginPath();ctx.moveTo(65,y);ctx.lineTo(W-20,y);ctx.stroke();ctx.fillText((high-(high-low)*i/4).toFixed(0),4,y+4);
      ctx.beginPath();ctx.moveTo(X(sec),18);ctx.lineTo(X(sec),H-40);ctx.stroke();ctx.fillText(`${sec.toFixed(1)}s`,X(sec)-15,H-17);
    }
    ctx.fillText('µV',5,12);ctx.strokeStyle='#244d56';ctx.lineWidth=1.1;ctx.beginPath();let first=true;
    task.x.forEach((v,i)=>{if(task.t[i]>=left&&task.t[i]<=right){if(first)ctx.moveTo(X(task.t[i]),Y(v));else ctx.lineTo(X(task.t[i]),Y(v));first=false;}});ctx.stroke();
    if(state.showVotes)data.votes.forEach(v=>{const i=v.index;if(task.t[i]>=left&&task.t[i]<=right){ctx.fillStyle=v.sign>0?'#17834e':v.sign<0?'#9b429a':'#b47808';ctx.beginPath();ctx.arc(X(task.t[i]),Y(task.x[i]),4,0,2*Math.PI);ctx.fill();}});
  }
  $('#prev').disabled=state.zoom===null||state.zoom<=0;$('#next').disabled=state.zoom!==null&&state.zoom+5>=task.focusEndSec;
  $('#toggle-votes').textContent=state.showVotes?'Hide vote marks':'Show vote marks';
}
async function save(status){
  if(!state.task||state.busy)return;
  const choice=radios().find(r=>r.checked)?.value||'';
  if(status==='complete'&&!choice){message('Choose one comparison before completing this minute.',true);return;}
  const id=state.task.id,body={choice,status,confidence:$('#confidence').value,notes:$('#notes').value,version:state.version};setBusy(true);
  try{const rating=await api(`/api/comparison/task/${encodeURIComponent(id)}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    state.task.rating=rating;state.version=rating.version;state.dirty=false;state.tasks.find(t=>t.id===id).rating=rating;message('Comparison saved');drawTasks();
    if(status==='complete'){const next=state.tasks.find(t=>t.participant===state.participant&&t.id!==id&&t.rating?.status!=='complete');if(next)await load(next.id);else message('All sampled disagreements for this participant are complete');}
  }catch(error){message(error.status===409?'Another device saved this minute. Your edits remain unsaved; reload this minute before trying again.':error.message,true);}
  finally{setBusy(false);}
}
$('#login-form').onsubmit=async event=>{event.preventDefault();$('#login-error').textContent='';try{await api('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewer:$('#name').value,accessCode:$('#code').value})});$('#code').value='';await boot();}catch(error){$('#login-error').textContent=error.message;}};
radios().forEach(input=>{input.onchange=dirty;});$('#confidence').onchange=dirty;$('#notes').oninput=dirty;
$('#minute').onclick=()=>{if(!state.task)return;state.zoom=null;drawPlots();};$('#detail').onclick=()=>{if(!state.task)return;state.zoom=0;drawPlots();};
$('#prev').onclick=()=>{if(!state.task)return;state.zoom=Math.max(0,(state.zoom||0)-5);drawPlots();};
$('#next').onclick=()=>{if(!state.task)return;state.zoom=state.zoom===null?0:Math.min(Math.floor((state.task.focusEndSec-.001)/5)*5,state.zoom+5);drawPlots();};
$('#toggle-votes').onclick=()=>{state.showVotes=!state.showVotes;drawPlots();};
$('#restore').onclick=()=>{if(state.busy||!state.task||!okayToLeave())return;applyRating(state.task.rating);};
$('#save').onclick=()=>save('in_progress');$('#complete').onclick=()=>save('complete');
window.addEventListener('beforeunload',event=>{if(state.dirty){event.preventDefault();event.returnValue='';}});
window.addEventListener('resize',drawPlots);
window.setInterval(async()=>{if(state.signedIn&&!state.busy&&!document.hidden)try{await refresh();}catch(error){if(error.status===401)state.signedIn=false;}},15000);
boot();
