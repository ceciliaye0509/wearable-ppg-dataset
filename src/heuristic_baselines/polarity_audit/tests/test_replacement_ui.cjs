const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const noop=()=>{},nodes=new Map();
function element(){return {value:'',checked:false,disabled:false,hidden:false,textContent:'',className:'',classList:{toggle:noop},replaceChildren:noop,append:noop,close:noop,showModal:noop,width:1000,height:320,getContext:()=>new Proxy({},{get:()=>noop,set:()=>true})};}
const get=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
const radios=['positive','negative','balanced','mixed','unreadable','unsure'].map(value=>Object.assign(element(),{value}));
const code=fs.readFileSync(path.join(__dirname,'../static/direction_truth.js'),'utf8');const calls=[];
const sandbox={document:{querySelector:get,querySelectorAll:()=>radios,createElement:element,hidden:false},window:{addEventListener:noop,setInterval:noop},location:{protocol:'file:',search:'?set=replacement'},confirm:()=>true,console,
fetch:async(url,options={})=>{calls.push({url,options});return {ok:true,json:async()=>({version:1,reviewer:'QA',direction:'negative',status:'complete',review_mode:'raw_direction_labeling_v1'})};}};
vm.createContext(sandbox);vm.runInContext(code,sandbox);const run=s=>vm.runInContext(s,sandbox);
assert.equal(run('state.cohort'),'replacement');
run(`state.hasRepair=true;state.participants=[{participant:'P17'}];state.tasks=[
 {id:'retained',participant:'P17',queue:'main',in_corrected_main:true,rating:{status:'complete'}},
 {id:'superseded',participant:'P17',queue:'main',in_corrected_main:false,superseded_in_corrected_main:true,rating:{status:'complete'}},
 {id:'fresh',participant:'P17',queue:'replacement',in_corrected_main:true}];
 state.cohortTasks={main:['retained','superseded'],corrected:['retained','fresh'],replacement:['fresh']};
 state.task={id:'fresh',participant:'P17',queue:'replacement',t:[0,.5,1],x:[0,-500,0],focusEndSec:60,rating:null};state.participant='P17';applyRating(null);refreshHeader();`);
assert(!radios.some(r=>r.checked));assert.equal(run('visibleTasks().length'),1);assert.equal(get('#results').disabled,true);
run('state.cohort="corrected";refreshHeader()');assert.equal(run('visibleTasks().length'),2);assert(!run('visibleTasks().some(t=>t.id==="superseded")'));assert.equal(get('#results').disabled,true);
run('state.cohort="replacement";refreshHeader()');
(async()=>{
 await run('save("complete")');assert.equal(calls.length,0);
 radios.find(r=>r.value==='negative').checked=true;await run('save("complete")');assert.equal(calls.length,1);assert.equal(calls[0].url,'/api/task/fresh/review');
 run('state.cohort="corrected";refreshHeader()');assert.equal(get('#results').disabled,false);
 assert.equal(run('state.tasks.find(t=>t.id==="superseded").rating.status'),'complete');
 run('state.cohort="main";refreshHeader()');assert.equal(run('visibleTasks().length'),2,'historical records remain available');
 console.log('Replacement UI passed: blank new labels, repaired membership, original record retention, corrected-score gate and independent new-label save.');
})().catch(error=>{console.error(error);process.exitCode=1;});
