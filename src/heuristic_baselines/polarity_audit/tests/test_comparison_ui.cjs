const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const noop=()=>{},nodes=new Map();
function element(){return {value:'',checked:false,disabled:false,hidden:false,textContent:'',className:'',classList:{toggle:noop},
  replaceChildren:noop,append:noop,close:noop,showModal:noop,width:1000,height:340,
  getContext:()=>new Proxy({},{get:()=>noop,set:()=>true})};}
const get=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
const radios=['A','B','same','neither','unsure'].map(value=>Object.assign(element(),{value}));
const html=fs.readFileSync(path.join(__dirname,'../static/comparison.html'),'utf8');
const code=fs.readFileSync(path.join(__dirname,'../static/comparison.js'),'utf8');
for(const id of code.matchAll(/\$\(['"]#([\w-]+)['"]\)/g))assert(html.includes(`id="${id[1]}"`),id[1]);
const calls=[];
const sandbox={document:{querySelector:get,querySelectorAll:()=>radios,createElement:element,hidden:false},
  window:{addEventListener:noop,setInterval:noop},location:{protocol:'file:'},confirm:()=>true,console,
  fetch:async(url,options={})=>{calls.push({url,options});return {ok:true,json:async()=>({version:1,reviewer:'QA',choice:'B',status:'complete'})};}};
vm.createContext(sandbox);vm.runInContext(code,sandbox);
const run=s=>vm.runInContext(s,sandbox);
run(`state.task={id:'pair-test',participant:'P2',t:[0,.5,1,2,3,4,5,6,7,8,9],x:[0,900,0,0,0,0,0,0,0,0,0],focusStartSec:0,focusEndSec:10,
  sides:{A:{direction:'positive',decision:'stable_positive',score:1,event_count:1,positive_votes:1,negative_votes:0,uncertain_beats:0,votes:[{index:1,sign:1}]},
         B:{direction:'ambiguous',decision:'review_ambiguous_polarity',score:0,event_count:2,positive_votes:1,negative_votes:1,uncertain_beats:0,votes:[{index:1,sign:1},{index:2,sign:-1}]}},rating:null};
  state.participant='P2';state.participants=[{participant:'P2',eligible_minutes:10}];state.tasks=[{id:'pair-test',participant:'P2'}];applyRating(null);drawPlots();`);
assert(!radios.some(r=>r.checked),'never select automatically');
assert.equal(get('#result-A').textContent,'Predominantly upward');
get('#detail').onclick();assert.equal(run('state.zoom'),0);
get('#next').onclick();assert.equal(run('state.zoom'),5);assert.equal(get('#next').disabled,true);
get('#prev').onclick();assert.equal(run('state.zoom'),0);
get('#minute').onclick();assert.equal(run('state.zoom'),null);
get('#toggle-votes').onclick();assert.equal(run('state.showVotes'),false);assert(!radios.some(r=>r.checked));
run(`applyRating({choice:'A',confidence:'high',notes:'saved',version:1,reviewer:'QA',status:'complete'});`);
assert.equal(radios.find(r=>r.checked).value,'A');
run('applyRating(null)');assert(!radios.some(r=>r.checked),'do not carry previous choice to new task');
(async()=>{
  await run('save("complete")');assert.equal(calls.length,0,'cannot complete without choosing');
  radios.find(r=>r.value==='B').checked=true;
  await run('save("complete")');assert.equal(calls.length,1);
  const body=JSON.parse(calls[0].options.body);assert.equal(body.choice,'B');assert.equal(body.version,0);
  assert(!('method_id' in body));assert.equal(run('state.version'),1);assert.equal(run('state.dirty'),false);
  run(`state.task.rating={choice:'same',confidence:'medium',notes:'saved choice',version:1,reviewer:'QA',status:'in_progress'};state.dirty=true;`);
  get('#restore').onclick();assert.equal(radios.find(r=>r.checked).value,'same');assert.equal(run('state.dirty'),false);
  sandbox.fetch=async()=>({ok:false,status:409,json:async()=>({error:'conflict'})});
  run('state.dirty=true');await run('save("in_progress")');assert.equal(run('state.dirty'),true);assert.equal(run('state.version'),1);
  assert.equal(get('#choices').disabled,false);console.log('Comparison UI checks passed: choice, zoom, save, conflict, no auto-answer');
})().catch(error=>{console.error(error);process.exitCode=1;});
