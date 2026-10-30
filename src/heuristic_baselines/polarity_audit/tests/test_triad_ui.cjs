const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const noop=()=>{},nodes=new Map();
function element(){return {value:'',checked:false,disabled:false,hidden:false,textContent:'',className:'',classList:{toggle:noop},replaceChildren:noop,append:noop,close:noop,showModal:noop,width:1000,height:290,getContext:()=>new Proxy({},{get:()=>noop,set:()=>true})};}
const get=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
const radios=['A','B','C','AB','AC','BC','same','neither','unsure'].map(value=>Object.assign(element(),{value}));
const html=fs.readFileSync(path.join(__dirname,'../static/triad.html'),'utf8'),code=fs.readFileSync(path.join(__dirname,'../static/triad.js'),'utf8');
for(const id of code.matchAll(/\$\(['"]#([\w-]+)['"]\)/g))assert(html.includes(`id="${id[1]}"`),id[1]);
const calls=[],sandbox={document:{querySelector:get,querySelectorAll:()=>radios,createElement:element,hidden:false},window:{addEventListener:noop,setInterval:noop},location:{protocol:'file:'},confirm:()=>true,console,
 fetch:async(url,options={})=>{calls.push({url,options});return {ok:true,json:async()=>({version:1,reviewer:'QA',choice:'BC',status:'complete'})};}};
vm.createContext(sandbox);vm.runInContext(code,sandbox);const run=s=>vm.runInContext(s,sandbox);
run(`state.task={id:'new-minute',participant:'P4',queue:'priority',t:[0,.5,1,2,3,4,5,6,7,8,9],x:[0,-900,0,0,0,0,0,0,0,0,0],focusEndSec:10,
 sides:{A:{direction:'ambiguous',support:'historical_rule',decision:'review_required',flip:false},B:{direction:'negative',support:'supported',decision:'stable_negative',flip:true},C:{direction:'negative',support:'limited_support',decision:'review_required',flip:false}},rating:null};
 state.participant='P4';state.tasks=[{id:'new-minute',participant:'P4',queue:'priority'},{id:'optional-minute',participant:'P8',queue:'optional'}];state.participants=[{participant:'P4',reused_direction_records:8},{participant:'P8',reused_direction_records:0}];applyRating(null);drawPlots();drawParticipants();`);
assert(!radios.some(r=>r.checked),'no automatic answer');
assert.equal(run('visibleTasks().length'),1,'optional checks hidden initially');
assert.equal(get('#result-B').textContent,'Predominantly downward');
get('#detail').onclick();assert.equal(run('state.zoom'),0);get('#next').onclick();assert.equal(run('state.zoom'),5);get('#prev').onclick();assert.equal(run('state.zoom'),0);
get('#display').onclick();assert.equal(run('state.proposed'),false);assert(!radios.some(r=>r.checked));
run('state.dirty=true');get('#optional').checked=true;get('#optional').onchange();assert.equal(run('visibleTasks().length'),2);assert.equal(run('state.dirty'),true,'showing optional tasks preserves unsaved priority edits');
run('state.dirty=false');
(async()=>{
 await run('save("complete")');assert.equal(calls.length,0,'require an explicit choice');
 radios.find(r=>r.value==='BC').checked=true;
 await run('save("complete")');assert.equal(calls.length,1);const body=JSON.parse(calls[0].options.body);assert.equal(body.choice,'BC');assert.equal(body.version,0);assert(!('method_id' in body));
 assert.equal(run('state.version'),1);assert.equal(run('state.dirty'),false);
 run(`state.task.rating={choice:'same',version:1,reviewer:'QA',status:'in_progress',notes:'saved'};state.dirty=true;`);get('#restore').onclick();assert.equal(radios.find(r=>r.checked).value,'same');assert.equal(run('state.dirty'),false);
 run('applyRating(null)');assert(!radios.some(r=>r.checked),'do not carry answers to the next task');
 radios.find(r=>r.value==='A').checked=true;run('state.dirty=true');sandbox.fetch=async()=>({ok:false,status:409,json:async()=>({error:'conflict'})});await run('save("in_progress")');assert.equal(run('state.dirty'),true);assert.equal(run('state.version'),1);assert.equal(get('#choices').disabled,false);
 console.log('Three-method UI checks passed: explicit choices, ties, display, optional checks, save, restore and conflicts.');
})().catch(error=>{console.error(error);process.exitCode=1;});
