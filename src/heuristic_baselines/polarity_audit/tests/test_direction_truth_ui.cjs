const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const noop=()=>{},nodes=new Map();
function element(){return {value:'',checked:false,disabled:false,hidden:false,textContent:'',className:'',classList:{toggle:noop},replaceChildren:noop,append:noop,close:noop,showModal:noop,width:1000,height:320,getContext:()=>new Proxy({},{get:()=>noop,set:()=>true})};}
const get=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
const radios=['positive','negative','balanced','mixed','unreadable','unsure'].map(value=>Object.assign(element(),{value}));
const html=fs.readFileSync(path.join(__dirname,'../static/direction_truth.html'),'utf8'),code=fs.readFileSync(path.join(__dirname,'../static/direction_truth.js'),'utf8');
for(const id of code.matchAll(/\$\(['"]#([\w-]+)['"]\)/g))assert(html.includes(`id="${id[1]}"`),id[1]);
assert(!html.includes('Method A'));assert(!html.includes('plot-A'));assert(!code.includes('task.sides'));
const calls=[],sandbox={document:{querySelector:get,querySelectorAll:()=>radios,createElement:element,hidden:false},window:{addEventListener:noop,setInterval:noop},location:{protocol:'file:'},confirm:()=>true,console,
 fetch:async(url,options={})=>{calls.push({url,options});return {ok:true,json:async()=>({version:1,reviewer:'QA',direction:'negative',status:'complete',review_mode:'raw_direction_labeling_v1'})};}};
vm.createContext(sandbox);vm.runInContext(code,sandbox);const run=s=>vm.runInContext(s,sandbox);
run(`state.task={id:'truth-demo',participant:'P4',queue:'priority',t:[0,.5,1,2,3,4,5,6,7,8,9],x:[0,-900,0,0,0,0,0,0,0,0,0],focusEndSec:10,rating:null};
 state.participant='P4';state.cohort='main';state.tasks=[{id:'truth-demo',participant:'P4',queue:'main'},{id:'optional-demo',participant:'P8',queue:'optional'}];state.participants=[{participant:'P4'},{participant:'P8'}];applyRating(null);drawPlot();refreshHeader();drawParticipants();`);
assert(!radios.some(r=>r.checked),'fresh labels must start blank');assert.equal(run('visibleTasks().length'),1);assert.equal(get('#results').disabled,true);
get('#detail').onclick();assert.equal(run('state.zoom'),0);get('#next').onclick();assert.equal(run('state.zoom'),5);get('#prev').onclick();assert.equal(run('state.zoom'),0);
run('state.dirty=true');sandbox.confirm=()=>false;get('#cohort').value='optional';get('#cohort').onchange();assert.equal(run('state.dirty'),true);assert.equal(run('state.cohort'),'main');assert.equal(run('visibleTasks().length'),1);
run('state.dirty=false');sandbox.confirm=()=>true;
(async()=>{
 await run('showResults()');assert.equal(calls.length,0,'no algorithm results before labels');
 await run('save("complete")');assert.equal(calls.length,0,'no completing an unselected label');
 radios.find(r=>r.value==='negative').checked=true;await run('save("complete")');assert.equal(calls.length,1);
 const body=JSON.parse(calls[0].options.body);assert.equal(body.direction,'negative');assert(!('choice' in body));assert.equal(run('state.resultsReady'),true);assert.equal(get('#results').disabled,false);
 run(`state.task.rating={direction:'balanced',version:1,reviewer:'QA',status:'in_progress',notes:'saved'};state.dirty=true;`);get('#restore').onclick();assert.equal(radios.find(r=>r.checked).value,'balanced');assert.equal(run('state.dirty'),false);
 run('applyRating(null)');assert(!radios.some(r=>r.checked),'no previous label carried to another task');
 run('dirty()');assert.equal(get('#comparison-results').hidden,true);assert.equal(get('#results').disabled,true);const before=calls.length;await run('showResults()');assert.equal(calls.length,before,'no results while editing unsaved labels');
 radios.find(r=>r.value==='positive').checked=true;sandbox.fetch=async()=>({ok:false,status:409,json:async()=>({error:'conflict'})});await run('save("in_progress")');assert.equal(run('state.dirty'),true);assert.equal(run('state.version'),1);assert.equal(get('#labels').disabled,false);
 run('state.dirty=false');get('#cohort').value='optional';get('#cohort').onchange();assert.equal(run('state.cohort'),'optional');assert.equal(run('state.task'),null);assert.equal(run('visibleTasks().length'),1);assert.equal(get('#results').disabled,true);
 console.log('Direction-label UI passed: blank start, no predictions, manual labels, cohort switching, unsaved protection, zoom, save, restore, result gate and conflicts.');
})().catch(error=>{console.error(error);process.exitCode=1;});
