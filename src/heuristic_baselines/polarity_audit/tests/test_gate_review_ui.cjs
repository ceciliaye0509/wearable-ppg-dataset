const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const noop=()=>{},nodes=new Map();
function element(){return {value:'',checked:false,disabled:false,hidden:false,textContent:'',className:'',classList:{toggle:noop},replaceChildren:noop,append:noop,close:noop,showModal:noop,width:1000,height:320,getContext:()=>new Proxy({},{get:()=>noop,set:()=>true})};}
const get=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
const radios=['positive','negative','balanced','mixed','unreadable','unsure'].map(value=>Object.assign(element(),{value}));
const html=fs.readFileSync(path.join(__dirname,'../static/gate_review.html'),'utf8'),code=fs.readFileSync(path.join(__dirname,'../static/gate_review.js'),'utf8');
for(const id of code.matchAll(/\$\(['"]#([\w-]+)['"]\)/g))assert(html.includes(`id="${id[1]}"`),id[1]);
assert(!code.includes('/api/task/'),'interval review must not write main minute labels');
const calls=[],sandbox={document:{querySelector:get,querySelectorAll:()=>radios,createElement:element,hidden:false},window:{addEventListener:noop,setInterval:noop},location:{protocol:'file:'},confirm:()=>true,console,
 fetch:async(url,options={})=>{calls.push({url,options});return {ok:true,json:async()=>({version:1,reviewer:'QA',status:'complete',direction:'negative',tile_readability:Array(12).fill('readable')})};}};
vm.createContext(sandbox);vm.runInContext(code,sandbox);const run=s=>vm.runInContext(s,sandbox);
run(`state.task={id:'gate-demo',source_id:'truth-demo',participant:'P4',t:[0,.5,1,6],x:[0,-900,0,50],focusEndSec:60,rating:null,tiles:Array.from({length:12},(_,i)=>({start_sec:i*5,end_sec:(i+1)*5,span_uv:1000,over_threshold:false}))};state.tasks=[{id:'gate-demo',participant:'P4'}];applyRating(null);drawPlots();`);
assert(!radios.some(r=>r.checked));assert.equal(run('state.labels.filter(Boolean).length'),0,'interval labels start blank');
get('#readable').onclick();assert.equal(run('state.labels[0]'),'readable');assert.equal(run('state.dirty'),true);
get('#next').onclick();assert.equal(run('state.tile'),1);get('#unreadable').onclick();assert.equal(run('state.labels[1]'),'unreadable');get('#clear-tile').onclick();assert.equal(run('state.labels[1]'),'');
(async()=>{
 await run('save("complete")');assert.equal(calls.length,0,'no implicit interval completion');
 radios.find(r=>r.value==='negative').checked=true;run('state.labels=Array(12).fill("readable")');await run('save("complete")');assert.equal(calls.length,1);
 assert.equal(calls[0].url,'/gate-api/task/gate-demo/review');const body=JSON.parse(calls[0].options.body);assert.equal(body.tile_readability.length,12);assert(!('peak_indices' in body));
 run('state.task.rating={version:1,status:"complete",direction:"negative",reviewer:"QA",tile_readability:Array(12).fill("unreadable")};state.dirty=true;');get('#restore').onclick();assert.equal(run('state.labels[0]'),'unreadable');run('state.labels[0]="readable"');assert.equal(run('state.task.rating.tile_readability[0]'),'unreadable','edits do not mutate saved labels');
 run('dirty()');sandbox.fetch=async()=>({ok:false,status:409,json:async()=>({error:'conflict'})});await run('save("in_progress")');assert.equal(run('state.dirty'),true);assert.equal(run('state.version'),1);
 console.log('Gate review UI passed: blank interval labels, detail navigation, independent save route, completion guard, saved-label restore and conflict protection.');
})().catch(error=>{console.error(error);process.exitCode=1;});
