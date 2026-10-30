// Lightweight event-level checks without a browser or live patient-data mutation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const nodes = new Map();
const noop = () => {};
function element() {
  return {value:'',checked:false,disabled:false,hidden:false,textContent:'',
    replaceChildren:noop,append:noop,close:noop,showModal:noop,
    width:1200,height:380,getBoundingClientRect:()=>({left:0,top:0,width:1200,height:380}),
    getContext:()=>new Proxy({}, {get:()=>noop,set:()=>true})};
}
const get = (id) => { if (!nodes.has(id)) nodes.set(id,element()); return nodes.get(id); };
const html = fs.readFileSync(path.join(__dirname,'../static/index.html'),'utf8');
const code = fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8');
for (const id of code.matchAll(/\$\("#([\w-]+)"\)/g)) assert(html.includes(`id="${id[1]}"`),id[1]);
const sandbox = {document:{querySelector:get,createElement:element},
  window:{addEventListener:noop,setInterval:noop},location:{protocol:'file:'},confirm:()=>true,console};
vm.createContext(sandbox); vm.runInContext(code,sandbox);
const run = (s) => vm.runInContext(s,sandbox);
run(`state.task={id:'test',t:[0,.5,1,1.5,2,2.5,3,3.5,4,4.5],x:[0,2,0,0,0,0,0,0,0,0],focusStartSec:0,focusEndSec:5,
  assistance:{peak_indices:[1]},rating:{peak_indices:[3],peak_origin:'manual',peaks_reviewed:true}};
  state.peaks=[1];state.peakOrigin='algorithm';state.zoom=0;drawPlot();`);
get('#peaks-reviewed').checked=true;
// Remove a suggested mark and invalidate previous verification.
run(`$('#plot').onclick({currentTarget:$('#plot'),clientX:state.plotTransform.X(.5),clientY:state.plotTransform.Y(2)});`);
assert.equal(run('state.peaks.length'),0); assert.equal(get('#peaks-reviewed').checked,false);
// Add an exact sample, then restore saved and algorithm marks without editing ratings.
run(`$('#plot').onclick({currentTarget:$('#plot'),clientX:state.plotTransform.X(2),clientY:state.plotTransform.Y(0)});`);
assert.equal(run('state.peaks[0]'),4);
get('#qrs').value='partially_readable';
get('#restore-peaks').onclick(); assert.equal(run('state.peaks[0]'),3);
assert.equal(get('#peaks-reviewed').checked,true); assert.equal(get('#qrs').value,'partially_readable');
get('#use-suggestions').onclick(); assert.equal(run('state.peaks[0]'),1);
assert.equal(get('#peaks-reviewed').checked,false);
get('#toggle-peaks').onclick(); assert.equal(run('state.peaks[0]'),1);
// Full-minute clicks do not silently edit a nearby peak.
run(`state.zoom=null;$('#plot').onclick({});`);assert.equal(run('state.peaks[0]'),1);
// An existing saved empty list wins over algorithm suggestions, not replaced by truthiness.
sandbox.fetch=async()=>({ok:true,json:async()=>({id:'test',t:[0,.5,1],x:[0,2,0],focusStartSec:0,focusEndSec:1,
  assistance:{peak_indices:[1],notice:'Check'},rating:{version:2,peak_indices:[],peak_origin:'manual',peaks_reviewed:true,
    qrs_readability:'readable',direction_determinability:'no',direction:'undetermined',confidence:'high'}})});
run('state.participants=[];state.tasks=[];load("test")').then(()=>{
  assert.equal(run('state.peaks.length'),0);assert.equal(get('#qrs').value,'readable');
  assert.equal(get('#peaks-reviewed').checked,true);console.log('UI interaction checks passed');
}).catch(error=>{console.error(error);process.exitCode=1;});
