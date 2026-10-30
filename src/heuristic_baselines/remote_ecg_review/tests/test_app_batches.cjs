const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const elements = new Map(); let created = 0;
function element(selector) {
  if (!elements.has(selector)) elements.set(selector, {
    hidden: false, value: '', textContent: '', attributes: {}, listeners: {}, children: [], dataset: {}, style: {},
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener(name, callback) { this.listeners[name] = callback; },
    setAttribute(name, value) { this.attributes[name] = value; },
    close() {}, showModal() {}, focus() {}, replaceChildren() { this.children = []; },
    append(...children) { this.children.push(...children); },
  });
  return elements.get(selector);
}
const sandbox = { assert, console,
  document: { querySelector: element, querySelectorAll: () => [], createElement: () => element(`created-${created++}`) },
  window: { location: { protocol: 'file:' }, setTimeout() {}, setInterval() {}, addEventListener() {}, confirm: () => true },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8'), sandbox);
vm.runInContext(`
drawPlot = () => {};
state.participants = [{participant:'P1',total:3,finished:1,saved:1,label:'In progress',batches:{
 existing:{total:1,finished:1,saved:1,label:'Complete'}, independent_blind_v1:{total:2,finished:0,saved:0,label:'Not started'}
}}];
state.batches = [{id:'existing',label:'Existing review'},{id:'independent_blind_v1',label:'Independent blind validation'}];
state.batch = 'independent_blind_v1'; drawBatches(); drawParticipants();
assert.equal($('#batch-filter').value,'independent_blind_v1');
assert.equal($('#batch-summary').textContent,'0/2 segments complete in this batch');
const clips = [{id:'old'},{id:'new1',reviewBatch:'independent_blind_v1'},{id:'new2',reviewBatch:'independent_blind_v1'}];
assert.equal(clipsInBatch(clips).length,2);
state.batch = 'existing'; assert.equal(clipsInBatch(clips)[0].id,'old'); drawParticipants();
assert.equal($('#batch-summary').textContent,'1/1 segments complete in this batch');
state.batch = 'all'; assert.equal(clipsInBatch(clips).length,3);
state.batch = 'independent_blind_v1'; state.clips = [{id:'new1',status:'unreviewed',startSec:20,endSec:40,blindReview:true,sourceLabel:'Independent blind validation'}]; state.index = 0;
api = async () => ({id:'new1',blindReview:true,sourceLabel:'Independent blind validation',peaks:[],suggested:[],restorePeaks:[],
 unreadableRanges:[],version:0,confidence:'high',status:'unreviewed',notes:'',startSec:20,endSec:40,
 t:[0,10,20],x:[0,1,0],qualityFlags:[],qualitySummary:{available:false,withheldForBlindReview:true}});
loadClip();
`, sandbox).then(() => {
  vm.runInContext(`
assert.equal(state.peaks.length,0);
assert.equal($('#reset-candidates').hidden,true);
assert.equal($('#quality-panel').hidden,true);
assert.match($('#clip-meta').textContent,/full 20-second segment/);
assert.match($('#blind-note').textContent,/no algorithm candidate peaks/);
`, sandbox);
  return vm.runInContext(`
state.clip = null;
api = async () => ({id:'new1',blindReview:false,wasBlindReview:true,sourceLabel:'Validation (algorithm-assisted; formerly blind)',
 assistance:{method:'neurokit_local'},screeningWithheld:true,aidAvailable:true,peaks:[1],suggested:[0,1,2],restorePeaks:[0,1,2],
 unreadableRanges:[[0,1]],version:2,confidence:'high',status:'complete',notes:'saved note',startSec:20,endSec:40,
 t:[0,10,20],x:[0,1,0],qualityFlags:[],qualitySummary:{available:false}});
loadClip();
`, sandbox);
}).then(() => {
  vm.runInContext(`
assert.equal(state.peaks.join(','),'1');
assert.equal(state.unreadableRanges[0].join(','),'0,1');
assert.equal($('#reset-candidates').hidden,false);
assert.equal($('#reset-candidates').textContent,'Restore algorithm suggestions');
assert.equal($('#quality-panel').hidden,true);
assert.match($('#clip-meta').textContent,/not independent blind validation/);
assert.match($('#blind-note').textContent,/Algorithm-assisted peak review/);
$('#reset-candidates').onclick();
assert.equal(state.peaks.join(','),'0,1,2');
assert.equal(state.unreadableRanges[0].join(','),'0,1');
assert.equal(state.savedPeaks.join(','),'1');
$('#restore-saved').onclick();
assert.equal(state.peaks.join(','),'1');
assert.equal(state.unreadableRanges[0].join(','),'0,1');
`, sandbox);
  console.log('Batch isolation, separate progress, blank blind canvas, hidden suggestions and screening: passed.');
}).catch(error => { console.error(error); process.exitCode = 1; });
