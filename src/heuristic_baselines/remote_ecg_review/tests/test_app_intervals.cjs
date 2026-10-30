// Offline interaction regression: selecting/cancelling intervals cannot clear peaks.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const elements = new Map();
function element(selector) {
  if (!elements.has(selector)) elements.set(selector, {
    hidden: false, value: '', textContent: '', attributes: {}, listeners: {},
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener(name, callback) { this.listeners[name] = callback; },
    setAttribute(name, value) { this.attributes[name] = value; },
    close() {}, showModal() {}, focus() {}, replaceChildren() {}, append() {},
  });
  return elements.get(selector);
}
const sandbox = {
  assert, console,
  document: { querySelector: element, querySelectorAll: () => [], createElement: () => element(`created-${elements.size}`) },
  window: { location: { protocol: 'file:' }, setTimeout() {}, setInterval() {}, addEventListener() {}, confirm: () => true },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8'), sandbox);
vm.runInContext(`
drawPlot = () => {}; drawUnreadableRanges = () => {}; drawParticipants = () => {}; drawClipList = () => {};
state.clip = { id: 'test_clip', t: [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], suggested: [1, 5, 9], restorePeaks: [1, 5, 9] };
state.peaks = [2, 4, 8]; state.savedPeaks = [2, 4, 8]; state.savedRanges = [[5, 7]];
state.unreadableRanges = [[5, 7]];
$('#save-unresolvable').onclick();
assert.equal(state.rangeMode, true);
assert.equal(JSON.stringify(state.peaks), '[2,4,8]');
assert.equal($('#save-unresolvable').attributes['aria-pressed'], 'true');
// A browser double-click dispatches two clicks before dblclick.
$('#save-unresolvable').onclick();
$('#save-unresolvable').ondblclick({preventDefault() {}});
assert.equal(state.rangeMode, false);
assert.equal(JSON.stringify(state.peaks), '[2,4,8]');
assert.equal(JSON.stringify(state.unreadableRanges), '[[5,7]]');
const plot = $('#plot'); plot.width = 1100;
plot.getBoundingClientRect = () => ({left: 0, width: 1100}); plot.setPointerCapture = () => {};
state.zoom = -1;
$('#save-unresolvable').onclick();
const pointer = (clientX) => ({clientX, pointerId: 1, button: 0, preventDefault() {}});
plot.listeners.pointerdown(pointer(266)); plot.listeners.pointermove(pointer(470)); plot.listeners.pointerup(pointer(470));
assert.equal(JSON.stringify(state.unreadableRanges), '[[2,7]]');
assert.equal(JSON.stringify(state.peaks), '[2,4,8]');
$('#restore-saved').onclick();
$('#save-unresolvable').onclick();
plot.listeners.pointerdown(pointer(164)); plot.listeners.pointerup(pointer(164));
assert.equal(state.rangeStart, 1);
$('#save-unresolvable').ondblclick({preventDefault() {}});
assert.equal(state.rangeStart, null);
assert.equal(JSON.stringify(state.unreadableRanges), '[[5,7]]');
addUnreadableRange(1, 3);
assert.equal(JSON.stringify(state.peaks), '[2,4,8]');
assert.equal(JSON.stringify(state.unreadableRanges), '[[1,3],[5,7]]');
$('#clear-peaks').onclick(); assert.equal(state.peaks.length, 0);
$('#restore-saved').onclick();
assert.equal(JSON.stringify(state.peaks), '[2,4,8]');
assert.equal(JSON.stringify(state.unreadableRanges), '[[5,7]]');
state.clip.qualityFlags = [{code:'ambiguous_polarity',label:'Unclear peak direction',startSec:.2,endSec:.6,startIndex:2,endIndex:6,reason:'Direction is unclear',scope:'minute',signAgreement:.8,rawSpanUv:200,flippedSampleCount:0}];
state.clip.qualitySummary = {available:true,requiresReview:true,rawP01P99SpanUv:200,flippedSampleCount:0,sampleCount:11};
drawQualityFlags();
assert.equal(JSON.stringify(state.peaks), '[2,4,8]');
assert.equal(JSON.stringify(state.unreadableRanges), '[[5,7]]');
assert.match($('#quality-status').textContent, /1 screening flag/);
assert.match($('#quality-rule-text').textContent, /90% same-direction/);
`, sandbox);
sandbox.fetch = async (url, options) => {
  if (options?.method === 'POST') {
    const payload = JSON.parse(options.body);
    assert.deepEqual(payload.peaks, [2, 4, 8]);
    assert.deepEqual(payload.unreadableRanges, [[5, 7]]);
    return { ok: true, headers: { get: () => 'application/json' }, json: async () => ({ version: 1, reviewer: 'QA', unreadableRanges: [[5, 7]] }) };
  }
  return { ok: true, headers: { get: () => 'application/json' }, json: async () => ({ participants: [] }) };
};
vm.runInContext(`state.clips = [{id:'test_clip',status:'unreviewed'}]; state.index = 0; save('in_progress');`, sandbox).then(() => {
  vm.runInContext(`assert.equal(state.version, 1); assert.equal(state.dirty, false); assert.equal(JSON.stringify(state.peaks), '[2,4,8]');`, sandbox);
  console.log('Interval selection, double-click cancellation, saved-mark restore, and save payload: passed.');
}).catch((error) => { console.error(error); process.exitCode = 1; });
