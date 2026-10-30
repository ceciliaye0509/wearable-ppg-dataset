const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const elements = new Map(); let created = 0;
const ctx = new Proxy({}, {get: (target, key) => target[key] || (() => {})});
function element(selector) {
  if (!elements.has(selector)) elements.set(selector, {
    hidden: false, value: '', textContent: '', attributes: {}, listeners: {}, children: [], dataset: {}, style: {},
    width: 1100, height: 450, getContext: () => ctx, getBoundingClientRect: () => ({left:0,width:1100}),
    classList: {add(){},remove(){},toggle(){}}, addEventListener(name, fn){this.listeners[name] = fn},
    setAttribute(name, value){this.attributes[name] = value}, close(){},showModal(){},focus(){},
    replaceChildren(){this.children=[]},append(...children){this.children.push(...children)},
  });
  return elements.get(selector);
}
const sandbox = {assert,console, document:{querySelector:element,querySelectorAll:()=>[],createElement:()=>element(`new-${created++}`)},
  window:{location:{protocol:'file:'},setTimeout(){},setInterval(){},addEventListener(){},confirm:()=>true}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8'),sandbox);
vm.runInContext(`
(async () => {
  state.reviewer = 'Reviewer A';
  state.clip = {id:'full',annotationScope:'reviewer',windowDurationSec:300,reviewPageSeconds:5,
    t:Array.from({length:30001},(_,i)=>i/100),x:Array.from({length:30001},(_,i)=>Math.sin(i/20)),
    reviewerProgress:[],samplingGaps:[[100,101]],restorePeaks:[100,200],suggested:[100,200]};
  state.clips = [{id:'full',status:'unreviewed',startSec:0,endSec:300,fullWindow:true}]; state.index=0;
  state.peaks = [100,200]; state.savedPeaks=[100,200]; state.unreadableRanges=[[300,400]];
  state.savedRanges=[[300,400]]; state.reviewedPages=[0,1]; state.savedPages=[0];
  state.zoom=59; drawPlot();
  assert.equal(pageCount(),60); assert.equal(view()[4],60);
  assert.equal($('#page-grid').children.length,60);
  assert.match($('#page-position').textContent,/295–300 sec/);
  assert.match($('#review-scope-note').textContent,/Reviewer A/);
  state.zoom=-1; drawPlot(); assert.equal($('#confirm-page').disabled,true);
  markAt({clientX:100}); assert.equal(state.peaks.join(','),'100,200');
  $('#next-unreviewed').onclick(); assert.equal(state.zoom,2);
  let payload;
  api = async (url,options) => {
    if(url.endsWith('/annotation')) {
      payload=JSON.parse(options.body);
      return {version:1,reviewer:'Reviewer A',unreadableRanges:payload.unreadableRanges,reviewedPages:payload.reviewedPages,reviewerProgress:[]};
    }
    return {participants:[]};
  };
  assert.equal(await save('complete'),false);
  await $('#confirm-page').onclick();
  assert.equal(payload.status,'in_progress');
  assert.equal(payload.reviewedPages.join(','),'0,1,2'); assert.equal(state.zoom,3);
  assert.equal(state.savedPages.join(','),'0,1,2');
  assert.equal(state.unreadableRanges[0].join(','),'300,400');
  $('#reset-candidates').onclick(); assert.equal(state.reviewedPages.length,0);
  assert.equal(state.unreadableRanges[0].join(','),'300,400');
  $('#restore-saved').onclick(); assert.equal(state.reviewedPages.join(','),'0,1,2');
  $('#clear-peaks').onclick(); assert.equal(state.peaks.length,0); assert.equal(state.reviewedPages.length,0);
  $('#restore-saved').onclick(); assert.equal(state.peaks.join(','),'100,200');
  state.reviewedPages=Array.from({length:60},(_,i)=>i);
  assert.equal(await save('complete'),true); assert.equal(payload.status,'complete');
  state.clip={t:[0,5,10,15,20],x:[0,1,0,1,0]}; state.zoom=0; drawPlot();
  assert.equal($('#full-window-panel').hidden,true);
  assert.equal($('#complete-clip').textContent,'Complete segment →');
})()
`,sandbox).then(()=>console.log('Full-window 60-page coverage, save/resume, overview safety, restore and legacy UI: passed.'))
.catch(error=>{console.error(error);process.exitCode=1});
