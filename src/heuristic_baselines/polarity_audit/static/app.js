const $ = (s) => document.querySelector(s);
const state = { tasks: [], participants: [], participant: null, task: null, version: 0, dirty: false, busy: false, signedIn: false, zoom: null, context: false,
  peaks: [], peakOrigin: "unreviewed", assistanceShown: false, showPeaks: true, plotTransform: null };
async function api(path, options = {}) {
  const r = await fetch(path, { credentials: "same-origin", ...options });
  const body = await r.json();
  if (!r.ok) { const error = new Error(body.error || `Request failed (${r.status})`); error.status = r.status; throw error; }
  return body;
}
function dirty() { state.dirty = true; $("#message").textContent = "Unsaved changes"; }
async function boot() {
  if (!["http:", "https:"].includes(location.protocol)) {
    $("#startup-error").hidden = false;
    $("#startup-error").textContent = "Start serve_audit.py and open the HTTP workspace address; do not open this file directly.";
    return;
  }
  try {
    const me = await api("/api/me"); state.signedIn = true; $("#reviewer").textContent = `Reviewer: ${me.reviewer}`; $("#login").close();
    const data = await api("/api/tasks"); state.tasks = data.tasks; state.participants = data.participants; drawParticipants();
  } catch (error) {
    if (error.status === 401) $("#login").showModal();
    else { $("#startup-error").hidden = false; $("#startup-error").textContent = error.message; }
  }
}
function drawParticipants() {
  $("#participants").replaceChildren();
  state.participants.forEach((pid) => {
    const rows = state.tasks.filter((t) => t.participant === pid), done = rows.filter((t) => t.rating?.status === "complete").length;
    const button = document.createElement("button"); button.className = `participant${state.participant === pid ? " active" : ""}`;
    const name = document.createElement("strong"); name.textContent = pid;
    const detail = document.createElement("small"); detail.textContent = rows.length ? `${done}/${rows.length} complete` : "No direction alerts";
    button.append(name, detail); button.disabled = !rows.length; button.onclick = () => chooseParticipant(pid); $("#participants").append(button);
  });
}
function okayToLeave() { return !state.dirty || confirm("Discard unsaved ratings for this minute?"); }
async function chooseParticipant(pid) {
  if (state.busy || !okayToLeave()) return;
  state.participant = pid; state.dirty = false;
  const rows = state.tasks.filter((t) => t.participant === pid);
  const first = rows.find((t) => t.rating?.status !== "complete") || rows[0];
  await load(first.id);
}
function drawTasks() {
  const rows = state.tasks.filter((t) => t.participant === state.participant);
  $("#participant-title").textContent = state.participant;
  $("#progress").textContent = `${rows.filter((t) => t.rating?.status === "complete").length}/${rows.length} audit minutes complete`;
  $("#tasks").replaceChildren();
  rows.forEach((row, i) => {
    const button = document.createElement("button");
    button.className = `${row.rating?.status || ""}${row.id === state.task?.id ? " active" : ""}`;
    button.textContent = `${row.rating?.status === "complete" ? "Complete" : row.rating ? "In progress" : "Not reviewed"} · ${i + 1}`;
    button.onclick = async () => { if (!state.busy && okayToLeave()) await load(row.id); };
    $("#tasks").append(button);
  }); drawParticipants();
}
async function load(id) {
  state.busy = true;
  try {
    state.task = await api(`/api/task/${encodeURIComponent(id)}`);
    const r = state.task.rating || {};
    state.version = r.version || 0; state.zoom = null; state.context = false; state.dirty = false;
    state.showPeaks = true; state.assistanceShown = true;
    state.peaks = [...(r.peak_indices ?? state.task.assistance.peak_indices)];
    state.peakOrigin = r.peak_origin || (r.peak_indices ? "manual" : "algorithm");
    $("#peaks-reviewed").checked = r.peaks_reviewed === true;
    $("#assistance-notice").textContent = state.task.assistance.notice;
    $("#qrs").value = r.qrs_readability || ""; $("#determinability").value = r.direction_determinability || "";
    $("#direction").value = r.direction || "undetermined"; $("#direction").disabled = r.direction_determinability !== "yes";
    $("#confidence").value = r.confidence || "medium"; $("#notes").value = r.notes || "";
    $("#message").textContent = r.version ? `Saved by ${r.reviewer} · ${r.status === "complete" ? "Complete" : "In progress"} · save to retain any new assistance/peak changes` : "Algorithm marks are not saved or verified until you save and check them";
    if (r.version && !Object.hasOwn(r, "peak_indices")) { state.dirty = true; $("#message").textContent = "Previous ratings preserved. New algorithm marks are unsaved and unverified."; }
    $("#task-label").textContent = `${state.task.id} · rate only 0–${state.task.focusEndSec.toFixed(1)} sec; surrounding context is optional`;
    $("#focus-label").textContent = `Focus interval: 0–${state.task.focusEndSec.toFixed(1)} seconds`;
    $("#notes").placeholder = `If partially readable, note uncertain intervals within 0–${state.task.focusEndSec.toFixed(1)} sec.`;
    $("#empty").hidden = true; $("#panel").hidden = false; drawTasks(); drawPlot();
  } catch (error) { $("#message").textContent = error.message; }
  finally { state.busy = false; }
}
function drawPlot() {
  if (!state.task) return;
  const { t, x, focusEndSec } = state.task;
  const left = state.zoom !== null ? state.zoom : state.context ? t[0] : 0;
  const right = state.zoom !== null ? Math.min(focusEndSec, left + 5) : state.context ? t[t.length - 1] : focusEndSec;
  let low = Infinity, high = -Infinity;
  x.forEach((v, i) => { if (t[i] >= left && t[i] <= right) { low = Math.min(low, v); high = Math.max(high, v); } });
  const margin = (high - low) * .08 || 1; low -= margin; high += margin;
  const cv = $("#plot"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  const X = (v) => 65 + (v - left) / Math.max(.001, right - left) * (W - 85);
  const Y = (v) => 18 + (high - v) / (high - low) * (H - 58);
  state.plotTransform = { left, right, X, Y, W, H };
  ctx.clearRect(0, 0, W, H); ctx.font = "11px system-ui"; ctx.lineWidth = 1;
  if (state.context && state.zoom === null) { ctx.fillStyle = "#e8f5f1"; ctx.fillRect(X(0), 18, X(focusEndSec) - X(0), H - 58); }
  for (let i = 0; i <= 4; i++) {
    const y = 18 + (H - 58) * i / 4, sec = left + (right - left) * i / 4;
    ctx.strokeStyle = "#e2ebed"; ctx.fillStyle = "#71838a";
    ctx.beginPath(); ctx.moveTo(65, y); ctx.lineTo(W - 20, y); ctx.stroke();
    ctx.fillText((high - (high - low) * i / 4).toFixed(0), 5, y + 4);
    ctx.beginPath(); ctx.moveTo(X(sec), 18); ctx.lineTo(X(sec), H - 40); ctx.stroke(); ctx.fillText(`${sec.toFixed(1)} s`, X(sec) - 15, H - 17);
  }
  ctx.strokeStyle = "#244d56"; ctx.lineWidth = 1; ctx.beginPath(); let first = true;
  x.forEach((v, i) => { if (t[i] >= left && t[i] <= right) { if (first) ctx.moveTo(X(t[i]), Y(v)); else ctx.lineTo(X(t[i]), Y(v)); first = false; } }); ctx.stroke();
  if (state.showPeaks) {
    ctx.fillStyle = state.peakOrigin === "algorithm" && !$("#peaks-reviewed").checked ? "#c8790e" : "#d64949";
    state.peaks.forEach((i) => { if (t[i] >= left && t[i] <= right) { ctx.beginPath(); ctx.arc(X(t[i]), Y(x[i]), 4, 0, 2 * Math.PI); ctx.fill(); } });
  }
  $("#peak-count").textContent = `${state.peaks.length} marks · ${$("#peaks-reviewed").checked ? "Reviewer checked" : "Not yet verified"}`;
  $("#toggle-peaks").textContent = state.showPeaks ? "Hide peak marks" : "Show peak marks";
  $("#restore-peaks").disabled = state.busy || !Object.hasOwn(state.task.rating || {}, "peak_indices");
  $("#prev").disabled = state.zoom === null || state.zoom === 0;
  $("#next").disabled = state.zoom !== null && state.zoom + 5 >= focusEndSec;
}
async function save(status) {
  if (!state.task || state.busy) return;
  const body = { qrs_readability: $("#qrs").value, direction_determinability: $("#determinability").value,
    direction: $("#direction").value, confidence: $("#confidence").value, notes: $("#notes").value, status, version: state.version,
    peak_indices: [...state.peaks], peak_origin: state.peakOrigin, peaks_reviewed: $("#peaks-reviewed").checked, assistance_shown: state.assistanceShown };
  state.busy = true; ["#save", "#complete", "#qrs", "#determinability", "#direction", "#confidence", "#notes", "#peaks-reviewed"].forEach((s) => { $(s).disabled = true; });
  try {
    const r = await api(`/api/task/${encodeURIComponent(state.task.id)}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    state.version = r.version; state.dirty = false; state.task.rating = r;
    state.tasks.find((t) => t.id === state.task.id).rating = r; $("#message").textContent = "Ratings saved"; drawTasks();
    if (status === "complete") {
      const next = state.tasks.find((t) => t.participant === state.participant && t.id !== state.task.id && t.rating?.status !== "complete");
      if (next) await load(next.id); else $("#message").textContent = "All sampled minutes for this participant are complete";
    }
  } catch (error) { $("#message").textContent = error.message; }
  finally { state.busy = false; ["#save", "#complete", "#qrs", "#determinability", "#confidence", "#notes", "#peaks-reviewed"].forEach((s) => { $(s).disabled = false; }); $("#direction").disabled = $("#determinability").value !== "yes"; }
}
$("#login-form").onsubmit = async (event) => {
  event.preventDefault(); $("#login-error").textContent = "";
  try { await api("/api/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer: $("#name").value, accessCode: $("#code").value }) }); $("#code").value = ""; await boot(); }
  catch (error) { $("#login-error").textContent = error.message; }
};
$("#determinability").onchange = () => { $("#direction").disabled = $("#determinability").value !== "yes"; if ($("#direction").disabled) $("#direction").value = "undetermined"; dirty(); };
["#qrs", "#direction", "#confidence"].forEach((s) => { $(s).onchange = dirty; }); $("#notes").oninput = dirty;
$("#minute").onclick = () => { state.zoom = null; state.context = false; drawPlot(); };
$("#context").onclick = () => { state.zoom = null; state.context = true; drawPlot(); };
$("#prev").onclick = () => { state.zoom = Math.max(0, (state.zoom || 0) - 5); drawPlot(); };
$("#next").onclick = () => { state.zoom = state.zoom === null ? 0 : Math.min(Math.floor((state.task.focusEndSec - .001) / 5) * 5, state.zoom + 5); drawPlot(); };
$("#save").onclick = () => save("in_progress"); $("#complete").onclick = () => save("complete");
function changePeaks(peaks, origin, reviewed = false) {
  if (state.busy) return;
  state.peaks = [...peaks]; state.peakOrigin = origin; $("#peaks-reviewed").checked = reviewed; dirty(); drawPlot();
}
$("#use-suggestions").onclick = () => {
  if (!state.task || state.busy || !confirm("Replace current peak marks with algorithm suggestions? Saved marks remain recoverable.")) return;
  state.showPeaks = true; state.assistanceShown = true;
  changePeaks(state.task.assistance.peak_indices, "algorithm");
};
$("#restore-peaks").onclick = () => {
  const r = state.task?.rating;
  if (!r || !Object.hasOwn(r, "peak_indices") || state.busy || !confirm("Restore last saved peak marks? Your current unsaved peak edits will be replaced; ratings stay unchanged.")) return;
  changePeaks(r.peak_indices, r.peak_origin || "manual", r.peaks_reviewed === true);
};
$("#toggle-peaks").onclick = () => { if (!state.task || state.busy) return; state.showPeaks = !state.showPeaks; if (state.showPeaks) state.assistanceShown = true; drawPlot(); };
$("#peaks-reviewed").onchange = () => { if (state.busy) { $("#peaks-reviewed").checked = state.task?.rating?.peaks_reviewed === true; return; } dirty(); drawPlot(); };
$("#plot").onclick = (event) => {
  if (!state.task || state.busy) return;
  if (state.zoom === null) { $("#message").textContent = "Open a 5-second view before editing peak marks for precise placement."; return; }
  if (!state.showPeaks) { $("#message").textContent = "Show peak marks before editing."; return; }
  const box = event.currentTarget.getBoundingClientRect(), p = state.plotTransform;
  const px = (event.clientX - box.left) * p.W / box.width, py = (event.clientY - box.top) * p.H / box.height;
  if (px < 65 || px > p.W - 20 || py < 18 || py > p.H - 40) return;
  const { t, x, focusStartSec, focusEndSec } = state.task;
  const hit = state.peaks.findIndex((i) => t[i] >= p.left && t[i] <= p.right && Math.hypot(p.X(t[i]) - px, p.Y(x[i]) - py) < 12);
  const peaks = [...state.peaks];
  if (hit >= 0) peaks.splice(hit, 1);
  else {
    const second = p.left + (px - 65) / (p.W - 85) * (p.right - p.left);
    let lo = 0, hi = t.length - 1;
    while (lo < hi) { const mid = Math.floor((lo + hi) / 2); if (t[mid] < second) lo = mid + 1; else hi = mid; }
    const index = lo > 0 && Math.abs(t[lo - 1] - second) < Math.abs(t[lo] - second) ? lo - 1 : lo;
    if (t[index] < focusStartSec || t[index] >= focusEndSec || peaks.includes(index)) return;
    peaks.push(index); peaks.sort((a, b) => a - b);
  }
  changePeaks(peaks, state.peakOrigin === "manual" || (!state.peaks.length && !state.task.assistance.peak_indices.length) ? "manual" : "mixed");
};
window.addEventListener("beforeunload", (e) => { if (state.dirty) { e.preventDefault(); e.returnValue = ""; } });
window.setInterval(async () => {
  if (!state.signedIn || state.busy || document.hidden) return;
  try { state.tasks = (await api("/api/tasks")).tasks; if (state.task) drawTasks(); else drawParticipants(); }
  catch (error) { if (error.status === 401) state.signedIn = false; }
}, 15000);
window.addEventListener("resize", drawPlot); boot();
