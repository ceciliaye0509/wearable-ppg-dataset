const $ = (selector) => document.querySelector(selector);
const state = { participants: [], participant: null, clips: [], allClips: [], batches: [], batch: null, index: 0, clip: null, peaks: [], unreadableRanges: [], savedPeaks: [], savedRanges: [], rangeMode: false, rangeStart: null, rangePreview: null, pointerStart: null, version: 0, zoom: -1, dirty: false, busy: false, signedIn: false };
state.reviewedPages = []; state.savedPages = []; state.reviewer = "";
const fullWindow = () => state.clip?.annotationScope === "reviewer" && state.clip?.windowDurationSec === 300;
const pageCount = () => fullWindow() ? Math.ceil(state.clip.windowDurationSec / 5) : 0;
state.showRRProposals = false;
function drawRRReview() {
  const review = state.clip?.rrCorrectionReview;
  $("#rr-review-panel").hidden = !review;
  if (!review) return;
  $("#rr-review-description").textContent = review.instructions;
  const toggle = $("#rr-review-toggle");
  toggle.textContent = state.showRRProposals ? "Hide RR proposals" : "Show RR proposals (read-only)";
  toggle.setAttribute("aria-pressed", String(state.showRRProposals));
  toggle.onclick = () => { state.showRRProposals = !state.showRRProposals; drawPlot(); };
  const jumps = $("#rr-review-jumps"); jumps.replaceChildren();
  review.priorityPages.forEach((page) => {
    const button = document.createElement("button"); button.type = "button"; button.className = "icon-button";
    button.textContent = `Review ${page * 5}–${(page + 1) * 5} s`;
    button.onclick = () => { if (!state.busy) { setRangeMode(false); state.zoom = page; drawPlot(); } };
    jumps.append(button);
  });
}
function nextUnreviewedPage() { return Array.from({length: pageCount()}, (_, i) => i).find((i) => !state.reviewedPages.includes(i)); }
function drawFullWindow() {
  drawRRReview();
  const deferred = Boolean(state.clip?.deferredScreeningReview);
  $("#deferred-screening-panel").hidden = !deferred;
  if (deferred) {
    const revealed = Boolean(state.clip.screeningRevealedAt);
    $("#deferred-screening-status").textContent = revealed
      ? "Screening revealed. Your first-pass peaks, intervals and page confirmations were archived. You may edit and save a second pass."
      : "First pass: review all 60 pages, mark unreadable intervals and correct peaks. Screening results are hidden. Complete and save before revealing; your first pass will be preserved.";
    $("#reveal-screening").hidden = revealed;
    $("#reveal-screening").disabled = state.busy || state.dirty || !["complete", "unresolvable"].includes(state.clip.status);
  }
  $("#full-window-panel").hidden = !fullWindow();
  $("#full-window-actions").hidden = !fullWindow();
  if (!fullWindow()) { $("#page-position").textContent = ""; $("#complete-clip").textContent = "Complete segment →"; return; }
  const count = pageCount(), page = state.zoom;
  $("#review-scope-note").textContent = `Your review: ${state.reviewer}. Saved separately under this name; other reviewers' marks do not replace yours.`;
  $("#full-window-progress").textContent = `${state.reviewedPages.length}/${count} pages confirmed · ${state.reviewedPages.length * 5}/300 seconds checked. Confirm each page to save and advance.`;
  $("#page-position").textContent = page < 0 ? "300-second overview" : `${page * 5}–${Math.min(300, (page + 1) * 5)} sec · ${page + 1}/${count}`;
  $("#complete-clip").textContent = "Complete 5-minute window →";
  $("#confirm-page").disabled = state.busy || page < 0;
  const host = $("#page-grid"); host.replaceChildren();
  for (let i = 0; i < count; i++) {
    const button = document.createElement("button"), reviewed = state.reviewedPages.includes(i);
    button.textContent = `${i * 5}–${(i + 1) * 5}`; button.className = `page-button${reviewed ? " reviewed" : ""}${page === i ? " active" : ""}`;
    button.setAttribute("aria-label", `${i * 5} to ${(i + 1) * 5} seconds, ${reviewed ? "confirmed" : "not confirmed"}`);
    button.setAttribute("aria-current", String(page === i));
    button.onclick = () => { if (!state.busy) { setRangeMode(false); state.zoom = i; drawPlot(); } }; host.append(button);
  }
  const team = state.clip.reviewerProgress || [];
  $("#team-progress").textContent = team.length ? `Saved reviews: ${team.map((r) => `${r.reviewer}: ${statusLabel(r.status)} (${r.reviewedPages}/60 pages)`).join(" · ")}` : "No reviewer has saved this window yet.";
}

const copyRanges = (ranges) => ranges.map((range) => [...range]);
function markDirty(message = "Unsaved changes") { state.dirty = true; $("#save-message").textContent = message; }
function mergeRanges(ranges) {
  const result = [];
  ranges.map(([a, b]) => [Math.min(a, b), Math.max(a, b)]).filter(([a, b]) => a < b)
    .sort((a, b) => a[0] - b[0]).forEach(([a, b]) => {
      if (result.length && a <= result[result.length - 1][1] + 1) result[result.length - 1][1] = Math.max(result[result.length - 1][1], b);
      else result.push([a, b]);
    });
  return result;
}
function setRangeMode(enabled) {
  state.rangeMode = enabled; state.rangeStart = null; state.rangePreview = null; state.pointerStart = null;
  $("#save-unresolvable").setAttribute("aria-pressed", String(enabled));
  $("#range-mode-help").hidden = !enabled; $("#mark-whole-unreadable").hidden = !enabled;
  $("#plot").classList.toggle("selecting-range", enabled); drawPlot();
}
function drawUnreadableRanges() {
  const host = $("#unreadable-list"); host.replaceChildren();
  if (!state.unreadableRanges.length) { host.textContent = "No unreadable intervals marked."; return; }
  state.unreadableRanges.forEach(([start, end], index) => {
    const button = document.createElement("button"); button.className = "interval-chip";
    const label = `${state.clip.t[start].toFixed(2)}–${state.clip.t[end].toFixed(2)} sec`;
    button.textContent = `${label} · Remove ×`; button.setAttribute("aria-label", `Remove unreadable interval ${label}`);
    button.onclick = () => { if (state.busy) return; state.unreadableRanges.splice(index, 1); markDirty("Unreadable interval removed; peak marks kept"); drawUnreadableRanges(); drawPlot(); };
    host.append(button);
  });
}
function addUnreadableRange(start, end) {
  if (start === end) { toast("Select two different points for the interval."); return; }
  state.unreadableRanges = mergeRanges([...state.unreadableRanges, [start, end]]);
  state.rangeStart = null; state.rangePreview = null;
  markDirty("Unreadable interval added; peak marks kept. Save to keep this change."); drawUnreadableRanges(); drawPlot();
}
function drawQualityFlags() {
  const clip = state.clip, flags = clip.qualityFlags || [], summary = clip.qualitySummary || {}, rule = clip.qualityRule || {};
  const panel = $("#quality-panel"), host = $("#quality-list"); host.replaceChildren();
  panel.hidden = Boolean(clip.blindReview || clip.screeningWithheld);
  if (panel.hidden) return;
  panel.classList.toggle("has-flags", flags.length > 0);
  panel.classList.toggle("has-severe", flags.some((flag) => flag.code.startsWith("severe_")));
  const flipCount = summary.flippedSampleCount || 0;
  const correction = flipCount ? `Existing correction: ${flipCount === summary.sampleCount ? "this segment was flipped" : "part of this segment was flipped"}.` : "Existing correction: orientation retained.";
  $("#quality-status").textContent = !summary.available
    ? "Screening metadata is unavailable for this segment. No screening conclusion is shown."
    : `${flags.length ? `${flags.length} screening flag${flags.length > 1 ? "s" : ""} require attention.` : "No screening exclusions overlap this segment."} ${correction} Raw segment P01–P99 span: ${Number(summary.rawP01P99SpanUv).toLocaleString("en-US", { maximumFractionDigits: 2 })} µV.`;
  flags.forEach((flag) => {
    const row = document.createElement("div"); row.className = `quality-flag${flag.code.startsWith("severe_") ? " is-severe" : ""}`;
    const title = document.createElement("strong"); title.textContent = `${flag.startSec.toFixed(2)}–${flag.endSec.toFixed(2)} sec · ${flag.label}`;
    const reason = document.createElement("p"); reason.textContent = flag.reason;
    const metrics = [];
    if (Number.isFinite(flag.rawSpanUv)) metrics.push(`Raw ${flag.scope === "minute" ? "minute" : "segment"} P01–P99 span: ${flag.rawSpanUv.toLocaleString("en-US", { maximumFractionDigits: 2 })} µV`);
    if (Number.isFinite(flag.signAgreement)) metrics.push(`Same-direction votes: ${(flag.signAgreement * 100).toFixed(1)}%`);
    metrics.push(flag.flippedSampleCount ? "Existing correction includes flipped samples here" : "No automatic flip in this interval");
    const detail = document.createElement("p"); detail.textContent = metrics.join(" · ");
    row.append(title, reason, detail); host.append(row);
  });
  const score = rule.netDirectionScoreThreshold ?? .8, agreement = rule.equivalentSameSignFraction ?? (1 + score) / 2;
  $("#quality-rule-text").textContent = `Raw P01–P99 spans above ${(rule.rawAmplitudeThresholdUv || 5000).toLocaleString("en-US")} µV are excluded from minute-level polarity inference. The recorded run uses a net direction score of at least ${score.toFixed(2)}, equivalent to at least ${(agreement * 100).toFixed(0)}% same-direction votes, and at least ${rule.minimumClassifiedMinuteBlocks || 3} same-sign minute blocks. It may bridge ${rule.maximumBridgedUncertainMinutes ?? 1} uncertain minute at a time between matching signs; bridged minutes are flagged here, including their actual flip state. Low-signal, short-run, and unassessed intervals require review. A separate segment-level amplitude flag is also shown. These flags do not change peak marks, manual unreadable intervals, or saved review status.`;
  if (clip.deferredScreeningReview) $("#quality-rule-text").textContent = "Fixed source-run anchored 5-second raw ECG blocks: P99−P01 above 5,000 µV triggers an amplitude flag. Shading marks the full source block intersection, not a proven artifact duration. This alert does not automatically label the whole window unreadable. Waveform polarity uses the frozen 2026-10-29 combination with at least three consecutive strongly supported minutes and no bridging.";
}

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  const body = (response.headers.get("content-type") || "").includes("json") ? await response.json() : await response.text();
  if (!response.ok) { const error = new Error(body?.error || `Request failed (${response.status})`); error.status = response.status; throw error; }
  return body;
}
function toast(message) {
  const el = $("#toast"); el.textContent = message; el.classList.add("show");
  window.setTimeout(() => el.classList.remove("show"), 2400);
}
function loginDialog() { $("#login-dialog").showModal(); window.setTimeout(() => $("#reviewer-input").focus(), 30); }
function showStartupError(message) {
  $("#app-shell").hidden = true;
  $("#login-dialog").close();
  $("#startup-error-message").textContent = message;
  $("#startup-error").hidden = false;
}

async function boot() {
  if (!['http:', 'https:'].includes(window.location.protocol)) {
    showStartupError("This app must be opened through its review server. Do not open index.html or an exported CSV directly.");
    return;
  }
  try {
    const me = await api("/api/me");
    state.signedIn = true;
    state.reviewer = me.reviewer;
    $("#reviewer-label").textContent = `Reviewer: ${me.reviewer}`;
    $("#logout").hidden = false; $("#login-dialog").close();
    const data = await api("/api/participants"); state.participants = data.participants;
    state.batches = data.reviewBatches || [{id: "existing", label: "Existing review"}];
    if (!state.batch) state.batch = state.batches.find((batch) => batch.default)?.id || (state.batches.some((batch) => batch.id === "independent_blind_v1") ? "independent_blind_v1" : "existing");
    drawBatches(); drawParticipants();
    if (state.participant) await chooseParticipant(state.participant);
  } catch (error) {
    if (error.status === 401) loginDialog();
    else showStartupError(`Could not connect to the review service (${error.message}). Start the server, then open http://127.0.0.1:8765/.`);
  }
}
function drawBatches() {
  const select = $("#batch-filter"); select.replaceChildren();
  [...state.batches, {id: "all", label: "All review batches"}].forEach((batch) => {
    const option = document.createElement("option"); option.value = batch.id;
    option.textContent = batch.label; select.append(option);
  });
  select.value = state.batch || "existing";
}
function participantInBatch(item) {
  if (!state.batch || state.batch === "all") return item;
  return item.batches?.[state.batch] ? {...item, ...item.batches[state.batch]} : item;
}
function clipsInBatch(items) { return items.filter((clip) => !state.batch || state.batch === "all" || (clip.reviewBatch || "existing") === state.batch); }
function drawParticipants() {
  const host = $("#participant-list"); host.replaceChildren();
  const query = $("#participant-search").value.trim().toLowerCase();
  const batchRows = state.participants.map(participantInBatch);
  const rows = batchRows.filter((item) => item.participant.toLowerCase().includes(query));
  const total = batchRows.reduce((sum, item) => sum + item.total, 0), finished = batchRows.reduce((sum, item) => sum + item.finished, 0);
  $("#batch-summary").textContent = `${finished}/${total} segments complete in this batch` + (state.batches.find((b) => b.id === state.batch)?.annotationScope === "reviewer" ? " · Your progress" : "");
  rows.forEach((item) => {
    const button = document.createElement("button");
    button.className = `participant-card${item.participant === state.participant ? " active" : ""}`;
    button.dataset.state = item.label;
    const name = document.createElement("span"); name.className = "participant-name"; name.textContent = item.participant;
    const label = document.createElement("span"); label.className = "participant-state"; label.textContent = item.label;
    const progress = document.createElement("span"); progress.className = "participant-progress";
    progress.textContent = item.total ? `${item.finished}/${item.total} complete · ${item.saved} saved` : (item.reason || "No review segments available");
    if (item.qualityFlagged) progress.textContent += ` · ${item.qualityFlagged} screening-flagged`;
    const track = document.createElement("span"); track.className = "progress-track";
    const fill = document.createElement("i"); fill.style.width = `${item.total ? Math.round(item.finished * 100 / item.total) : 0}%`; track.append(fill);
    button.append(name, label, progress, track); button.disabled = !item.total; button.onclick = () => chooseParticipant(item.participant); host.append(button);
  });
  if (!rows.length) { const empty = document.createElement("p"); empty.className = "subtle"; empty.textContent = "No matching participants."; host.append(empty); }
}
async function chooseParticipant(participant) {
  if (state.busy) return;
  if (state.dirty && !window.confirm("This segment has unsaved changes. They will be lost if you leave. Continue?")) return;
  state.participant = participant; state.clip = null; state.dirty = false;
  $("#participant-title").textContent = participant; $("#empty-state").hidden = true; $("#review-panel").hidden = false; drawParticipants();
  const data = await api(`/api/participant/${encodeURIComponent(participant)}/clips`); state.allClips = data.clips; state.clips = clipsInBatch(data.clips);
  if (!state.clips.length) { $("#review-panel").hidden = true; $("#empty-state").hidden = false; return; }
  const open = state.clips.findIndex((item) => !["complete", "unresolvable"].includes(item.status)); state.index = open < 0 ? 0 : open;
  drawClipList(); await loadClip();
}
const statusLabel = (status) => ({ complete: "Complete", unresolvable: "Cannot reliably read", in_progress: "In progress", unreviewed: "Not reviewed" })[status] || status;
function drawClipList() {
  const host = $("#clip-list"); host.replaceChildren();
  state.clips.forEach((item, index) => {
    const button = document.createElement("button");
    button.className = `clip-button ${item.status || "unreviewed"}${index === state.index ? " active" : ""}`;
    const itemLabel = item.status !== "unreviewed" ? statusLabel(item.status) : (
      item.isPrevious ? (item.priorUnreadable ? "Prior: Cannot reliably read" : item.hasPriorMarks ? "Prior manual marks" : "Prior segment") : statusLabel(item.status)
    );
    const name = document.createElement("strong"); name.textContent = `${itemLabel} · ${index + 1}`;
    const source = (item.assistanceAvailable ? " · Algorithm-assisted" : item.blindReview ? ` · ${item.sourceLabel || "Blind review"}` : item.isPrevious ? " · Previous set" : " · New set") + (item.unreadableRangeCount ? ` · ${item.unreadableRangeCount} unreadable` : "") + (item.qualityWarningCount ? ` · ⚠ ${item.qualityWarningCount} screening flags` : "");
    if (item.qualityWarningLabels?.length) button.title = item.qualityWarningLabels.join("; ");
    const detail = document.createElement("span"); detail.textContent = `${item.startSec.toFixed(0)}–${item.endSec.toFixed(0)} sec${source}${item.fullWindow ? " · Your own review" : ""}`;
    button.append(name, detail); button.onclick = async () => {
      if (state.busy) return;
      if (state.dirty && !window.confirm("This segment has unsaved changes. They will be lost if you switch. Continue?")) return;
      state.index = index; await loadClip();
    }; host.append(button);
  });
}
async function loadClip() {
  const item = state.clips[state.index]; if (!item) return;
  state.clip = await api(`/api/clip/${encodeURIComponent(item.id)}`);
  state.showRRProposals = false;
  state.peaks = [...state.clip.peaks].sort((a, b) => a - b); state.version = state.clip.version;
  state.unreadableRanges = copyRanges(state.clip.unreadableRanges || []);
  state.savedPeaks = [...state.peaks]; state.savedRanges = copyRanges(state.unreadableRanges);
  state.reviewedPages = [...(state.clip.reviewedPages || [])]; state.savedPages = [...state.reviewedPages];
  setRangeMode(false); drawUnreadableRanges();
  drawQualityFlags();
  state.zoom = fullWindow() ? (nextUnreviewedPage() ?? 0) : -1; state.dirty = false;
  $("#confidence").value = state.clip.confidence; $("#notes").value = state.clip.notes || "";
  $("#save-message").textContent = state.clip.status === "unreviewed"
    ? (state.clip.isPrevious ? "Previous marks loaded · not yet saved in this workspace" : "Not yet saved")
    : `Last saved by ${state.clip.reviewer || "reviewer"} · ${statusLabel(state.clip.status)}`;
  const humanRecheck = state.clip.assistance?.method === "saved_human_marks_recheck_v1";
  const hint = humanRecheck ? "Previously saved human peaks and unreadable intervals are loaded for rechecking. Edits are saved as a separate segment; the original annotation is preserved."
    : fullWindow() ? "Review all 300 seconds, five seconds per page. Rodrigues suggestions are editable; saved human marks take precedence. Confirm every page before completing the window."
    : state.clip.assistance ? `${state.clip.sourceLabel}. ${state.clip.aidAvailable ? "Editable algorithm suggestions are provided; saved human marks take precedence." : "No usable algorithm suggestions were generated; mark peaks manually."} Review the full 20-second segment. This is not independent blind validation.` : state.clip.blindReview ? `${state.clip.sourceLabel}. Manual blind review: no algorithm markers or screening judgments are shown. Mark the full 20-second segment; use unreadable intervals where needed.` : state.clip.isPrevious
    ? `${state.clip.sourceLabel}: ${state.clip.hasPriorMarks ? `${state.clip.restorePeaks.length} previous manual marks loaded` : "no saved manual marks"}. Changes are not saved until you save this segment.${state.clip.priorUnreadable ? " This segment was previously marked as unreadable." : ""}`
    : state.clip.aidAvailable ? "Anonymous algorithm suggestions are available; please verify them." : "No usable algorithm suggestions were generated. Mark peaks manually or choose ‘Cannot reliably read’.";
  $("#clip-meta").textContent = `${state.clip.id} · ${state.clip.startSec.toFixed(0)}–${state.clip.endSec.toFixed(0)} sec (relative to sampled window) · ${hint}${state.clip.reviewInstructions ? " " + state.clip.reviewInstructions : ""}`;
  $("#reset-candidates").textContent = state.clip.isPrevious ? "Restore previous manual marks" : "Restore algorithm suggestions";
  $("#reset-candidates").hidden = Boolean(state.clip.blindReview);
  $("#blind-note").textContent = humanRecheck
    ? "Recheck the saved human marks. Keep clear QRS peaks, remove non-QRS marks, and mark only uncertain intervals as Cannot reliably read. Do not force any RR interval to fit a threshold. Original marks remain preserved."
    : state.clip.assistance
    ? "Algorithm-assisted peak review. Check each QRS and its exact position. Add, remove, clear, or restore suggestions as needed; saved peaks and unreadable intervals are never replaced automatically. Save your confirmed marks and confidence before completing the segment."
    : state.clip.blindReview
    ? "Independent manual marking: no algorithm candidate peaks are provided in this batch. Restore last saved marks restores only your saved annotations. Review the full segment before completing it."
    : state.clip.isPrevious
    ? "These are previously saved human marks from a prior assisted review. Edits in this workspace are saved separately and do not alter the source annotation file."
    : "New segments show anonymous algorithm-assisted candidates. Candidate identity is hidden, so this is not an independent blinded accuracy estimate.";
  const done = state.clips.filter((clip) => ["complete", "unresolvable"].includes(clip.status)).length;
  $("#participant-progress").textContent = `${done}/${state.clips.length} segments complete`;
  drawClipList(); drawPlot();
  if (state.clip.assistance && state.clip.suggested?.length) {
    try { await api(`/api/clip/${encodeURIComponent(state.clip.id)}/assistance-seen`, {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"}); }
    catch (_) { toast("Could not record assistance exposure. Your peak edits are still available; please save your review."); }
  }
}
function nearest(values, value) {
  let low = 0, high = values.length - 1;
  while (low < high) { const mid = (low + high) >> 1; if (values[mid] < value) low = mid + 1; else high = mid; }
  return low > 0 && Math.abs(values[low - 1] - value) <= Math.abs(values[low] - value) ? low - 1 : low;
}
function view() {
  const t = state.clip.t, last = t[t.length - 1], blocks = fullWindow() ? pageCount() : Math.max(1, Math.ceil((last - t[0]) / 5));
  if (state.zoom < 0) return [0, t.length - 1, t[0], last, blocks];
  state.zoom = Math.min(state.zoom, blocks - 1);
  const core = t[0] + state.zoom * 5, overlap = fullWindow() ? .25 : 0;
  const left = Math.max(t[0], core - overlap), right = Math.min(last, core + 5 + overlap);
  return [nearest(t, left), nearest(t, right), left, right, blocks];
}
function drawPlot() {
  if (!state.clip) return;
  const canvas = $("#plot"), ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height;
  const pad = { l: 62, r: 18, t: 24, b: 43 }, w = W - pad.l - pad.r, h = H - pad.t - pad.b;
  const [first, last, left, right, blocks] = view(), signal = state.clip.x.slice(first, last + 1);
  let min = Infinity, max = -Infinity;
  for (const value of signal) { min = Math.min(min, value); max = Math.max(max, value); }
  const margin = (max - min) * .08 || 1; min -= margin; max += margin;
  const X = (i) => pad.l + ((state.clip.t[i] - left) / Math.max(1e-9, right - left)) * w;
  const Y = (i) => pad.t + ((max - state.clip.x[i]) / (max - min)) * h;
  ctx.clearRect(0, 0, W, H); ctx.font = "11px system-ui, sans-serif"; ctx.lineWidth = 1; ctx.strokeStyle = "#e3e9e8"; ctx.fillStyle = "#718189";
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + h * i / 4, value = max - (max - min) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.fillText(value.toFixed(0), 7, y + 4);
    const x = pad.l + w * i / 4, sec = left + (right - left) * i / 4;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke(); ctx.fillText(`${sec.toFixed(1)} s`, x - 15, H - 20);
  }
  const shade = ([start, end], preview = false) => {
    if (state.clip.t[end] < left || state.clip.t[start] > right) return;
    const a = Math.max(pad.l, X(start)), b = Math.min(W - pad.r, X(end));
    ctx.fillStyle = preview ? "rgba(169,107,30,.10)" : "rgba(235,172,64,.22)"; ctx.fillRect(a, pad.t, Math.max(1, b - a), h);
    ctx.strokeStyle = "#a96b1e"; ctx.setLineDash(preview ? [5, 4] : []); ctx.strokeRect(a, pad.t, Math.max(1, b - a), h); ctx.setLineDash([]);
    if (b - a > 70) { ctx.fillStyle = "#79551f"; ctx.fillText(preview ? "Selecting…" : "Unreadable", a + 5, pad.t + 14); }
  };
  (state.clip.qualityFlags || []).forEach((flag) => {
    if (state.clip.t[flag.endIndex] < left || state.clip.t[flag.startIndex] > right) return;
    const a = Math.max(pad.l, X(flag.startIndex)), b = Math.min(W - pad.r, X(flag.endIndex));
    ctx.fillStyle = "rgba(119,76,165,.07)"; ctx.fillRect(a, pad.t, Math.max(1, b - a), h);
    ctx.fillStyle = "rgba(119,76,165,.65)"; ctx.fillRect(a, pad.t, Math.max(1, b - a), 5);
  });
  state.unreadableRanges.forEach((range) => shade(range));
  (state.clip.samplingGaps || []).forEach(([start, end]) => {
    if (state.clip.t[end] < left || state.clip.t[start] > right) return;
    ctx.fillStyle = "rgba(180,70,70,.17)"; const a = Math.max(pad.l, X(start)), b = Math.min(W - pad.r, X(end));
    ctx.fillRect(a, pad.t, Math.max(2, b - a), h); ctx.fillStyle = "#993333"; ctx.fillText("Missing samples", a + 3, pad.t + 30);
  });
  if (state.rangePreview) shade([...state.rangePreview].sort((a, b) => a - b), true);
  ctx.strokeStyle = "#30494d"; ctx.lineWidth = 1.15; ctx.beginPath();
  for (let i = first; i <= last; i++) (i === first || state.clip.t[i] - state.clip.t[i - 1] > .2501) ? ctx.moveTo(X(i), Y(i)) : ctx.lineTo(X(i), Y(i)); ctx.stroke();
  ctx.fillStyle = "#d54a46"; state.peaks.filter((i) => i >= first && i <= last).forEach((i) => { ctx.beginPath(); ctx.arc(X(i), Y(i), 4.2, 0, 2 * Math.PI); ctx.fill(); });
  if (state.showRRProposals && state.clip.rrCorrectionReview) {
    ctx.strokeStyle = "#7b45a6"; ctx.lineWidth = 1.7;
    state.clip.rrCorrectionReview.proposalTimesSec.filter((sec) => sec >= left && sec <= right).forEach((sec) => {
      const index = nearest(state.clip.t, sec), x = pad.l + (sec - left) / Math.max(1e-9, right - left) * w, y = Y(index);
      ctx.beginPath(); ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5); ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5); ctx.stroke();
    });
  }
  $("#peak-count").textContent = fullWindow() ? `${state.peaks.filter((i) => i >= first && i <= last).length} visible / ${state.peaks.length} total markers` : `${state.peaks.length} markers`;
  $("#zoom-prev").disabled = state.zoom < 0 || state.zoom === 0; $("#zoom-next").disabled = state.zoom >= blocks - 1;
  drawFullWindow();
}
function markAt(event) {
  if (!state.clip || state.busy || state.rangeMode) return;
  if (fullWindow() && state.zoom < 0) { toast("Open a five-second page to edit peaks precisely."); return; }
  const canvas = $("#plot"), rect = canvas.getBoundingClientRect(), padLeft = 62, padRight = 18, w = canvas.width - padLeft - padRight;
  const x = (event.clientX - rect.left) / rect.width * canvas.width, [first, last, left, right] = view();
  const sec = left + (Math.max(padLeft, Math.min(canvas.width - padRight, x)) - padLeft) / w * (right - left), index = nearest(state.clip.t, sec);
  const tolerance = Math.max(.055, (right - left) / w * 7); let closest = -1, distance = Infinity;
  state.peaks.forEach((peak, at) => { if (peak >= first && peak <= last && Math.abs(state.clip.t[peak] - sec) < distance) { closest = at; distance = Math.abs(state.clip.t[peak] - sec); } });
  if (closest >= 0 && distance <= tolerance) state.peaks.splice(closest, 1); else state.peaks.push(index);
  state.peaks = [...new Set(state.peaks)].sort((a, b) => a - b); state.dirty = true;
  $("#save-message").textContent = "Unsaved changes"; drawPlot();
}
async function save(status, next = false) {
  if (!state.clip || state.busy) return false;
  if (state.rangeStart !== null || state.pointerStart) { toast("Finish the interval or double-click ‘Cannot reliably read’ to cancel selection."); return false; }
  if (fullWindow() && status === "complete" && state.reviewedPages.length !== pageCount()) { toast("Confirm all 60 five-second pages before completing this window."); return false; }
  state.busy = true; document.querySelectorAll(".save-actions button, #notes, #confidence, #mark-whole-unreadable").forEach((button) => { button.disabled = true; });
  try {
    const result = await api(`/api/clip/${encodeURIComponent(state.clip.id)}/annotation`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ peaks: state.peaks, unreadableRanges: state.unreadableRanges, reviewedPages: state.reviewedPages, version: state.version, status, confidence: $("#confidence").value, notes: $("#notes").value }),
    });
    state.version = result.version; state.dirty = false;
    state.unreadableRanges = copyRanges(result.unreadableRanges); state.savedPeaks = [...state.peaks]; state.savedRanges = copyRanges(state.unreadableRanges);
    state.reviewedPages = [...(result.reviewedPages || [])]; state.savedPages = [...state.reviewedPages];
    state.clip.reviewerProgress = result.reviewerProgress || [];
    state.clips[state.index].unreadableRangeCount = state.unreadableRanges.length;
    drawUnreadableRanges(); drawPlot();
    state.clips[state.index].status = status; state.clips[state.index].reviewer = result.reviewer;
    state.clip.status = status;
    $("#save-message").textContent = `Saved: ${statusLabel(status)}`;
    $("#participant-progress").textContent = `${state.clips.filter((item) => ["complete", "unresolvable"].includes(item.status)).length}/${state.clips.length} segments complete`;
    state.participants = (await api("/api/participants")).participants; drawParticipants(); drawClipList();
    if (next) {
      const index = state.clips.findIndex((item, i) => i > state.index && !["complete", "unresolvable"].includes(item.status));
      if (index >= 0) { state.index = index; await loadClip(); } else toast("All segments for this participant have been reviewed.");
    }
    return true;
  } catch (error) {
    $("#save-message").textContent = error.message;
    toast(error.status === 409 ? "This segment was updated on another device. Reload before editing." : error.message);
    return false;
  } finally { state.busy = false; document.querySelectorAll(".save-actions button, #notes, #confidence, #mark-whole-unreadable").forEach((button) => { button.disabled = false; }); drawFullWindow(); }
}

function sampleAtPointer(event) {
  const canvas = $("#plot"), rect = canvas.getBoundingClientRect(), [first, last, left, right] = view();
  const x = (event.clientX - rect.left) / rect.width * canvas.width;
  const sec = left + (Math.max(62, Math.min(canvas.width - 18, x)) - 62) / (canvas.width - 80) * (right - left);
  return nearest(state.clip.t, sec);
}
$("#plot").addEventListener("pointerdown", (event) => {
  if (!state.rangeMode || state.busy || !state.clip || (event.button !== undefined && event.button !== 0)) return;
  state.pointerStart = { index: sampleAtPointer(event), x: event.clientX };
  $("#plot").setPointerCapture(event.pointerId); event.preventDefault();
});
$("#plot").addEventListener("pointermove", (event) => {
  if (!state.rangeMode || state.busy || !state.clip) return;
  const index = sampleAtPointer(event);
  if (state.pointerStart && Math.abs(event.clientX - state.pointerStart.x) > 3) state.rangePreview = [state.pointerStart.index, index];
  else if (state.rangeStart !== null) state.rangePreview = [state.rangeStart, index];
  drawPlot();
});
$("#plot").addEventListener("pointerup", (event) => {
  if (!state.rangeMode || !state.pointerStart || state.busy) return;
  const start = state.pointerStart, index = sampleAtPointer(event); state.pointerStart = null;
  if (Math.abs(event.clientX - start.x) > 3) addUnreadableRange(start.index, index);
  else if (state.rangeStart !== null) addUnreadableRange(state.rangeStart, index);
  else { state.rangeStart = index; state.rangePreview = [index, index]; toast("Click the end of the unreadable interval."); drawPlot(); }
});
$("#plot").addEventListener("pointercancel", () => { state.pointerStart = null; state.rangePreview = null; drawPlot(); });

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault(); $("#login-error").textContent = ""; $(".login-submit").disabled = true;
  try {
    await api("/api/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer: $("#reviewer-input").value, accessCode: $("#access-code").value }) });
    $("#access-code").value = ""; await boot();
  } catch (error) { $("#login-error").textContent = error.message; }
  finally { $(".login-submit").disabled = false; }
});
$("#logout").addEventListener("click", async () => {
  if (state.busy || (state.dirty && !window.confirm("You have unsaved changes. Sign out without saving?"))) return;
  await api("/api/logout", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  state.signedIn = false;
  state.participant = null; state.clip = null; state.participants = []; state.clips = []; state.dirty = false;
  state.unreadableRanges = []; state.savedPeaks = []; state.savedRanges = []; setRangeMode(false);
  $("#reviewer-label").textContent = ""; $("#logout").hidden = true; $("#review-panel").hidden = true; $("#empty-state").hidden = false; $("#participant-list").replaceChildren(); loginDialog();
});
$("#participant-search").addEventListener("input", drawParticipants);
$("#batch-filter").addEventListener("change", async () => {
  if (state.busy || (state.dirty && !window.confirm("This segment has unsaved changes. Continue without saving?"))) { $("#batch-filter").value = state.batch; return; }
  state.batch = $("#batch-filter").value; state.dirty = false; state.clip = null;
  drawParticipants();
  const participant = state.participants.find((item) => item.participant === state.participant && participantInBatch(item).total) || state.participants.find((item) => participantInBatch(item).total);
  if (participant) await chooseParticipant(participant.participant);
});
$("#notes").addEventListener("input", () => { state.dirty = true; $("#save-message").textContent = "Unsaved changes"; });
$("#confidence").addEventListener("change", () => { state.dirty = true; $("#save-message").textContent = "Unsaved changes"; });
$("#plot").addEventListener("click", markAt);
$("#zoom-out").onclick = () => { state.zoom = -1; drawPlot(); };
$("#zoom-prev").onclick = () => { if (state.busy) return; setRangeMode(false); state.zoom = Math.max(0, state.zoom - 1); drawPlot(); };
$("#zoom-next").onclick = () => { if (state.busy) return; setRangeMode(false); state.zoom = Math.min(view()[4] - 1, state.zoom + 1); drawPlot(); };
$("#next-unreviewed").onclick = () => { if (!fullWindow() || state.busy) return; setRangeMode(false); state.zoom = nextUnreviewedPage() ?? 0; drawPlot(); };
$("#reveal-screening").onclick = async () => {
  if (state.busy || state.dirty || !state.clip?.deferredScreeningReview) return;
  try {
    await api(`/api/clip/${encodeURIComponent(state.clip.id)}/reveal-screening`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({version: state.version})});
    await loadClip();
  } catch (error) { toast(error.message); }
};
$("#confirm-page").onclick = async () => {
  if (!fullWindow() || state.busy || state.zoom < 0) return;
  if (state.rangeStart !== null || state.pointerStart) { toast("Finish or cancel your interval first."); return; }
  const page = state.zoom;
  state.reviewedPages = [...new Set([...state.reviewedPages, page])].sort((a, b) => a - b);
  markDirty("Page confirmed; saving your review…");
  if (await save("in_progress")) { setRangeMode(false); state.zoom = nextUnreviewedPage() ?? page; drawPlot(); }
};
$("#restore-saved").onclick = () => { if (state.busy || !state.clip) return; state.peaks = [...state.savedPeaks]; state.unreadableRanges = copyRanges(state.savedRanges); state.reviewedPages = [...state.savedPages]; setRangeMode(false); markDirty("Last saved marks and intervals restored"); drawUnreadableRanges(); drawPlot(); };
$("#reset-candidates").onclick = () => { if (state.busy || !state.clip) return; state.peaks = [...(state.clip.restorePeaks || state.clip.suggested)]; if (fullWindow()) state.reviewedPages = []; markDirty(state.clip.isPrevious ? "Previous manual marks restored; not yet saved" : "Algorithm suggestions restored; not yet saved"); drawPlot(); };
$("#clear-peaks").onclick = () => { if (state.busy || !state.clip) return; state.peaks = []; if (fullWindow()) state.reviewedPages = []; markDirty("Peak marks cleared; use Restore last saved marks to undo"); drawPlot(); };
$("#save-progress").onclick = () => save("in_progress");
$("#save-unresolvable").onclick = () => { if (!state.busy && state.clip) setRangeMode(true); };
$("#save-unresolvable").ondblclick = (event) => { event.preventDefault(); if (!state.busy) { setRangeMode(false); toast("Unreadable interval selection cancelled; peak marks kept."); } };
$("#mark-whole-unreadable").onclick = () => { if (!state.busy && state.clip) addUnreadableRange(0, state.clip.t.length - 1); };
$("#complete-clip").onclick = () => save("complete", !state.clip?.deferredScreeningReview);
window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
window.addEventListener("resize", drawPlot);
window.setInterval(async () => {
  if (!state.signedIn || document.hidden) return;
  try {
    state.participants = (await api("/api/participants")).participants;
    drawParticipants();
    if (state.participant) {
      const updated = (await api(`/api/participant/${encodeURIComponent(state.participant)}/clips`)).clips;
      state.allClips = updated; state.clips = clipsInBatch(updated);
      drawClipList();
      $("#participant-progress").textContent = `${state.clips.filter((item) => ["complete", "unresolvable"].includes(item.status)).length}/${state.clips.length} segments complete`;
      if (!state.dirty && state.clip) {
        const current = state.clips[state.index];
        if (!current) return;
        const newState = await api(`/api/clip/${encodeURIComponent(current.id)}`);
        state.clip.reviewerProgress = newState.reviewerProgress || [];
        drawFullWindow();
        if (newState.version !== state.version) await loadClip();
      }
    }
  } catch (error) { if (error.status === 401) state.signedIn = false; }
}, 15000);
boot();
