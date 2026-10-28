const state = {
  page: "overview",
  graph: null,
  documents: [],
  datasets: [],
  models: { nlp: [], nlp_by_task: { ner: [], relation: [] }, nlp_checkpoints: [], gnn: [], nli: [] },
  experiments: [],
  job: null,
  selectedEntity: null,
  chat: [],
};

const pageNames = {
  overview: "Overview",
  graph: "Knowledge graph",
  documents: "Documents",
  verification: "Claim verification",
  assistant: "GraphRAG assistant",
  evidence: "Evidence explorer",
  "model-lab": "Model Lab",
  datasets: "Dataset manager",
  experiments: "Experiments",
  settings: "Settings",
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));
const formatNumber = (value) => new Intl.NumberFormat().format(Number(value || 0));
const shortPath = (value) => String(value || "").split(/[\\/]/).slice(-2).join("/");
const isSyntheticCheckpoint = (path) => (state.models.nlp_checkpoints || []).some((item) => item.path === path && item.synthetic_data);
const checkpointLabel = (path) => `${shortPath(path)}${isSyntheticCheckpoint(path) ? " · synthetic smoke model" : ""}`;
const dateLabel = (value) => {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "—" : date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
};

async function api(path, options = {}) {
  const isFormData = typeof FormData !== "undefined" && options.body instanceof FormData;
  const response = await fetch(path, {
    ...options,
    headers: { ...(options.body && !isFormData ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) },
  });
  let payload = {};
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("error", error);
  node.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("show"), 3300);
}

function badge(value) {
  const key = String(value || "unknown").toLowerCase();
  const tone = ["completed", "approved", "human_reviewed", "supported"].includes(key) ? "green"
    : ["queued", "running", "review_required", "not enough information"].includes(key) ? "amber"
      : ["failed", "rejected", "refuted"].includes(key) ? "red"
        : ["predicted", "candidate", "extracted_candidate"].includes(key) ? "violet" : "neutral";
  return `<span class="badge ${tone}">${escapeHTML(String(value || "unknown").replaceAll("_", " "))}</span>`;
}

function setPage(page) {
  state.page = pageNames[page] ? page : "overview";
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.page === state.page));
  $("#page-title").textContent = pageNames[state.page];
  $("#sidebar").classList.remove("open");
  renderPage();
}

async function refreshState() {
  const results = await Promise.allSettled([
    api("/health"),
    api("/graph"),
    api("/documents"),
    api("/datasets"),
    api("/models"),
    api("/experiments"),
  ]);
  if (results[0].status === "fulfilled") {
    $("#api-status").textContent = "API connected";
    $("#sidebar-status").textContent = `${results[0].value.graph} · local API`;
    $(".environment-pill .status-dot").classList.add("ok");
  } else {
    $("#api-status").textContent = "API unavailable";
    $("#sidebar-status").textContent = "Could not reach Flask API";
    $(".environment-pill .status-dot").classList.add("error");
  }
  if (results[1].status === "fulfilled") state.graph = results[1].value;
  if (results[2].status === "fulfilled") state.documents = results[2].value.documents || [];
  if (results[3].status === "fulfilled") state.datasets = results[3].value.datasets || [];
  if (results[4].status === "fulfilled") state.models = results[4].value;
  if (results[5].status === "fulfilled") state.experiments = results[5].value.experiments || [];
  renderPage();
}

function pageHeading(title, description, actions = "") {
  return `<div class="page-heading"><div><div class="eyebrow">KNOWGRAPH AI</div><h1>${escapeHTML(title)}</h1><p class="subtitle">${escapeHTML(description)}</p></div><div class="heading-actions">${actions}</div></div>`;
}

function emptyState(icon, title, detail) {
  return `<div class="empty-state"><div class="empty-icon">${escapeHTML(icon)}</div><strong>${escapeHTML(title)}</strong><div>${escapeHTML(detail)}</div></div>`;
}

function metricCard(label, value, caption, icon) {
  return `<article class="card"><div class="metric-label">${escapeHTML(label)}<span class="metric-icon">${icon}</span></div><div class="metric-value">${escapeHTML(formatNumber(value))}</div><div class="metric-caption">${escapeHTML(caption)}</div></article>`;
}

async function renderPage() {
  const root = $("#page-content");
  try {
    switch (state.page) {
      case "overview": root.innerHTML = await renderOverview(); break;
      case "graph": root.innerHTML = renderGraph(); bindGraph(); break;
      case "documents": root.innerHTML = renderDocuments(); break;
      case "verification": root.innerHTML = renderVerification(); break;
      case "assistant": root.innerHTML = renderAssistant(); break;
      case "evidence": root.innerHTML = renderEvidence(); break;
      case "model-lab": root.innerHTML = renderModelLab(); bindModelLab(); break;
      case "datasets": root.innerHTML = renderDatasets(); break;
      case "experiments": root.innerHTML = renderExperiments(); break;
      case "settings": root.innerHTML = renderSettings(); break;
      default: root.innerHTML = emptyState("?", "Page not found", "Choose another workspace section.");
    }
  } catch (error) {
    root.innerHTML = `<div class="error-state">${escapeHTML(error.message)}</div>`;
  }
}

async function renderOverview() {
  const result = await api("/overview");
  const experiments = result.recent_experiments || [];
  const recent = experiments.length ? `<div class="list">${experiments.map((item) => `
    <div class="list-row"><div class="list-main"><strong>${escapeHTML(item.task || item.job_id || "Experiment")}</strong><small>${escapeHTML(dateLabel(item.created_at))} · ${escapeHTML(item.result?.task || "training run")}</small></div>${badge(item.status)}</div>`).join("")}</div>`
    : emptyState("◷", "No completed experiments yet", "Run a model from Model Lab to record real training results.");
  const activity = (result.activity || []).length ? `<div class="list">${result.activity.map((item) => `
    <div class="list-row"><div class="list-main"><strong>${escapeHTML(item.document_id)}</strong><small>${escapeHTML(dateLabel(item.updated_at))}</small></div>${badge(item.status)}</div>`).join("")}</div>`
    : emptyState("▤", "No document activity", "Documents you ingest will appear here.");
  return `${pageHeading("A clear view of your knowledge", "Explore the graph, inspect source evidence, and run experiments from one local research workspace.",
    `<button class="button" data-navigate="graph">Explore graph</button><button class="button primary" data-navigate="model-lab">Open Model Lab</button>`)}
    <div class="grid stats">
      ${metricCard("Graph entities", result.graph.entities, "Current stored entity records", "⌘")}
      ${metricCard("Relationships", result.graph.relationships, "Observed, reviewed, and other stored edges", "↔")}
      ${metricCard("Documents", result.documents, "Persisted local source documents", "▤")}
      ${metricCard("Experiments", state.experiments.length, "Persisted model and job records", "◷")}
    </div>
    <div class="grid two">
      <article class="card"><div class="card-header"><div><h2>Recent experiments</h2><p>Results recorded by actual training jobs.</p></div><button class="button small" data-navigate="experiments">View all</button></div>${recent}</article>
      <article class="card"><div class="card-header"><div><h2>Recent document activity</h2><p>Ingestion and human-review status.</p></div><button class="button small" data-navigate="documents">Open library</button></div>${activity}</article>
    </div>`;
}

function relationRows(relations) {
  if (!relations.length) return emptyState("⌘", "No graph relationships", "Add verified or observed facts to see them in the graph.");
  return `<div class="table-wrap"><table><thead><tr><th>Subject</th><th>Predicate</th><th>Object</th><th>Type</th><th>Source</th></tr></thead><tbody>${relations.map((relation) => `
    <tr><td class="cell-primary">${escapeHTML(relation.subject)}</td><td>${escapeHTML(relation.predicate)}</td><td class="cell-primary">${escapeHTML(relation.object)}</td><td>${badge(relation.relation_type)}</td><td>${escapeHTML(relation.provenance?.document_id || relation.source || "—")}</td></tr>`).join("")}</tbody></table></div>`;
}

function renderGraph() {
  const relations = state.graph?.relations || [];
  return `${pageHeading("Knowledge graph", "Browse stored graph data. Dashed styling is reserved for predicted relationships; only backend facts are shown here.",
    `<button class="button" data-action="reload">↻ Refresh graph</button>`)}
    <div class="card">
      <div class="card-header"><div><h2>Graph explorer</h2><p>${formatNumber(state.graph?.entities?.length || 0)} entities · ${formatNumber(relations.length)} relationships</p></div>
        <div class="inline-form"><input class="search-input" id="graph-search" placeholder="Search entities or predicates"><select id="graph-filter" class="search-input" aria-label="Filter relation type"><option value="">All relation types</option>${[...new Set(relations.map((row) => row.relation_type))].map((name) => `<option>${escapeHTML(name)}</option>`).join("")}</select></div>
      </div>
      <div class="graph-layout"><div class="graph-canvas"><div class="graph-toolbar"><button class="button" data-action="graph-reset">Reset view</button><span class="badge neutral">Observed facts</span></div><svg id="graph-svg" viewBox="0 0 800 480" role="img" aria-label="Interactive knowledge graph"></svg></div>
      <aside class="card" id="entity-details">${emptyState("◉", "Select an entity", "Click a graph node to inspect its relationships and source provenance.")}</aside></div>
      <div class="section-gap"><div class="section-label">Stored relationships</div><div id="graph-table">${relationRows(relations)}</div></div>
    </div>`;
}

function graphFilteredRelations() {
  const query = ($("#graph-search")?.value || "").trim().toLowerCase();
  const type = $("#graph-filter")?.value || "";
  return (state.graph?.relations || []).filter((relation) =>
    (!type || relation.relation_type === type)
    && (!query || `${relation.subject} ${relation.predicate} ${relation.object}`.toLowerCase().includes(query)));
}

function bindGraph() {
  const search = $("#graph-search");
  const filter = $("#graph-filter");
  if (!search) return;
  const update = () => {
    const relations = graphFilteredRelations();
    $("#graph-table").innerHTML = relationRows(relations);
    drawGraph(relations);
  };
  search.addEventListener("input", update);
  filter.addEventListener("change", update);
  drawGraph(graphFilteredRelations());
  const svg = $("#graph-svg");
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const view = state.graphViewport || { x: 0, y: 0, width: 800, height: 480 };
    const rect = svg.getBoundingClientRect();
    const px = (event.clientX - rect.left) / rect.width;
    const py = (event.clientY - rect.top) / rect.height;
    const scale = event.deltaY < 0 ? 0.88 : 1.14;
    const width = Math.min(1600, Math.max(260, view.width * scale));
    const height = width * 0.6;
    state.graphViewport = {
      x: view.x + px * (view.width - width),
      y: view.y + py * (view.height - height),
      width,
      height,
    };
    svg.setAttribute("viewBox", `${state.graphViewport.x} ${state.graphViewport.y} ${width} ${height}`);
  }, { passive: false });
  svg.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".graph-node")) return;
    state.graphPan = {
      pointer: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      view: state.graphViewport || { x: 0, y: 0, width: 800, height: 480 },
    };
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (!state.graphPan || state.graphPan.pointer !== event.pointerId) return;
    const rect = svg.getBoundingClientRect();
    const dx = (event.clientX - state.graphPan.x) * state.graphPan.view.width / rect.width;
    const dy = (event.clientY - state.graphPan.y) * state.graphPan.view.height / rect.height;
    state.graphViewport = { ...state.graphPan.view, x: state.graphPan.view.x - dx, y: state.graphPan.view.y - dy };
    const view = state.graphViewport;
    svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.width} ${view.height}`);
  });
  svg.addEventListener("pointerup", () => { state.graphPan = null; });
  svg.addEventListener("pointercancel", () => { state.graphPan = null; });
}

function drawGraph(relations) {
  const svg = $("#graph-svg");
  if (!svg) return;
  const entities = [...new Set(relations.flatMap((relation) => [relation.subject, relation.object]))];
  const view = state.graphViewport || { x: 0, y: 0, width: 800, height: 480 };
  svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.width} ${view.height}`);
  if (!entities.length) {
    svg.innerHTML = `<text x="400" y="235" text-anchor="middle" fill="#8991a1" font-size="12">No matching graph relationships</text>`;
    return;
  }
  const positions = new Map(entities.map((entity, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index / entities.length);
    return [entity, { x: 400 + Math.cos(angle) * Math.min(290, 90 + entities.length * 9), y: 240 + Math.sin(angle) * Math.min(170, 55 + entities.length * 5) }];
  }));
  const edges = relations.map((relation) => {
    const start = positions.get(relation.subject);
    const end = positions.get(relation.object);
    const predicted = ["predicted", "candidate", "model_prediction"].includes(relation.relation_type);
    return `<g><line class="graph-edge ${predicted ? "predicted" : ""}" x1="${start.x}" y1="${start.y}" x2="${end.x}" y2="${end.y}"/><text class="graph-edge-label" x="${(start.x + end.x) / 2}" y="${(start.y + end.y) / 2 - 4}" text-anchor="middle">${escapeHTML(relation.predicate)}</text></g>`;
  }).join("");
  const nodes = entities.map((entity) => {
    const point = positions.get(entity);
    const label = entity.length > 18 ? `${entity.slice(0, 16)}…` : entity;
    return `<g class="graph-node ${state.selectedEntity === entity ? "selected" : ""}" data-entity="${escapeHTML(entity)}" transform="translate(${point.x} ${point.y})"><circle r="18"/><text text-anchor="middle" y="33">${escapeHTML(label)}</text></g>`;
  }).join("");
  svg.innerHTML = `<g id="graph-viewport">${edges}${nodes}</g>`;
  $$(".graph-node", svg).forEach((node) => node.addEventListener("click", () => selectEntity(node.dataset.entity)));
}

function selectEntity(entity) {
  state.selectedEntity = entity;
  const relations = (state.graph?.relations || []).filter((item) => item.subject === entity || item.object === entity);
  const node = (state.graph?.entities || []).find((item) => item.entity_id === entity);
  $("#entity-details").innerHTML = `<div class="card-header"><div><div class="eyebrow">ENTITY</div><h2>${escapeHTML(node?.label || entity)}</h2><p>${escapeHTML(entity)}</p></div></div>
    <div class="detail-row"><small>Attributes</small><span class="mono">${escapeHTML(JSON.stringify(node?.attributes || {}))}</span></div>
    <div class="section-label section-gap">Relationships (${relations.length})</div>
    ${relations.length ? relations.map((row) => `<div class="detail-row"><small>${escapeHTML(row.subject === entity ? "Outgoing" : "Incoming")} · ${escapeHTML(row.predicate)}</small><span>${escapeHTML(row.subject)} → ${escapeHTML(row.object)}</span><span>${badge(row.relation_type)}</span><span class="muted">${escapeHTML(row.provenance?.document_id || row.source || "No source recorded")}</span>${row.provenance?.snippet ? `<div class="evidence-block section-gap">${escapeHTML(row.provenance.snippet)}</div>` : ""}</div>`).join("") : emptyState("↔", "No relationships", "This entity has no stored edges.")}`;
  drawGraph(graphFilteredRelations());
}

function renderDocuments() {
  const docs = state.documents;
  const nerCheckpoints = state.models.nlp_by_task?.ner || (state.models.nlp || []).filter((path) => path.includes("/ner-"));
  const relationCheckpoints = state.models.nlp_by_task?.relation || (state.models.nlp || []).filter((path) => path.includes("/relation-"));
  const setupNotice = !nerCheckpoints.length || !relationCheckpoints.length
    ? `<div class="notice warning section-gap">Extraction needs both compatible saved checkpoints. Missing: ${!nerCheckpoints.length ? "NER" : ""}${!nerCheckpoints.length && !relationCheckpoints.length ? " and " : ""}${!relationCheckpoints.length ? "relation classifier" : ""}. <button class="button small section-gap" data-action="setup-demo-models">Train lightweight demo checkpoints</button> <button class="button small section-gap" data-navigate="model-lab">Open Model Lab</button><div class="field-help section-gap">Demo setup performs real training on the bundled synthetic sample dataset and marks the checkpoints as smoke-test models. It is for trying the workflow, not reliable extraction from research documents.</div></div>`
    : "";
  const syntheticWarning = nerCheckpoints.some(isSyntheticCheckpoint) || relationCheckpoints.some(isSyntheticCheckpoint)
    ? `<div class="notice warning section-gap">One or more available NLP checkpoints were trained on synthetic smoke data. Predictions are only for workflow demonstration and must not be treated as reliable extraction. Inspect every source snippet before approval.</div>`
    : "";
  const list = docs.length ? `<div class="list">${docs.map((doc) => `
    <div class="list-row"><div class="list-main"><strong>${escapeHTML(doc.filename || doc.document_id)}</strong><small>${escapeHTML(doc.document_id)} · ${escapeHTML(dateLabel(doc.updated_at))} · ${escapeHTML(doc.preview || "")}</small>
      ${doc.extraction ? `<small>Extraction: ${formatNumber(doc.extraction.entity_count)} entities, ${formatNumber(doc.extraction.candidate_count)} candidates${doc.extraction.synthetic_models ? " · synthetic smoke models" : ""}.</small>${doc.extraction.entity_count === 0 ? `<small class="muted">No entities were detected. The lightweight local demo NER conservatively ignores tokens absent from its training set; train/select a transformer checkpoint for general vocabulary.</small>` : doc.extraction.candidate_count === 0 ? `<small class="muted">No relation candidates were returned. A missing prediction is not evidence that the document contains no facts.</small>` : ""}` : ""}
      ${(doc.candidate_relations || []).map((candidate, index) => `<div class="evidence-block section-gap"><strong>${escapeHTML(candidate.subject)} — ${escapeHTML(candidate.label)} → ${escapeHTML(candidate.object)}</strong><div>${escapeHTML(candidate.evidence_snippet)}</div><div class="form-actions">${badge(candidate.status)}${candidate.synthetic_model ? badge("synthetic smoke") : ""}${candidate.status === "extracted_candidate" ? `<button class="button small" data-action="review-candidate" data-id="${escapeHTML(doc.document_id)}" data-index="${index}" data-decision="approve">Approve</button><button class="button small danger" data-action="review-candidate" data-id="${escapeHTML(doc.document_id)}" data-index="${index}" data-decision="reject">Reject</button>` : ""}</div></div>`).join("")}
    </div><div>${badge(doc.status)}${["stored", "extracted"].includes(doc.status) && !(doc.candidate_relations || []).length ? `<button class="button small section-gap" data-action="extract-document" data-id="${escapeHTML(doc.document_id)}">${doc.status === "stored" ? "Extract" : "Retry extraction"}</button>` : ""}</div></div>`).join("")}</div>`
    : emptyState("▤", "Your document library is empty", "Upload a text, Markdown, PDF, or DOCX source document to begin.");
  return `${pageHeading("Documents", "Upload a source file, extract entity/relation candidates with saved local checkpoints, then review facts before adding them to the graph.",
    `<button class="button" data-action="reload">↻ Refresh</button>`)}
    ${setupNotice}
    ${syntheticWarning}
    <div class="split-panel"><article class="card"><div class="card-header"><div><h2>Add a source document</h2><p>Text is stored locally with its source ID. Extraction remains an explicit step.</p></div></div>
      <form id="document-form" class="form-grid">
        <div class="field"><label for="document-id">Document ID (optional)</label><input id="document-id" name="document_id" placeholder="auto-generated if empty"></div>
        <div class="field full"><label for="document-file">Upload a file</label><input id="document-file" name="file" type="file" accept=".txt,.md,.pdf,.docx,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"><div class="field-help">Supported: UTF-8 TXT/Markdown, text-based PDF, DOCX. Max file size is configured by the server (16 MB by default); scanned PDFs need OCR first.</div></div>
        <div class="field full"><label for="document-text">Or paste document text</label><textarea class="large" id="document-text" name="text" placeholder="Paste text from a source you are authorized to process…"></textarea></div>
        <div class="field"><label for="ner-checkpoint">NER checkpoint</label><select id="ner-checkpoint"><option value="">Select a saved NER checkpoint</option>${nerCheckpoints.map((path) => `<option value="${escapeHTML(path)}">${escapeHTML(checkpointLabel(path))}</option>`).join("")}</select></div>
        <div class="field"><label for="relation-checkpoint">Relation checkpoint</label><select id="relation-checkpoint"><option value="">Select a saved relation checkpoint</option>${relationCheckpoints.map((path) => `<option value="${escapeHTML(path)}">${escapeHTML(checkpointLabel(path))}</option>`).join("")}</select></div>
        <div class="field-help full">Extraction is best-effort, not exhaustive; predicted relations remain candidates until human review.</div>
        <div class="form-actions field full"><button class="button" type="submit">Store pasted text</button><button class="button" type="button" data-action="upload-document">Upload file</button><button class="button primary" type="button" data-action="upload-and-extract">Upload + extract</button></div>
      </form></article>
      <article class="card"><div class="card-header"><div><h2>Ingestion notes</h2><p>Provenance is retained across extraction and review.</p></div></div>
        <div class="notice">The NER/relation checkpoints produce best-effort candidates, not a guarantee of every fact. Review source snippets before approving facts into the graph. Approved relationships create graph entities and edges with source provenance.</div>
        <div class="list section-gap"><div class="list-row"><div class="list-main"><strong>Stored</strong><small>Document text and identifier persisted locally.</small></div>${badge("stored")}</div><div class="list-row"><div class="list-main"><strong>Review required</strong><small>Extracted entity pairs await a human decision.</small></div>${badge("review_required")}</div><div class="list-row"><div class="list-main"><strong>Human reviewed</strong><small>Approved facts enter the graph with source provenance.</small></div>${badge("human_reviewed")}</div></div>
      </article></div>
      <article class="card section-gap"><div class="card-header"><div><h2>Document library</h2><p>${docs.length} locally stored documents</p></div></div>${list}</article>`;
}

function renderVerification() {
  return `${pageHeading("Claim verification", "Inspect graph evidence for a claim. Missing facts are not treated as contradictions.",
    `<span class="badge neutral">Evidence-backed</span>`)}
    <div class="split-panel"><article class="card"><div class="card-header"><div><h2>Check a claim</h2><p>Verification uses the current graph and returns supporting provenance when available.</p></div></div>
      <form id="verify-form"><div class="field"><label for="claim-input">Claim</label><textarea class="large" id="claim-input" required placeholder="Enter a factual claim to check against stored graph evidence."></textarea></div><div class="form-actions"><button class="button primary" type="submit">Verify claim</button></div></form><div id="verification-result" class="section-gap"></div>
    </article><aside class="card"><div class="card-header"><div><h2>Verdict guide</h2><p>Optional fine-tuned NLI checkpoint for evidence/claim classification.</p></div></div>
    <div class="field"><label for="verification-checkpoint">NLI checkpoint</label><select id="verification-checkpoint"><option value="">Evidence-only baseline</option>${(state.models.nli || []).map((path) => `<option value="${escapeHTML(path)}">${escapeHTML(shortPath(path))}</option>`).join("")}</select><div class="form-actions"><button class="button small" data-action="load-nli">Load checkpoint</button><span class="field-help" id="verification-model-status">Current verifier configuration is active.</span></div></div>
    <div class="list section-gap">
      <div class="list-row"><div class="list-main"><strong>SUPPORTED</strong><small>Matching evidence was found in the graph.</small></div>${badge("SUPPORTED")}</div>
      <div class="list-row"><div class="list-main"><strong>REFUTED</strong><small>Evidence directly conflicts with the claim.</small></div>${badge("REFUTED")}</div>
      <div class="list-row"><div class="list-main"><strong>NOT ENOUGH INFORMATION</strong><small>No relevant evidence is available. This is not a negative finding.</small></div>${badge("NOT ENOUGH INFORMATION")}</div>
    </div><div class="notice warning section-gap">Evidence strength and model confidence are distinct. A prediction alone is not source evidence.</div></aside></div>`;
}

function renderAssistant() {
  const thread = state.chat.length ? state.chat.map((message) => `<div class="chat-message ${message.role}">${escapeHTML(message.text)}${message.evidence?.length ? `<div class="chat-sources">${message.evidence.map((item) => `<span class="source-chip">${escapeHTML(item.source || "Source not recorded")}</span>`).join("")}</div>${message.evidence.map((item) => `<div class="evidence-block section-gap">${escapeHTML(item.fact)}<br><span class="muted">${escapeHTML((item.graph_path || []).join(" → "))}</span></div>`).join("")}` : ""}</div>`).join("")
    : `<div class="chat-message assistant">Ask a question about facts stored in the graph. Answers summarize retrieved records and cite their stored source identifiers where available.</div>`;
  return `${pageHeading("GraphRAG assistant", "A retrieval interface over graph facts and provenance. It does not generate independent citations or claim proof.",
    `<span class="badge violet">Local graph retrieval</span>`)}
    <article class="card"><div class="chat-thread" id="chat-thread">${thread}</div><form id="rag-form" class="inline-form"><input class="search-input" id="rag-query" placeholder="Ask a question about the current graph…" required><button class="button primary" type="submit">Retrieve evidence</button></form><div class="field-help section-gap">Predicted edges are not used as supporting evidence by default.</div></article>`;
}

function renderEvidence() {
  const relations = state.graph?.relations || [];
  return `${pageHeading("Evidence explorer", "Search stored relationships and inspect their snippets, source IDs, confidence, and review state.",
    `<button class="button" data-navigate="graph">Open graph</button>`)}
    <article class="card"><div class="card-header"><div><h2>Source evidence</h2><p>Evidence is shown only when preserved in graph provenance.</p></div><input class="search-input" id="evidence-search" placeholder="Search fact, source, or snippet" style="max-width:260px"></div>
      <div id="evidence-list">${renderEvidenceRows(relations)}</div>
    </article>`;
}

function renderEvidenceRows(relations) {
  const query = ($("#evidence-search")?.value || "").toLowerCase();
  const rows = relations.filter((item) => {
    const text = `${item.subject} ${item.predicate} ${item.object} ${item.source} ${item.provenance?.document_id || ""} ${item.provenance?.snippet || ""}`.toLowerCase();
    return text.includes(query);
  });
  if (!rows.length) return emptyState("⌕", "No matching evidence", "Try another term or ingest a source document.");
  return `<div class="list">${rows.map((item) => `<div class="list-row"><div class="list-main"><strong>${escapeHTML(item.subject)} — ${escapeHTML(item.predicate)} → ${escapeHTML(item.object)}</strong><small>Source: ${escapeHTML(item.provenance?.document_id || item.source || "Not recorded")} · confidence ${escapeHTML(item.confidence ?? "—")}</small>${item.provenance?.snippet ? `<div class="evidence-block section-gap">${escapeHTML(item.provenance.snippet)}</div>` : `<div class="field-help section-gap">No evidence snippet was stored for this graph fact.</div>`}<div class="section-gap">${badge(item.relation_type)}</div></div></div>`).join("")}</div>`;
}

function checkpointOptions(task) {
  return (state.models[task] || []).map((path) => `<option value="${escapeHTML(path)}">${escapeHTML(shortPath(path))}</option>`).join("");
}

function renderModelLab() {
  const nlpDatasets = state.datasets.filter((item) => item.status === "available");
  const defaultDataset = nlpDatasets.find((item) => item.name === "sample_nlp_dataset.json")?.path;
  const datasetOptions = nlpDatasets.map((item) => `<option value="${escapeHTML(item.path)}" ${item.path === defaultDataset ? "selected" : ""}>${escapeHTML(item.dataset || item.name)} · ${formatNumber(item.examples)} examples${item.synthetic ? " · synthetic smoke data" : ""}</option>`).join("");
  const currentJob = state.job;
  return `${pageHeading("Model Lab", "Run actual local training jobs, inspect their status and losses, and use saved checkpoints for inference.",
    `<button class="button" data-action="reload">↻ Refresh registry</button><button class="button" data-action="setup-demo-models">Train demo extraction models</button>`)}
    <div class="split-panel"><article class="card"><div class="card-header"><div><h2>Training workspace</h2><p>Jobs run in the background. Results are reported only after their underlying task completes.</p></div></div>
      <form id="training-form" class="form-grid">
        <div class="field"><label for="training-task">Training task</label><select id="training-task"><option value="ner">NER</option><option value="relation">Relation classification</option><option value="nli">Claim NLI fine-tuning</option><option value="gnn">R-GCN link prediction</option></select></div>
        <div class="field" id="dataset-field"><label for="training-dataset">Dataset file</label><select id="training-dataset"><option value="">Select local prepared dataset</option>${datasetOptions}</select><input id="dataset-path" placeholder="or enter a local dataset path"></div>
        <div class="field"><label for="epochs">Epochs</label><input id="epochs" type="number" min="1" value="3"></div>
        <div class="field"><label for="batch-size">Batch size</label><input id="batch-size" type="number" min="1" value="8"></div>
        <div class="field"><label for="learning-rate">Learning rate</label><input id="learning-rate" type="number" min="0.0000001" step="any" value="0.00002"></div>
        <div class="field"><label for="seed">Random seed</label><input id="seed" type="number" value="42"></div>
        <div class="field"><label for="device">Device</label><select id="device"><option value="auto">Auto</option><option value="cpu">CPU</option><option value="cuda">CUDA (if available)</option></select></div>
        <div class="field hidden" id="ner-architecture-field"><label for="ner-architecture">NER architecture</label><select id="ner-architecture"><option value="local-context" selected>Lightweight local PyTorch model</option><option value="pretrained-transformer">Pretrained transformer fine-tuning</option></select><div class="field-help">Trains offline from your labeled dataset; this is not pretrained and small datasets may generalize poorly.</div></div>
        <div class="field hidden" id="pretrained-field"><label for="pretrained-model">Pretrained model</label><input id="pretrained-model" value="distilbert/distilbert-base-uncased"></div>
        <div class="form-actions field full"><button class="button primary" type="submit">Start training job</button></div>
      </form>
      <div class="notice warning section-gap">Lightweight local NER trains its own weights offline from labeled examples. Pretrained transformer NER/NLI need accessible pretrained files. Demo models train on synthetic smoke data and are not reliable document extractors or benchmark results.</div>
    </article>
    <article class="card"><div class="card-header"><div><h2>Job monitor</h2><p>Live status, epoch logs, and errors from the Flask job store.</p></div></div>
      <div id="job-panel">${currentJob ? renderJob(currentJob) : emptyState("◷", "No active job", "Start a training run to see its real status and logs.")}</div>
    </article></div>
    <article class="card section-gap"><div class="card-header"><div><h2>Checkpoint registry</h2><p>Saved local model paths ready for supported inference tasks.</p></div></div>${renderCheckpointRegistry()}</article>
    <article class="card section-gap"><div class="card-header"><div><h2>Inference testing</h2><p>Run a saved checkpoint and inspect its raw output. Predictions remain candidates, not graph facts.</p></div></div>
      <form id="inference-form" class="form-grid">
        <div class="field"><label for="inference-task">Task</label><select id="inference-task"><option value="ner">NER</option><option value="relation">Relation classification</option><option value="gnn">GNN candidate ranking</option></select></div>
        <div class="field"><label for="inference-checkpoint">Checkpoint</label><select id="inference-checkpoint"></select></div>
        <div class="field full" id="inference-text-field"><label for="inference-text">Text</label><textarea id="inference-text" placeholder="Text to analyze"></textarea></div>
        <div class="field"><label for="inference-subject">Subject span (start,end)</label><input id="inference-subject" placeholder="0,3"></div>
        <div class="field"><label for="inference-object">Object span (start,end)</label><input id="inference-object" placeholder="13,17"></div>
        <div class="field"><label for="inference-entity">GNN source entity</label><input id="inference-entity" placeholder="Existing entity ID"></div>
        <div class="field"><label for="inference-predicate">GNN relation</label><input id="inference-predicate" placeholder="Existing predicate"></div>
        <div class="form-actions field full"><button class="button primary" type="submit">Run inference</button></div>
      </form><pre id="inference-result" class="job-log section-gap hidden"></pre>
    </article>`;
}

function renderJob(job) {
  const logs = job.logs || [];
  const result = job.result || {};
  const lossValues = (result.train_loss || []).map(Number).filter(Number.isFinite);
  return `<div class="list-row"><div class="list-main"><strong>${escapeHTML(job.task)}</strong><small>Job ${escapeHTML(job.job_id)} · ${escapeHTML(dateLabel(job.created_at))}</small></div>${badge(job.status)}</div>
    ${job.error ? `<div class="error-state section-gap">${escapeHTML(job.error)}</div>` : ""}
    ${logs.length ? `<div class="job-log section-gap">${logs.map(escapeHTML).join("\n")}</div>` : ""}
    ${job.status === "queued" ? `<button class="button small danger section-gap" data-action="cancel-job" data-id="${escapeHTML(job.job_id)}">Cancel queued job</button>` : ""}
    ${job.status === "failed" ? `<button class="button small section-gap" data-action="retry-job" data-id="${escapeHTML(job.job_id)}">Retry failed run</button>` : ""}
    ${result.checkpoint ? `<div class="detail-row section-gap"><small>Checkpoint</small><span class="mono">${escapeHTML(result.checkpoint)}</span></div>` : ""}
    ${result.evaluation || result.metrics ? `<div class="metric-list section-gap">${Object.entries(result.evaluation || result.metrics).filter(([, value]) => typeof value === "number").map(([key, value]) => `<div class="metric-cell"><small>${escapeHTML(key.replaceAll("_", " "))}</small><strong>${Number(value).toFixed(4)}</strong></div>`).join("")}</div>` : ""}
    ${result.baseline ? `<div class="notice section-gap">Baseline comparison (${escapeHTML(result.baseline.name || "train-split baseline")}): ${escapeHTML(JSON.stringify(result.baseline.evaluation || result.baseline))}</div>` : ""}
    ${lossValues.length ? `<div class="section-gap"><div class="section-label">Recorded training loss by epoch</div>${lossValues.map((value, index) => `<div class="detail-row"><small>Epoch ${index + 1}</small><span>${value.toFixed(6)}</span></div>`).join("")}</div>` : ""}
    ${result.warning ? `<div class="notice warning section-gap">${escapeHTML(result.warning)}</div>` : ""}`;
}

function renderCheckpointRegistry() {
  const groups = [
    ["NLP", state.models.nlp || []],
    ["GNN", state.models.gnn || []],
    ["Claim NLI", state.models.nli || []],
  ];
  const rows = groups.flatMap(([task, paths]) => paths.map((path) => ({ task, path, synthetic: isSyntheticCheckpoint(path) })));
  if (!rows.length) return emptyState("⚗", "No model checkpoints", "Completed training jobs will appear here; existing checkpoints are preserved.");
  return `<div class="table-wrap"><table><thead><tr><th>Family</th><th>Checkpoint path</th><th>Action</th></tr></thead><tbody>${rows.map((item) => `<tr><td>${badge(item.task)}${item.synthetic ? ` ${badge("synthetic smoke")}` : ""}</td><td class="mono">${escapeHTML(item.path)}</td><td><button class="button small" data-action="copy-path" data-path="${escapeHTML(item.path)}">Copy path</button></td></tr>`).join("")}</tbody></table></div>`;
}

function bindModelLab() {
  const task = $("#training-task");
  if (!task) return;
  const updateTaskFields = () => {
    const graph = task.value === "gnn";
    $("#dataset-field").classList.toggle("hidden", graph);
    $("#ner-architecture-field").classList.toggle("hidden", task.value !== "ner");
    const usesPretrained = task.value === "nli" || (task.value === "ner" && $("#ner-architecture").value === "pretrained-transformer");
    $("#pretrained-field").classList.toggle("hidden", !usesPretrained);
    $("#pretrained-model").value = task.value === "nli" ? "cross-encoder/nli-distilroberta-base" : "distilbert/distilbert-base-uncased";
    $("#learning-rate").value = task.value === "gnn" || (task.value === "ner" && !usesPretrained)
      ? "0.01"
      : task.value === "relation" ? "0.001" : "0.00002";
  };
  task.addEventListener("change", updateTaskFields);
  $("#ner-architecture").addEventListener("change", updateTaskFields);
  updateTaskFields();
  const inferenceTask = $("#inference-task");
  const updateInference = () => {
    const family = inferenceTask.value === "gnn" ? "gnn" : "nlp";
    const checkpoints = inferenceTask.value === "ner" || inferenceTask.value === "relation"
      ? state.models.nlp_by_task?.[inferenceTask.value] || []
      : state.models[family] || [];
    $("#inference-checkpoint").innerHTML = `<option value="">Select a saved checkpoint</option>${checkpoints.map((path) => `<option value="${escapeHTML(path)}">${escapeHTML(checkpointLabel(path))}</option>`).join("")}`;
    $("#inference-text-field").classList.toggle("hidden", inferenceTask.value === "gnn");
    $$("#inference-subject, #inference-object").forEach((input) => input.closest(".field").classList.toggle("hidden", inferenceTask.value !== "relation"));
    $$("#inference-entity, #inference-predicate").forEach((input) => input.closest(".field").classList.toggle("hidden", inferenceTask.value !== "gnn"));
  };
  inferenceTask.addEventListener("change", updateInference);
  updateInference();
}

function renderDatasets() {
  const rows = state.datasets || [];
  const content = rows.length ? `<div class="table-wrap"><table><thead><tr><th>Name</th><th>Format / label</th><th>Rows</th><th>Source status</th><th>Validation</th></tr></thead><tbody>${rows.map((item) => `<tr><td class="cell-primary">${escapeHTML(item.name)}</td><td>${escapeHTML(item.dataset || "JSON")}${item.synthetic ? ` ${badge("synthetic smoke")}` : ""}</td><td>${item.examples == null ? "—" : formatNumber(item.examples)}</td><td>${badge(item.status)}</td>    <td>${item.status === "available" ? `<button class="button small" data-action="validate-dataset" data-task="ner" data-path="${escapeHTML(item.path)}">NER</button> <button class="button small" data-action="validate-dataset" data-task="relation" data-path="${escapeHTML(item.path)}">Relation</button> <button class="button small" data-action="validate-dataset" data-task="nli" data-path="${escapeHTML(item.path)}">NLI</button>` : escapeHTML(item.error || "—")}</td></tr>`).join("")}</tbody></table></div>`
    : emptyState("▦", "No local datasets found", "Place prepared JSON datasets in the configured data directory, or use the CLI preparation commands.");
  return `${pageHeading("Dataset manager", "Inspect local prepared datasets and validate annotation structure before training.",
    `<button class="button" data-action="reload">↻ Refresh datasets</button>`)}
    <article class="card"><div class="card-header"><div><h2>Available local data</h2><p>These are real files discovered by the Flask API; datasets are never silently downloaded.</p></div></div>${content}<div id="dataset-validation" class="section-gap"></div></article>
    <div class="grid two section-gap">
      <article class="card"><h2>NER format</h2><p class="subtitle">JSON examples include word-level <span class="mono">ner_tokens</span> and aligned BIO <span class="mono">ner_tags</span>. CoNLL-2003 conversion retains the official train/validation/test assignment.</p><div class="notice section-gap">Confirm the dataset's access terms and license before preparing or training on it.</div></article>
      <article class="card"><h2>Relation format</h2><p class="subtitle">Each row carries text, one subject/object character span pair, and an explicitly mapped relation label. DocRED evidence sentence IDs and text remain attached to the row.</p><div class="notice warning section-gap">Current relation classification is a text/span classifier, not a document-level multi-hop DocRED system.</div></article>
    </div>`;
}

function renderExperiments() {
  const items = state.experiments || [];
  const table = items.length ? `<div class="table-wrap"><table><thead><tr><th>Task</th><th>Status</th><th>Created</th><th>Result</th><th>Checkpoint</th></tr></thead><tbody>${items.slice().reverse().map((item) => `<tr><td class="cell-primary">${escapeHTML(item.task || item.job_id || "Experiment")}</td><td>${badge(item.status)}</td><td>${escapeHTML(dateLabel(item.created_at))}</td><td>${escapeHTML(JSON.stringify(item.result?.evaluation || item.result?.metrics || item.result?.split_sizes || "—"))}</td><td class="mono">${escapeHTML(shortPath(item.result?.checkpoint || "—"))}</td></tr>`).join("")}</tbody></table></div>`
    : emptyState("◷", "No experiment records", "Training job outcomes are persisted when jobs run in this workspace.");
  return `${pageHeading("Experiments & evaluation", "Inspect persisted training runs and measured evaluation outputs without fabricated trends.",
    `<button class="button" data-action="reload">↻ Refresh runs</button>`)}
    <article class="card"><div class="card-header"><div><h2>Experiment history</h2><p>${items.length} persisted job records</p></div></div>${table}</article>
    <article class="card section-gap"><h2>Evaluation notes</h2><p class="subtitle">Classification runs report metrics calculated on the configured held-out split. Link prediction reports filtered head and tail MRR/Hits@K plus the train-graph degree baseline. Small or synthetic datasets must not be interpreted as generalization benchmarks.</p></article>`;
}

function renderSettings() {
  return `${pageHeading("Settings", "Inspect this local instance and its configured model and dataset paths.",
    `<button class="button" data-action="reload">↻ Check status</button>`)}
    <div class="grid two"><article class="card"><div class="card-header"><div><h2>Service status</h2><p>Values are retrieved from the running Flask application.</p></div></div><div id="settings-health">Loading status…</div></article>
    <article class="card"><div class="card-header"><div><h2>Model registry paths</h2><p>Checkpoint discovery is limited to configured directories.</p></div></div>${renderCheckpointRegistry()}</article></div>
    <article class="card section-gap"><h2>Privacy & deployment</h2><p class="subtitle">The UI uses same-origin Flask endpoints. Documents and checkpoints are stored locally. Do not expose this development API publicly without authentication, upload limits, path restrictions, and deployment hardening.</p></article>`;
}

async function loadSettingsHealth() {
  if (state.page !== "settings") return;
  const health = await api("/health");
  $("#settings-health").innerHTML = `<div class="detail-row"><small>API</small><span>${escapeHTML(health.status)}</span></div><div class="detail-row"><small>Graph</small><span>${escapeHTML(health.graph)}</span></div><div class="detail-row"><small>Entities</small><span>${formatNumber(health.entities)}</span></div><div class="detail-row"><small>Relationships</small><span>${formatNumber(health.relationships)}</span></div><div class="detail-row"><small>Documents</small><span>${formatNumber(health.documents)}</span></div>`;
}

function renderVerificationResult(result) {
  const verdict = result.verdict || result.label || "NOT ENOUGH INFORMATION";
  const evidence = result.evidence || [];
  return `<div class="card-header"><div><div class="eyebrow">RESULT</div><h2>${escapeHTML(verdict)}</h2><p>${escapeHTML(result.explanation || result.reason || "Review evidence and provenance below.")}</p></div>${badge(verdict)}</div>
    ${result.confidence != null ? `<div class="detail-row"><small>Model confidence</small><span>${escapeHTML(result.confidence)} · ${escapeHTML(result.confidence_type || "model score")}</span></div>` : ""}
    ${result.evidence_strength != null ? `<div class="detail-row"><small>Evidence strength</small><span>${escapeHTML(result.evidence_strength)}</span></div>` : ""}
    ${evidence.length ? evidence.map((item) => `<div class="evidence-block section-gap">${escapeHTML(item.snippet || item.fact || "Evidence passage")}<div class="field-help">Source: ${escapeHTML(item.source || item.provenance?.document_id || "Not recorded")}</div></div>`).join("") : `<div class="notice warning section-gap">No supporting passage was returned. Absence of evidence is not proof of falsity.</div>`}`;
}

async function submitDocument(form, alsoExtract = false) {
  const file = $("#document-file").files[0];
  if (file) {
    const created = await uploadDocument(file);
    if (alsoExtract) await extractDocument(created.document.document_id);
    else toast("Document uploaded and stored locally.");
    await refreshState();
    return;
  }
  const text = $("#document-text").value;
  if (!text.trim()) throw new Error("Choose a document file or enter text to store.");
  const documentId = $("#document-id").value.trim();
  const payload = { text, ...(documentId ? { document_id: documentId } : {}) };
  const result = await api("/documents", { method: "POST", body: JSON.stringify(payload) });
  const id = result.document.document_id;
  if (alsoExtract) await extractDocument(id, text);
  else toast("Document stored locally.");
  await refreshState();
}

async function uploadDocument(file = $("#document-file").files[0]) {
  if (!file) throw new Error("Choose a document file to upload.");
  const formData = new FormData();
  formData.append("file", file);
  const documentId = $("#document-id").value.trim();
  if (documentId) formData.append("document_id", documentId);
  return api("/documents/upload", { method: "POST", body: formData });
}

async function extractDocument(documentId, selectedCheckpoints = {}) {
  const ner = selectedCheckpoints.ner ?? $("#ner-checkpoint").value;
  const relation = selectedCheckpoints.relation ?? $("#relation-checkpoint").value;
  await api("/documents/extract", {
    method: "POST",
    body: JSON.stringify({ document_id: documentId, ner_checkpoint: ner, relation_checkpoint: relation }),
  });
  toast("Extraction complete. Candidate relationships await review.");
  await refreshState();
}

function drawChat() {
  const thread = $("#chat-thread");
  if (!thread) return;
  thread.innerHTML = state.chat.map((message) => `<div class="chat-message ${message.role}">${escapeHTML(message.text)}${message.evidence?.length ? `<div class="chat-sources">${message.evidence.map((item) => `<span class="source-chip">${escapeHTML(item.source || "Source not recorded")}</span>`).join("")}</div>${message.evidence.map((item) => `<div class="evidence-block section-gap">${escapeHTML(item.fact)}<br><span class="muted">${escapeHTML((item.graph_path || []).join(" → "))}</span></div>`).join("")}` : ""}</div>`).join("");
  thread.scrollTop = thread.scrollHeight;
}

async function startTraining(form) {
  const task = $("#training-task").value;
  const path = $("#dataset-path").value.trim() || $("#training-dataset").value;
  const config = {
    epochs: Number($("#epochs").value),
    batch_size: Number($("#batch-size").value),
    random_seed: Number($("#seed").value),
    device: $("#device").value,
    pretrained_model: $("#pretrained-model").value.trim(),
    learning_rate: Number($("#learning-rate").value),
    ner_architecture: $("#ner-architecture").value,
  };
  let endpoint, payload;
  if (task === "gnn") {
    endpoint = "/gnn/train";
    payload = { config };
  } else if (task === "nli") {
    if (!path) throw new Error("Choose an NLI dataset path.");
    endpoint = "/verify/train";
    payload = { dataset_path: path, config };
  } else {
    if (!path) throw new Error("Choose an NLP dataset path.");
    endpoint = "/nlp/train";
    payload = { task, dataset_path: path, config };
  }
  if (config.epochs < 1 || config.batch_size < 1 || !Number.isFinite(config.learning_rate) || config.learning_rate <= 0) throw new Error("Training parameters must be positive finite values.");
  const response = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
  state.job = { job_id: response.job_id, task, status: response.status, logs: ["Job accepted by the Flask background worker."], created_at: new Date().toISOString() };
  renderPage();
  pollJob(response.job_id);
}

async function pollJob(jobId) {
  try {
    const job = await api(`/jobs/${encodeURIComponent(jobId)}`);
    state.job = job;
    if (state.page === "model-lab") $("#job-panel").innerHTML = renderJob(job);
    if (["queued", "running"].includes(job.status)) {
      setTimeout(() => pollJob(jobId), 1500);
    } else {
      await refreshState();
      toast(job.status === "completed" ? "Training job completed." : `Training job ${job.status}.`, job.status === "failed");
    }
  } catch (error) {
    state.job = { ...state.job, status: "failed", error: error.message };
    if (state.page === "model-lab") $("#job-panel").innerHTML = renderJob(state.job);
  }
}

document.addEventListener("click", async (event) => {
  const nav = event.target.closest("[data-page], [data-navigate]");
  if (nav) {
    event.preventDefault();
    setPage(nav.dataset.page || nav.dataset.navigate);
    return;
  }
  const action = event.target.closest("[data-action]");
  if (!action) return;
  try {
    switch (action.dataset.action) {
      case "reload": await refreshState(); toast("Workspace data refreshed."); break;
      case "graph-reset": state.selectedEntity = null; state.graphViewport = null; drawGraph(graphFilteredRelations()); $("#entity-details").innerHTML = emptyState("◉", "Select an entity", "Click a graph node to inspect its relationships and source provenance."); break;
      case "upload-document": {
        const form = $("#document-form");
        const created = await uploadDocument();
        form.reset();
        await refreshState();
        toast(`Uploaded ${created.document.filename}.`);
        break;
      }
      case "upload-and-extract": {
        const form = $("#document-form");
        const selectedCheckpoints = {
          ner: $("#ner-checkpoint").value,
          relation: $("#relation-checkpoint").value,
        };
        const created = await uploadDocument();
        try {
          await extractDocument(created.document.document_id, selectedCheckpoints);
          form.reset();
        } catch (error) {
          await refreshState();
          throw error;
        }
        break;
      }
      case "extract-document": {
        const doc = state.documents.find((item) => item.document_id === action.dataset.id);
        if (!doc) throw new Error("Document is no longer available. Refresh and retry.");
        await extractDocument(doc.document_id);
        break;
      }
      case "review-candidate": {
        await api(`/documents/${encodeURIComponent(action.dataset.id)}/review`, {
          method: "POST",
          body: JSON.stringify({ candidate_index: Number(action.dataset.index), decision: action.dataset.decision }),
        });
        toast(action.dataset.decision === "approve" ? "Fact approved with source provenance." : "Candidate rejected.");
        await refreshState();
        break;
      }
      case "validate-dataset": {
        const endpoint = action.dataset.task === "nli" ? "/verify/dataset/validate" : "/nlp/dataset/validate";
        const validation = await api(endpoint, {
          method: "POST",
          body: JSON.stringify({ dataset_path: action.dataset.path, task: action.dataset.task }),
        });
        $("#dataset-validation").innerHTML = `<div class="notice ${validation.synthetic_data ? "warning" : ""}">Dataset valid for ${escapeHTML(validation.task)}: ${formatNumber(validation.examples)} examples · ${escapeHTML(JSON.stringify(validation.splits))}${validation.synthetic_data ? " · Synthetic smoke data; not benchmark performance." : ""}</div>`;
        break;
      }
      case "cancel-job": {
        await api(`/jobs/${encodeURIComponent(action.dataset.id)}/cancel`, { method: "POST", body: JSON.stringify({}) });
        toast("Queued job cancelled.");
        await pollJob(action.dataset.id);
        break;
      }
      case "setup-demo-models": {
        const response = await api("/nlp/demo-setup", {
          method: "POST",
          body: JSON.stringify({}),
        });
        state.job = {
          job_id: response.job_id,
          task: "nlp-demo-setup",
          status: response.status,
          logs: ["Real NER and relation training queued on the bundled synthetic smoke dataset."],
          created_at: new Date().toISOString(),
        };
        setPage("model-lab");
        pollJob(response.job_id);
        break;
      }
      case "retry-job": {
        const response = await api(`/jobs/${encodeURIComponent(action.dataset.id)}/retry`, { method: "POST", body: JSON.stringify({}) });
        state.job = { job_id: response.job_id, task: "retry", status: response.status, logs: ["Retry queued."], created_at: new Date().toISOString() };
        renderPage();
        pollJob(response.job_id);
        break;
      }
      case "load-nli": {
        const checkpoint = $("#verification-checkpoint").value;
        if (!checkpoint) throw new Error("Select an NLI checkpoint to load.");
        await api("/models/load", { method: "POST", body: JSON.stringify({ task: "nli", checkpoint }) });
        $("#verification-model-status").textContent = `Loaded ${shortPath(checkpoint)}`;
        toast("NLI checkpoint loaded for verification.");
        break;
      }
      case "copy-path":
        await navigator.clipboard.writeText(action.dataset.path);
        toast("Checkpoint path copied.");
        break;
    }
  } catch (error) { toast(error.message, true); }
});

document.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    if (event.target.id === "document-form") {
      await submitDocument(event.target);
      event.target.reset();
    } else if (event.target.id === "verify-form") {
      const result = await api("/verify", { method: "POST", body: JSON.stringify({ claim: $("#claim-input").value }) });
      $("#verification-result").innerHTML = renderVerificationResult(result);
    } else if (event.target.id === "rag-form") {
      const query = $("#rag-query").value.trim();
      state.chat.push({ role: "user", text: query });
      const response = await api("/rag", { method: "POST", body: JSON.stringify({ query }) });
      state.chat.push({ role: "assistant", text: response.answer, evidence: response.evidence || [] });
      $("#rag-query").value = "";
      renderPage();
    } else if (event.target.id === "training-form") {
      await startTraining(event.target);
    } else if (event.target.id === "inference-form") {
      const task = $("#inference-task").value;
      const checkpoint = $("#inference-checkpoint").value;
      if (!checkpoint) throw new Error("Select a saved checkpoint.");
      let endpoint, payload;
      if (task === "gnn") {
        endpoint = "/gnn/predict";
        payload = {
          checkpoint,
          subject: $("#inference-entity").value.trim(),
          predicate: $("#inference-predicate").value.trim(),
          top_k: 10,
        };
      } else {
        endpoint = "/nlp/infer";
        payload = { task, checkpoint, text: $("#inference-text").value };
        if (task === "relation") {
          const parseSpan = (value) => {
            const parts = value.split(",").map((part) => Number(part.trim()));
            if (parts.length !== 2 || parts.some((part) => !Number.isInteger(part) || part < 0)) throw new Error("Enter each span as non-negative start,end character offsets.");
            return { start: parts[0], end: parts[1] };
          };
          payload.subject = parseSpan($("#inference-subject").value);
          payload.object = parseSpan($("#inference-object").value);
        }
      }
      const response = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
      const resultNode = $("#inference-result");
      resultNode.textContent = JSON.stringify(response, null, 2);
      resultNode.classList.remove("hidden");
    }
  } catch (error) { toast(error.message, true); }
});

document.addEventListener("input", (event) => {
  if (event.target.id === "evidence-search") $("#evidence-list").innerHTML = renderEvidenceRows(state.graph?.relations || []);
});

$("#menu-button").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
$("#refresh-button").addEventListener("click", () => refreshState().catch((error) => toast(error.message, true)));

const renderPageAfterSettings = renderPage;
renderPage = async function () {
  await renderPageAfterSettings();
  await loadSettingsHealth().catch((error) => {
    const node = $("#settings-health");
    if (node) node.textContent = error.message;
  });
};

refreshState().catch((error) => toast(error.message, true));
