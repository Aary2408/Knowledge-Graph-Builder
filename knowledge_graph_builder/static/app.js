const state = {
  page: "home",
  graph: null,
  graphStatus: "loading",
  documents: [],
  datasets: [],
  models: { nlp: [], nlp_by_task: { ner: [], relation: [] }, nlp_checkpoints: [], gnn: [], nli: [] },
  experiments: [],
  job: null,
  selectedEntity: null,
  selectedDocument: null,
  processing: false,
  documentAnswer: null,
  graphExplorer: {
    model: null,
    expanded: new Set(),
    autoExpanded: new Set(),
    collapsed: new Set(),
    connectionLimits: new Map(),
    rootLimits: new Map(),
    viewport: { x: 28, y: 30, scale: 1 },
    drag: null,
  },
  chat: [],
};

const pageNames = {
  home: "Home",
  overview: "Home",
  graph: "Knowledge graph",
  documents: "Documents",
  "ask-ai": "Ask AI",
  "advanced-lab": "Advanced Lab",
  verification: "Claim verification",
  assistant: "GraphRAG assistant",
  evidence: "Evidence explorer",
  "model-lab": "Model Lab",
  datasets: "Dataset manager",
  experiments: "Experiments",
  settings: "Settings",
};
const mainPages = new Set(["home", "documents", "graph", "ask-ai", "advanced-lab"]);

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
  const tone = ["completed", "approved", "human_reviewed", "supported", "ready"].includes(key) ? "green"
    : ["queued", "running", "processing", "analyzing", "uploaded", "stored", "review_required", "not enough information"].includes(key) ? "amber"
      : ["failed", "rejected", "refuted"].includes(key) ? "red"
        : ["predicted", "candidate", "extracted_candidate"].includes(key) ? "violet" : "neutral";
  return `<span class="badge ${tone}">${escapeHTML(String(value || "unknown").replaceAll("_", " "))}</span>`;
}

function setPage(page) {
  state.page = pageNames[page] ? page : "home";
  const activePage = state.page === "overview" ? "home" : mainPages.has(state.page) ? state.page : "advanced-lab";
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.page === activePage));
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
  if (results[1].status === "fulfilled") {
    state.graph = results[1].value;
    state.graphStatus = "ready";
  } else {
    state.graphStatus = "error";
  }
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
      case "home":
      case "overview": root.innerHTML = await renderOverview(); break;
      case "graph": root.innerHTML = renderGraph(); bindGraphExplorer(); break;
      case "documents": root.innerHTML = renderDocuments(); break;
      case "ask-ai": root.innerHTML = renderAssistant(); break;
      case "advanced-lab": root.innerHTML = renderAdvancedLab(); break;
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
  const recent = state.documents.slice(0, 4);
  return `${pageHeading("Understand your documents", "Upload a source and get its summary, key facts, entities, and relationships.",
    `<button class="button" data-navigate="documents">Browse documents</button>`)}
    ${documentUploadPanel()}
    <div class="card section-gap"><div class="card-header"><div><h2>Recent documents</h2><p>Reopen an analysis at any time.</p></div><button class="button small" data-navigate="documents">View all</button></div>
      ${recent.length ? `<div class="list">${recent.map((doc) => `<div class="list-row"><div class="list-main"><strong>${escapeHTML(doc.filename || doc.document_id)}</strong><small>${escapeHTML(doc.preview || "")}</small></div><button class="button small" data-action="open-document" data-id="${escapeHTML(doc.document_id)}">${doc.processing_status === "ready" ? "Open results" : "Open"}</button></div>`).join("")}</div>` : emptyState("▤", "Your library is ready", "Analyze your first document to see its results here.")}
    </div>`;
}

function documentUploadPanel() {
  const processing = state.processing ? `<div class="processing-state" id="processing-state"><span class="spinner"></span>Analyzing your document. This can take a little while.</div>` : "";
  return `<article class="card upload-card"><div class="card-header"><div><h2>Analyze a document</h2><p>PDF, TXT, Markdown, or DOCX · source text stays in your local workspace.</p></div></div>
    <form id="document-form" class="form-grid">
      <div class="field"><label for="document-id">Document name (optional)</label><input id="document-id" name="document_id" placeholder="Auto-generated"></div>
      <div class="field"><label for="document-file">Choose a document</label><input id="document-file" name="file" type="file" accept=".txt,.md,.pdf,.docx,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"><div class="field-help">Text-based PDFs are supported. Scanned PDFs need OCR first.</div></div>
      <div class="field full"><div class="drop-zone" id="document-dropzone" tabindex="0" role="button" aria-label="Drop a document or browse files">Drop a document here, or <label class="browse-button" for="document-file">Browse Files</label><div class="selected-file" id="selected-file" aria-live="polite"></div></div></div>
      <div class="field full"><label for="document-text">Or paste text</label><textarea class="large" id="document-text" name="text" placeholder="Paste text from a source you are authorized to process…"></textarea></div>
      ${processing}
      <div class="form-actions field full"><button class="button primary" type="submit" ${state.processing ? "disabled" : ""}>${state.processing ? "Processing…" : "Process Document"}</button><span class="field-help">Extracted relationships remain candidates until reviewed.</span></div>
    </form></article>`;
}

function renderGraph() {
  const heading = pageHeading("Knowledge graph", "Follow connected entities through a hierarchical relationship tree. The stored graph remains unchanged.",
    `<button class="button" data-action="reload">↻ Refresh graph</button>`);
  if (state.graphStatus === "loading") {
    return `${heading}<div class="card loading-state"><span class="spinner"></span>Loading graph entities and relationships…</div>`;
  }
  if (state.graphStatus === "error") {
    return `${heading}<div class="error-state">Could not load graph data. <button class="button small" data-action="reload">Retry</button></div>`;
  }
  const explorer = window.KnowGraphExplorer;
  state.graphExplorer.model = explorer.buildModel(state.graph);
  const model = state.graphExplorer.model;
  const categories = explorer.CATEGORY_ORDER.filter((category) =>
    [...model.entities.values()].some((entity) => explorer.entityCategory(entity) === category)
  );
  const predicates = [...new Set(model.relations.map((relation) => relation.predicate).filter(Boolean))].sort();
  const sources = [...new Set(model.relations.map(explorer.relationSource).filter(Boolean))].sort();
  const options = (values) => values.map((value) => `<option value="${escapeHTML(value)}">${escapeHTML(value)}</option>`).join("");
  return `${heading}
    <section class="graph-explorer">
      <div class="explorer-toolbar">
        <label class="explorer-search"><span aria-hidden="true">⌕</span><input id="graph-search" type="search" placeholder="Search entity names…" aria-label="Search entities"></label>
        <select id="graph-type-filter" class="explorer-filter" aria-label="Filter entity type"><option value="">All entity types</option>${options(categories)}</select>
        <select id="graph-relation-filter" class="explorer-filter" aria-label="Filter relationship type"><option value="">All relationships</option>${options(predicates)}</select>
        <select id="graph-source-filter" class="explorer-filter" aria-label="Filter source document"><option value="">All source documents</option>${sources.map((source) => `<option value="${escapeHTML(source)}">${escapeHTML(explorerSourceLabel(source))}</option>`).join("")}</select>
        <div class="explorer-actions"><button class="button small" data-action="explorer-expand-all">Expand all</button><button class="button small" data-action="explorer-collapse-all">Collapse all</button></div>
      </div>
      <div class="explorer-summary"><span><strong>${formatNumber(model.entities.size)}</strong> entities</span><span>${formatNumber(model.relations.length)} relationships</span><span id="explorer-result-count"></span></div>
      <nav class="explorer-breadcrumb" id="graph-breadcrumb" aria-label="Entity navigation path"></nav>
      <div class="explorer-layout">
        <section class="card explorer-tree-panel" aria-label="Entity tree">
          <div class="explorer-panel-heading"><div><h2>Relationship tree</h2><p>Cross-links and cycles appear as navigable references.</p></div><span class="badge neutral">${formatNumber(model.roots.length)} roots</span></div>
          <div class="tree-controls">
            <span class="tree-pan-hint">Drag to pan · Scroll to zoom</span>
            <div class="tree-zoom-controls" role="group" aria-label="Tree zoom controls">
              <button class="icon-button" data-action="explorer-zoom-out" aria-label="Zoom out" title="Zoom out">−</button>
              <button class="icon-button" data-action="explorer-zoom-reset" aria-label="Reset tree zoom" title="Reset zoom">⟲</button>
              <button class="icon-button" data-action="explorer-zoom-in" aria-label="Zoom in" title="Zoom in">+</button>
            </div>
          </div>
          <div class="tree-viewport" id="tree-viewport" aria-label="Zoomable relationship tree">
            <div class="tree-stage" id="tree-stage"><div class="explorer-tree" id="entity-tree" role="tree"></div></div>
          </div>
        </section>
        <aside class="card explorer-details" id="entity-details" aria-label="Entity details">${emptyState("◉", "Select an entity", "Choose a tree item or relationship reference to inspect graph details and provenance.")}</aside>
      </div>
    </section>`;
}

function explorerSourceLabel(source) {
  const doc = state.documents.find((item) => item.document_id === source);
  return doc ? `${doc.filename || doc.document_id} · ${source}` : source;
}

function bindGraphExplorer() {
  if (!$("#tree-viewport")) return;
  const update = () => {
    state.graphExplorer.collapsed.clear();
    renderGraphTree();
  };
  ["graph-search", "graph-type-filter", "graph-relation-filter", "graph-source-filter"].forEach((id) => {
    const control = $(`#${id}`);
    control.addEventListener(id === "graph-search" ? "input" : "change", update);
  });
  bindTreeViewport();
  renderGraphTree();
  if (state.selectedEntity) renderExplorerDetails(state.selectedEntity);
}

function updateTreeTransform() {
  const stage = $("#tree-stage");
  if (!stage) return;
  const { x, y, scale } = state.graphExplorer.viewport;
  stage.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
}

function zoomTree(factor, clientX, clientY) {
  const viewport = $("#tree-viewport");
  if (!viewport) return;
  const rect = viewport.getBoundingClientRect();
  const localX = clientX == null ? rect.width / 2 : clientX - rect.left;
  const localY = clientY == null ? rect.height / 2 : clientY - rect.top;
  const view = state.graphExplorer.viewport;
  const scale = Math.min(2.4, Math.max(.35, view.scale * factor));
  const contentX = (localX - view.x) / view.scale;
  const contentY = (localY - view.y) / view.scale;
  state.graphExplorer.viewport = {
    x: localX - contentX * scale,
    y: localY - contentY * scale,
    scale,
  };
  updateTreeTransform();
}

function bindTreeViewport() {
  const viewport = $("#tree-viewport");
  if (!viewport) return;
  updateTreeTransform();
  viewport.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomTree(event.deltaY < 0 ? 1.12 : 1 / 1.12, event.clientX, event.clientY);
  }, { passive: false });
  viewport.addEventListener("pointerdown", (event) => {
    if (event.target.closest("[data-action]")) return;
    state.graphExplorer.drag = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      viewX: state.graphExplorer.viewport.x,
      viewY: state.graphExplorer.viewport.y,
    };
    viewport.classList.add("is-panning");
    viewport.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  viewport.addEventListener("pointermove", (event) => {
    const drag = state.graphExplorer.drag;
    if (!drag || drag.pointerId !== event.pointerId) return;
    state.graphExplorer.viewport.x = drag.viewX + event.clientX - drag.x;
    state.graphExplorer.viewport.y = drag.viewY + event.clientY - drag.y;
    updateTreeTransform();
  });
  const stopDragging = () => {
    state.graphExplorer.drag = null;
    viewport.classList.remove("is-panning");
  };
  viewport.addEventListener("pointerup", stopDragging);
  viewport.addEventListener("pointercancel", stopDragging);
}

function explorerFilters() {
  return {
    query: $("#graph-search")?.value || "",
    type: $("#graph-type-filter")?.value || "",
    predicate: $("#graph-relation-filter")?.value || "",
    source: $("#graph-source-filter")?.value || "",
  };
}

function renderGraphTree() {
  const model = state.graphExplorer.model;
  const tree = $("#entity-tree");
  if (!model || !tree) return;
  const explorer = window.KnowGraphExplorer;
  const filters = explorerFilters();
  const hasFilters = Object.values(filters).some(Boolean);
  const { matches, relevant } = explorer.matchingBranches(model, filters);
  const autoExpanded = new Set();
  const revealPath = (id) => {
    let cursor = id;
    while (model.parentByEntity.has(cursor)) {
      cursor = model.parentByEntity.get(cursor).entityId;
      autoExpanded.add(cursor);
    }
  };
  matches.forEach(revealPath);
  if (state.selectedEntity) revealPath(state.selectedEntity);
  state.graphExplorer.autoExpanded = autoExpanded;
  const labels = {
    People: ["●", "people"],
    Organizations: ["▦", "organizations"],
    Locations: ["⌖", "locations"],
    Concepts: ["✦", "concepts"],
    Events: ["◷", "events"],
    Other: ["◇", "other"],
  };
  const renderBranch = (id, ancestorIds = new Set()) => {
    const entity = model.entities.get(id);
    const category = explorer.entityCategory(entity);
    const categoryIcon = labels[category];
    const connectionLimit = state.graphExplorer.connectionLimits.get(id) || 60;
    const connectionPage = explorer.visibleConnections(
      model,
      id,
      connectionLimit,
      0,
      (connection) => relevant.has(connection.targetId)
    );
    const children = connectionPage.connections;
    const expanded = explorer.isExpanded(
      id,
      state.graphExplorer.expanded,
      autoExpanded,
      state.graphExplorer.collapsed
    );
    const ancestors = new Set(ancestorIds);
    ancestors.add(id);
    const relationBranches = expanded ? children.map(({ targetId, links }) => {
      const target = model.entities.get(targetId);
      const isTreeChild = model.parentByEntity.get(targetId)?.entityId === id && !ancestors.has(targetId);
      const edgeMarkup = links.map(({ relation, direction }) =>
        `<span class="relationship-label" title="${escapeHTML(`${direction}: ${relation.subject} ${relation.predicate} ${relation.object}`)}">${direction === "outgoing" ? "→" : "←"} ${escapeHTML(relation.predicate || "related")}</span>`
      ).join("");
      const connectedEntity = isTreeChild
        ? renderBranch(targetId, ancestors)
        : `<button class="entity-reference" data-action="explorer-reference" data-id="${escapeHTML(targetId)}" title="Navigate to ${escapeHTML(target?.label || targetId)}">↗ ${escapeHTML(target?.label || targetId)}</button>`;
      return `<div class="tree-child" role="treeitem"><div class="tree-connection-labels">${edgeMarkup}</div>${connectedEntity}</div>`;
    }).join("") : "";
    const moreConnections = expanded && connectionPage.hasMore
      ? `<button class="tree-load-more" data-action="explorer-more-connections" data-id="${escapeHTML(id)}">Show ${Math.min(60, connectionPage.total - connectionLimit)} more connections</button>`
      : "";
    return `<div class="tree-node type-${categoryIcon[1]} ${state.selectedEntity === id ? "entity-selected" : ""}" role="treeitem" aria-expanded="${expanded}">
      <div class="tree-node-card ${matches.has(id) ? "entity-match" : ""}">
        <button class="tree-node-select" data-action="explorer-select" data-id="${escapeHTML(id)}">
          <span class="tree-node-type type-${categoryIcon[1]}" aria-hidden="true">${categoryIcon[0]}</span>
          <span class="tree-node-copy"><strong>${escapeHTML(entity.label || id)}</strong><small>${escapeHTML(id)}</small></span>
          <span class="entity-type-badge type-${categoryIcon[1]}">${escapeHTML(category)}</span>
        </button>
        <button class="tree-node-toggle" data-action="explorer-toggle" data-id="${escapeHTML(id)}" aria-label="${expanded ? "Collapse" : "Expand"} ${escapeHTML(entity.label || id)}" aria-expanded="${expanded}" ${connectionPage.total ? "" : "disabled"}>${connectionPage.total ? (expanded ? "−" : "+") : "·"}</button>
      </div>
      ${expanded && relationBranches ? `<div class="tree-descendants"><div class="tree-child-rail ${children.length === 1 ? "single-child" : ""}">${relationBranches}</div>${moreConnections}</div>` : ""}
    </div>`;
  };

  const categoryMarkup = explorer.CATEGORY_ORDER.map((category) => {
    const roots = model.rootsByCategory.get(category).filter((id) => relevant.has(id));
    if (!roots.length) return "";
    const [icon, colorClass] = labels[category];
    const rootLimit = state.graphExplorer.rootLimits.get(category) || 80;
    const prioritized = hasFilters
      ? [...roots].sort((left, right) => Number(matches.has(right)) - Number(matches.has(left)))
      : roots;
    const shownRoots = prioritized.slice(0, rootLimit);
    const moreRoots = roots.length > rootLimit
      ? `<button class="tree-load-more" data-action="explorer-more-roots" data-category="${escapeHTML(category)}">Show ${Math.min(80, roots.length - rootLimit)} more roots</button>`
      : "";
    return `<section class="entity-group" role="group"><h3><span class="entity-type-icon type-${colorClass}" aria-hidden="true">${icon}</span>${escapeHTML(category)}<small>${formatNumber(roots.length)} roots</small></h3><div class="tree-root-rail">${shownRoots.map((id) => renderBranch(id)).join("")}</div>${moreRoots}</section>`;
  }).join("");
  tree.innerHTML = categoryMarkup || emptyState("⌕", model.entities.size ? "No matching entities" : "This graph is empty", model.entities.size ? "Adjust or clear the search and filters to see more entities." : "Entities and their relationships will appear here when the graph contains data.");
  $("#explorer-result-count").textContent = hasFilters ? `${formatNumber(matches.size)} matching entities` : "";
  renderExplorerBreadcrumb();
}

function renderExplorerBreadcrumb() {
  const node = $("#graph-breadcrumb");
  if (!node) return;
  const model = state.graphExplorer.model;
  const current = state.selectedEntity;
  if (!current || !model?.entities.has(current)) {
    node.innerHTML = `<span>All entities</span>`;
    return;
  }
  const path = [current];
  let cursor = current;
  while (model.parentByEntity.has(cursor)) {
    cursor = model.parentByEntity.get(cursor).entityId;
    path.unshift(cursor);
  }
  node.innerHTML = `<button data-action="explorer-clear-selection">All entities</button>${path.map((id, index) =>
    `${index ? '<span aria-hidden="true">›</span>' : ""}<button data-action="explorer-select" data-id="${escapeHTML(id)}" ${index === path.length - 1 ? 'aria-current="page"' : ""}>${escapeHTML(model.entities.get(id).label || id)}</button>`
  ).join("")}`;
}

function renderExplorerDetails(entityId) {
  const details = $("#entity-details");
  const model = state.graphExplorer.model;
  const entity = model?.entities.get(entityId);
  if (!details || !entity) return;
  const explorer = window.KnowGraphExplorer;
  const attributes = entity.attributes || {};
  const description = attributes.description || attributes.summary || "";
  const rawType = explorer.entityType(entity);
  const connected = model.incidentByEntity.get(entityId) || [];
  const renderRelationship = (relation, incoming) => {
    const connectedId = incoming ? relation.subject : relation.object;
    const connectedEntity = model.entities.get(connectedId);
    const source = explorer.relationSource(relation);
    const document = state.documents.find((item) => item.document_id === source);
    return `<article class="detail-relationship">
      <div class="detail-relationship-heading"><span class="relationship-label">${incoming ? "←" : "→"} ${escapeHTML(relation.predicate || "related")}</span>${badge(relation.relation_type)}</div>
      <button class="connected-entity" data-action="explorer-reference" data-id="${escapeHTML(connectedId)}">${escapeHTML(connectedEntity?.label || connectedId)}</button>
      <div class="detail-source">Source: ${escapeHTML(document?.filename || source || "Not recorded")}${source && document ? ` · ${escapeHTML(source)}` : ""}${relation.confidence != null ? ` · confidence ${escapeHTML(relation.confidence)}` : ""}</div>
      ${relation.provenance?.snippet ? `<div class="evidence-block">${escapeHTML(relation.provenance.snippet)}</div>` : ""}
      ${relation.provenance && Object.keys(relation.provenance).some((key) => !["document_id", "snippet"].includes(key))
        ? `<details class="provenance-extra"><summary>More provenance</summary><pre>${escapeHTML(JSON.stringify(Object.fromEntries(Object.entries(relation.provenance).filter(([key]) => !["document_id", "snippet"].includes(key))), null, 2))}</pre></details>` : ""}
    </article>`;
  };
  const outgoing = model.relations.filter((relation) => relation.subject === entityId);
  const incoming = model.relations.filter((relation) => relation.object === entityId && relation.subject !== entityId);
  const properties = Object.fromEntries(Object.entries(attributes).filter(([key]) =>
    !["type", "entity_type", "entityType", "category", "description", "summary"].includes(key)
  ));
  details.innerHTML = `<header class="entity-detail-heading">
    <div class="eyebrow">ENTITY DETAILS</div><h2>${escapeHTML(entity.label || entityId)}</h2>
    <span class="entity-type-badge type-${explorer.entityCategory(entity).toLowerCase()}">${escapeHTML(rawType || "Other")}</span>
  </header>
  <div class="detail-row"><small>Entity ID</small><span class="mono">${escapeHTML(entityId)}</span></div>
  ${description ? `<div class="detail-row"><small>Description</small><span>${escapeHTML(description)}</span></div>` : ""}
  ${Object.keys(properties).length ? `<div class="detail-row"><small>Properties</small><pre class="entity-properties">${escapeHTML(JSON.stringify(properties, null, 2))}</pre></div>` : ""}
  <div class="entity-detail-counts"><span><strong>${formatNumber(outgoing.length)}</strong> outgoing</span><span><strong>${formatNumber(incoming.length)}</strong> incoming</span><span><strong>${formatNumber(connected.length)}</strong> connections</span></div>
  <section class="detail-relations"><h3>Outgoing relationships</h3>${outgoing.length ? outgoing.map((relation) => renderRelationship(relation, false)).join("") : `<p class="detail-empty">No outgoing relationships.</p>`}</section>
  <section class="detail-relations"><h3>Incoming relationships</h3>${incoming.length ? incoming.map((relation) => renderRelationship(relation, true)).join("") : `<p class="detail-empty">No incoming relationships.</p>`}</section>`;
  renderExplorerBreadcrumb();
}

function selectExplorerEntity(entityId) {
  if (!state.graphExplorer.model?.entities.has(entityId)) return;
  state.selectedEntity = entityId;
  state.graphExplorer.collapsed.clear();
  renderExplorerDetails(entityId);
  renderGraphTree();
}

function selectRelation(relation) {
  if (!relation) return;
  const details = $("#entity-details");
  if (!details) return;
  details.innerHTML = `<div class="card-header"><div><div class="eyebrow">RELATIONSHIP EVIDENCE</div><h2>${escapeHTML(relation.subject)} ${escapeHTML(relation.predicate)} ${escapeHTML(relation.object)}</h2><p>${badge(relation.relation_type)}</p></div></div>
    ${relation.provenance?.snippet ? `<div class="evidence-block">${escapeHTML(relation.provenance.snippet)}</div>` : `<p class="muted">No source excerpt was recorded for this edge.</p>`}
    <div class="detail-row"><small>Source</small><span>${escapeHTML(relation.provenance?.document_id || relation.source || "Not recorded")}</span></div>
    ${relation.provenance?.evidence?.length ? `<div class="detail-row"><small>Additional source excerpts</small>${relation.provenance.evidence.map((item) => `<div class="evidence-block section-gap">${escapeHTML(item.snippet)}<small>${escapeHTML(item.page ? `Page ${item.page}` : "")}</small></div>`).join("")}</div>` : ""}
    <details class="technical-details"><summary>Technical details</summary><span class="mono">${escapeHTML(JSON.stringify({ confidence: relation.confidence, source: relation.source, provenance: relation.provenance || {} }))}</span></details>`;
}

function renderDocuments() {
  const docs = state.documents;
  const list = docs.length ? `<div class="list">${docs.map((doc) => `
    <div class="list-row"><div class="list-main"><strong>${escapeHTML(doc.filename || doc.document_id)}</strong><small>${escapeHTML(doc.preview || doc.document_id)}</small><small>${escapeHTML(dateLabel(doc.updated_at))} · ${escapeHTML(doc.processing_status || doc.status || "uploaded")}</small></div>
      <div class="document-actions">${badge(doc.processing_status || doc.status)}<button class="button small" data-action="open-document" data-id="${escapeHTML(doc.document_id)}">Open</button></div></div>`).join("")}</div>`
    : emptyState("▤", "Your document library is empty", "Upload a PDF, TXT, Markdown, or DOCX document to begin.");
  const selected = state.selectedDocument ? documentResults(state.selectedDocument) : "";
  return `${pageHeading("Documents", "Your sources and their analysis results.", `<button class="button" data-action="reload">Refresh</button>`)}
    ${documentUploadPanel()}
    <article class="card section-gap"><div class="card-header"><div><h2>Document library</h2><p>${docs.length} saved source${docs.length === 1 ? "" : "s"}</p></div></div>${list}</article>
    ${selected}`;
}

function documentResults(doc) {
  const facts = doc.key_facts || [];
  const entities = doc.entities || [];
  const candidates = doc.candidate_relations || [];
  const relationships = candidates.length ? `<div class="list">${candidates.map((item, index) => `
    <div class="list-row"><div class="list-main"><strong>${escapeHTML(item.subject)} <span class="muted">${escapeHTML(item.label)}</span> ${escapeHTML(item.object)}</strong>
      <small>${escapeHTML(item.evidence_snippet || "")}</small><small>${escapeHTML(item.page ? `Page ${item.page} · ` : "")}${escapeHTML(item.extraction_method === "lightweight_rule_based" ? "Rule match; not a trained model." : item.synthetic_model ? "Synthetic-checkpoint prediction; review carefully." : "Model-extracted candidate.")}</small></div>
      <div class="document-actions">${badge(item.status)}${item.status === "extracted_candidate" ? `<button class="button small" data-action="review-candidate" data-id="${escapeHTML(doc.document_id)}" data-index="${index}" data-decision="approve">Approve</button><button class="button small danger" data-action="review-candidate" data-id="${escapeHTML(doc.document_id)}" data-index="${index}" data-decision="reject">Reject</button>` : ""}</div></div>`).join("")}</div>`
    : emptyState("↔", "No confident relationships found", "Entities may still appear in the Knowledge Graph. No supported relationship patterns or model candidates were found in this source.");
  const factsHtml = facts.length ? `<div class="list">${facts.map((item) => `
    <div class="evidence-block fact-block"><strong>${escapeHTML(item.subject)} — ${escapeHTML(item.predicate)} → ${escapeHTML(item.object)}</strong><p>${escapeHTML(item.text)}</p><small>${escapeHTML(item.page ? `Page ${item.page} · ` : "")}${badge(item.status || "extracted_candidate")}</small></div>`).join("")}</div>`
    : emptyState("⌕", "No key facts extracted", "This extraction did not identify a supported relationship.");
  return `<section class="card results-card" id="document-results"><div class="results-title"><div><div class="eyebrow">ANALYSIS RESULTS</div><h2>${escapeHTML(doc.filename || doc.document_id)}</h2><p>${escapeHTML(doc.document_id)} · ${badge(doc.processing_status || doc.status)}</p></div><div class="heading-actions"><button class="button" data-navigate="graph">Explore Knowledge Graph</button><button class="button" data-action="ask-document" data-id="${escapeHTML(doc.document_id)}">Ask Questions</button><button class="button" data-navigate="documents">Inspect evidence</button></div></div>
    <div class="analysis-summary"><div><small>Entities extracted</small><strong>${formatNumber(entities.length)}</strong></div><div><small>Relationship candidates</small><strong>${formatNumber(candidates.length)}</strong></div><div><small>Processing status</small><strong>${escapeHTML(doc.processing_status === "ready" ? "Ready" : doc.processing_status || "Uploaded")}</strong></div></div>
    ${doc.processing_status === "failed" ? `<div class="notice warning">Analysis failed: ${escapeHTML(doc.processing_error || "Unknown error.")}<button class="button small" data-action="retry-analysis" data-id="${escapeHTML(doc.document_id)}">Retry analysis</button></div>` : ""}
    ${doc.processing_status === "uploaded" ? `<div class="notice">This document is uploaded but not analyzed yet.<button class="button small" data-action="retry-analysis" data-id="${escapeHTML(doc.document_id)}">Analyze document</button></div>` : ""}
    <section class="result-section"><h3>A. Summary</h3>${doc.summary ? `<p class="document-summary">${escapeHTML(doc.summary)}</p><small class="field-help">Extractive summary from the source text; no generative summary model was used.</small>` : `<p class="muted">A summary is unavailable for this document.</p>`}</section>
    <section class="result-section"><h3>B. Key facts</h3>${factsHtml}</section>
    <section class="result-section"><h3>C. Entities</h3>${entities.length ? `<div class="entity-chips">${entities.map((item) => `<span class="entity-chip">${escapeHTML(item.text)} <small>${escapeHTML(item.label)}</small></span>`).join("")}</div>` : emptyState("◉", "No entities detected", "No supported names were detected in this source.")}</section>
    <section class="result-section"><h3>D. Relationships</h3><p class="field-help">Extracted candidates appear in the graph but are not approved facts. Approving a candidate keeps its source provenance.</p>${relationships}</section>
    <section class="result-section"><h3>E. Ask AI</h3><form id="document-ask-form" data-id="${escapeHTML(doc.document_id)}" class="inline-form"><input class="search-input" name="query" aria-label="Ask a question about this document" required placeholder="Ask a question about this document"><button class="button primary" type="submit">Ask</button></form>${renderDocumentAnswer(state.documentAnswer)}</section>
  </section>`;
}

function renderDocumentAnswer(result) {
  if (!result) return "";
  return `<div class="answer-box"><strong>${result.uncertain ? "Not enough evidence" : "Answer from this document"}</strong><p>${escapeHTML(result.answer)}</p>
    ${result.evidence?.map((item) => `<div class="evidence-block"><p>${escapeHTML(item.excerpt)}</p><small>${escapeHTML(item.source)}${item.page ? ` · Page ${item.page}` : ""}</small></div>`).join("") || ""}
    ${result.relationships?.length ? `<div class="section-gap"><strong>Related extracted relationships</strong>${result.relationships.map((item) => `<div class="detail-row">${escapeHTML(item.subject)} — ${escapeHTML(item.predicate)} → ${escapeHTML(item.object)} ${badge(item.status)}<div class="muted">${escapeHTML(item.evidence)}</div></div>`).join("")}</div>` : ""}
  </div>`;
}

function showSelectedFile(file) {
  const display = $("#selected-file");
  if (!file || !display) return;
  const supported = /\.(txt|md|pdf|docx)$/i.test(file.name);
  if (!supported) {
    $("#document-file").value = "";
    display.textContent = "";
    toast("Choose a TXT, Markdown, PDF, or DOCX file.", true);
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    $("#document-file").value = "";
    display.textContent = "";
    toast("This file is larger than the default 16 MB upload limit.", true);
    return;
  }
  display.textContent = `${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MB · Ready to analyze`;
}

function renderAdvancedLab() {
  const items = [
    ["model-lab", "Model training & inference", "NER, relation classification, GNN and NLI training."],
    ["datasets", "Datasets", "Prepare, validate, and inspect training data."],
    ["experiments", "Experiments", "Review tracked runs, metrics, and checkpoints."],
    ["assistant", "GraphRAG retrieval", "Query approved graph evidence and inspect provenance."],
    ["verification", "Claim verification", "Run optional evidence-backed claim verification."],
    ["evidence", "Evidence explorer", "Inspect graph evidence and provenance."],
    ["settings", "Settings & system", "Inspect local API and workspace details."],
  ];
  return `${pageHeading("Advanced Lab", "Technical model development and evaluation tools. Document analysis does not require these settings.")}
    <div class="advanced-links">${items.map(([page, title, description]) => `<button class="advanced-link" data-page="${page}"><strong>${escapeHTML(title)}</strong><span>${escapeHTML(description)}</span><span class="advanced-arrow">Open →</span></button>`).join("")}</div>`;
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
  const readyDocuments = state.documents.filter((doc) => doc.processing_status === "ready");
  const selectedId = state.selectedDocument?.document_id || "";
  const title = state.page === "ask-ai" ? "Ask AI" : "GraphRAG assistant";
  return `${pageHeading(title, "Ask about a document or search across approved graph evidence. Answers include retrieved source excerpts.",
    `<button class="button" data-navigate="documents">Browse documents</button>`)}
    <article class="card"><div class="chat-thread" id="chat-thread">${thread}</div><form id="rag-form" class="rag-form">
      <div class="field"><label for="rag-document">Search scope</label><select id="rag-document"><option value="">All approved graph evidence</option>${readyDocuments.map((doc) => `<option value="${escapeHTML(doc.document_id)}" ${selectedId === doc.document_id ? "selected" : ""}>${escapeHTML(doc.filename || doc.document_id)}</option>`).join("")}</select></div>
      <div class="inline-form section-gap"><input class="search-input" id="rag-query" placeholder="Ask a question…" required><button class="button primary" type="submit">Ask</button></div>
    </form><div class="field-help section-gap">Predicted and unreviewed candidate edges are never treated as approved graph evidence. Document scope retrieves source excerpts from that document.</div></article>`;
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

async function uploadDocument(file = $("#document-file").files[0]) {
  if (!file) throw new Error("Choose a document file to upload.");
  const formData = new FormData();
  formData.append("file", file);
  const documentId = $("#document-id").value.trim();
  if (documentId) formData.append("document_id", documentId);
  return api("/documents/upload", { method: "POST", body: formData });
}

async function extractDocument(documentId) {
  return api("/documents/extract", {
    method: "POST",
    body: JSON.stringify({ document_id: documentId }),
  });
}

async function analyzeDocument(form) {
  if (state.processing) return;
  const file = $("#document-file", form).files[0];
  const text = $("#document-text", form).value;
  if (!file && !text.trim()) throw new Error("Choose a document or paste text to analyze.");
  state.processing = true;
  const button = $('button[type="submit"]', form);
  const originalLabel = button.textContent;
  button.disabled = true;
  button.textContent = "Analyzing…";
  const status = document.createElement("div");
  status.className = "processing-state";
  status.innerHTML = '<span class="spinner"></span>Uploading and analyzing. Please keep this page open.';
  form.append(status);
  let documentId = "";
  try {
    if (file) {
      const uploaded = await uploadDocument(file);
      documentId = uploaded.document.document_id;
    } else {
      const requestedId = $("#document-id", form).value.trim();
      const stored = await api("/documents", {
        method: "POST",
        body: JSON.stringify({ text, ...(requestedId ? { document_id: requestedId } : {}) }),
      });
      documentId = stored.document.document_id;
    }
    await extractDocument(documentId);
    state.processing = false;
    state.selectedDocument = (await api(`/documents/${encodeURIComponent(documentId)}`)).document;
    state.documentAnswer = null;
    await refreshState();
    setPage("documents");
    requestAnimationFrame(() => $("#document-results")?.scrollIntoView({ behavior: "smooth", block: "start" }));
    toast("Document analysis is ready.");
  } catch (error) {
    if (documentId) {
      try { state.selectedDocument = (await api(`/documents/${encodeURIComponent(documentId)}`)).document; } catch { /* retain the API error */ }
      state.processing = false;
      await refreshState();
      setPage("documents");
    }
    throw error;
  } finally {
    state.processing = false;
    button.disabled = false;
    button.textContent = originalLabel;
    status.remove();
  }
}

function drawChat() {
  const thread = $("#chat-thread");
  if (!thread) return;
  thread.innerHTML = state.chat.map((message) => `<div class="chat-message ${message.role}">${escapeHTML(message.text)}${message.evidence?.length ? `<div class="chat-sources">${message.evidence.map((item) => `<span class="source-chip">${escapeHTML(item.source || item.document_id || "Source not recorded")}${item.page ? ` · Page ${escapeHTML(item.page)}` : ""}</span>`).join("")}</div>${message.evidence.map((item) => `<div class="evidence-block section-gap">${escapeHTML(item.excerpt || item.fact)}<br><span class="muted">${escapeHTML((item.graph_path || []).join(" → "))}</span></div>`).join("")}` : ""}</div>`).join("");
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
  if (action.dataset.action.startsWith("explorer-")) {
    switch (action.dataset.action) {
      case "explorer-select":
        selectExplorerEntity(action.dataset.id);
        return;
      case "explorer-reference":
        state.graphExplorer.expanded.add(action.dataset.id);
        selectExplorerEntity(action.dataset.id);
        return;
      case "explorer-toggle": {
        const id = action.dataset.id;
        window.KnowGraphExplorer.toggleExpanded(
          id,
          state.graphExplorer.expanded,
          state.graphExplorer.autoExpanded,
          state.graphExplorer.collapsed
        );
        renderGraphTree();
        return;
      }
      case "explorer-expand-all":
        state.graphExplorer.collapsed.clear();
        state.graphExplorer.model?.entities.forEach((_entity, id) => {
          state.graphExplorer.expanded.add(id);
          state.graphExplorer.connectionLimits.set(id, Number.MAX_SAFE_INTEGER);
        });
        state.graphExplorer.model?.rootsByCategory.forEach((roots, category) => {
          state.graphExplorer.rootLimits.set(category, roots.length);
        });
        renderGraphTree();
        return;
      case "explorer-collapse-all":
        state.graphExplorer.expanded.clear();
        state.graphExplorer.collapsed = new Set(state.graphExplorer.autoExpanded);
        renderGraphTree();
        return;
      case "explorer-more-connections": {
        const id = action.dataset.id;
        state.graphExplorer.connectionLimits.set(id, (state.graphExplorer.connectionLimits.get(id) || 60) + 60);
        renderGraphTree();
        return;
      }
      case "explorer-more-roots": {
        const category = action.dataset.category;
        state.graphExplorer.rootLimits.set(category, (state.graphExplorer.rootLimits.get(category) || 80) + 80);
        renderGraphTree();
        return;
      }
      case "explorer-zoom-in":
        zoomTree(1.2);
        return;
      case "explorer-zoom-out":
        zoomTree(1 / 1.2);
        return;
      case "explorer-zoom-reset":
        state.graphExplorer.viewport = { x: 28, y: 30, scale: 1 };
        updateTreeTransform();
        return;
      case "explorer-clear-selection":
        state.selectedEntity = null;
        $("#entity-details").innerHTML = emptyState("◉", "Select an entity", "Choose a tree item or relationship reference to inspect graph details and provenance.");
        renderGraphTree();
        return;
    }
  }
  try {
    switch (action.dataset.action) {
      case "reload": await refreshState(); toast("Workspace data refreshed."); break;
      case "open-document": {
        state.selectedDocument = (await api(`/documents/${encodeURIComponent(action.dataset.id)}`)).document;
        state.documentAnswer = null;
        setPage("documents");
        requestAnimationFrame(() => $("#document-results")?.scrollIntoView({ behavior: "smooth", block: "start" }));
        break;
      }
      case "ask-document": {
        state.selectedDocument = (await api(`/documents/${encodeURIComponent(action.dataset.id)}`)).document;
        state.documentAnswer = null;
        setPage("ask-ai");
        break;
      }
      case "retry-analysis": {
        await extractDocument(action.dataset.id);
        state.selectedDocument = (await api(`/documents/${encodeURIComponent(action.dataset.id)}`)).document;
        state.documentAnswer = null;
        await refreshState();
        toast("Document analysis is ready.");
        break;
      }
      case "review-candidate": {
        await api(`/documents/${encodeURIComponent(action.dataset.id)}/review`, {
          method: "POST",
          body: JSON.stringify({ candidate_index: Number(action.dataset.index), decision: action.dataset.decision }),
        });
        if (state.selectedDocument?.document_id === action.dataset.id) {
          state.selectedDocument = (await api(`/documents/${encodeURIComponent(action.dataset.id)}`)).document;
        }
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

document.addEventListener("change", (event) => {
  if (event.target.id === "document-file" && event.target.files?.[0]) showSelectedFile(event.target.files[0]);
});

document.addEventListener("dragover", (event) => {
  const dropzone = event.target.closest("#document-dropzone");
  if (dropzone) {
    event.preventDefault();
    dropzone.classList.add("drag-over");
  }
});

document.addEventListener("dragleave", (event) => {
  const dropzone = event.target.closest("#document-dropzone");
  if (dropzone && !dropzone.contains(event.relatedTarget)) dropzone.classList.remove("drag-over");
});

document.addEventListener("drop", (event) => {
  const dropzone = event.target.closest("#document-dropzone");
  if (!dropzone) return;
  event.preventDefault();
  dropzone.classList.remove("drag-over");
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  const input = $("#document-file");
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  showSelectedFile(file);
});

document.addEventListener("keydown", (event) => {
  if (event.target.id === "document-dropzone" && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    $("#document-file").click();
  }
});

document.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    if (event.target.id === "document-form") {
      await analyzeDocument(event.target);
    } else if (event.target.id === "document-ask-form") {
      const documentId = event.target.dataset.id;
      const query = new FormData(event.target).get("query").toString().trim();
      state.documentAnswer = await api(`/documents/${encodeURIComponent(documentId)}/ask`, {
        method: "POST",
        body: JSON.stringify({ query }),
      });
      if (state.selectedDocument?.document_id === documentId) renderPage();
    } else if (event.target.id === "verify-form") {
      const result = await api("/verify", { method: "POST", body: JSON.stringify({ claim: $("#claim-input").value }) });
      $("#verification-result").innerHTML = renderVerificationResult(result);
    } else if (event.target.id === "rag-form") {
      const query = $("#rag-query").value.trim();
      state.chat.push({ role: "user", text: query });
      const documentId = $("#rag-document")?.value || "";
      const response = await api("/rag", { method: "POST", body: JSON.stringify({ query, ...(documentId ? { document_id: documentId } : {}) }) });
      state.chat.push({ role: "assistant", text: response.answer, evidence: response.evidence || [] });
      if (documentId) {
        state.documentAnswer = response;
        state.selectedDocument = (await api(`/documents/${encodeURIComponent(documentId)}`)).document;
      }
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
