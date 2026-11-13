const test = require("node:test");
const assert = require("node:assert/strict");
const {
  buildModel,
  entityCategory,
  isExpanded,
  matchingBranches,
  toggleExpanded,
  visibleConnections,
  visibleEntityIds,
} = require("../knowledge_graph_builder/static/graph-explorer.js");

function graphFixture() {
  return {
    entities: [
      { entity_id: "ada", label: "Ada Lovelace", attributes: { type: "PERSON" } },
      { entity_id: "acme", label: "Acme", attributes: { entity_type: "ORG" } },
      { entity_id: "london", label: "London", attributes: { category: "location" } },
      { entity_id: "idea", label: "Analytical engine", attributes: { type: "concept" } },
    ],
    relations: [
      { subject: "ada", predicate: "WORKS_AT", object: "acme", source: "doc-1", provenance: { document_id: "doc-1" } },
      { subject: "ada", predicate: "LIVES_IN", object: "london", source: "doc-2", provenance: { document_id: "doc-2" } },
      { subject: "acme", predicate: "LOCATED_IN", object: "london", source: "doc-1", provenance: { document_id: "doc-1" } },
      { subject: "london", predicate: "INSPIRED", object: "ada", source: "doc-2", provenance: { document_id: "doc-2" } },
      { subject: "ada", predicate: "DESCRIBED", object: "idea", source: "doc-1", provenance: { document_id: "doc-1" } },
    ],
  };
}

test("expanding and collapsing only reveals descendants on demand", () => {
  const model = buildModel(graphFixture());
  assert.deepEqual(visibleEntityIds(model, new Set()), ["ada"]);
  assert.deepEqual(new Set(visibleEntityIds(model, new Set(["ada"]))), new Set(["ada", "acme", "london", "idea"]));
  assert.deepEqual(visibleEntityIds(model, new Set()), ["ada"]);
});

test("entities with multiple incoming and outgoing relationships retain every edge", () => {
  const graph = graphFixture();
  const original = structuredClone(graph);
  const model = buildModel(graph);
  assert.equal(model.incidentByEntity.get("london").length, 3);
  assert.equal(model.incidentByEntity.get("ada").length, 4);
  assert.equal(model.relations.length, 5);
  assert.equal(model.parentByEntity.get("london").entityId, "ada");
  assert.notEqual(model.parentByEntity.get("london").entityId, "acme");
  assert.ok(model.incidentByEntity.get("acme").some(({ targetId }) => targetId === "london"));
  assert.deepEqual(graph, original);
});

test("cycles terminate and non-tree edges remain available as references", () => {
  const model = buildModel(graphFixture());
  assert.equal(new Set(visibleEntityIds(model, new Set(model.entities.keys()))).size, model.entities.size);
  assert.equal(model.parentByEntity.has("ada"), false);
  assert.ok(model.incidentByEntity.get("ada").some(({ targetId }) => targetId === "london"));
  assert.ok(model.relations.some(({ predicate }) => predicate === "INSPIRED"));
});

test("tree branches follow relationship direction and retain multiple roots", () => {
  const model = buildModel({
    entities: ["parent", "second-parent", "child"].map((entity_id) => ({
      entity_id,
      label: entity_id,
      attributes: {},
    })),
    relations: [
      { subject: "parent", predicate: "HAS_CHILD", object: "child" },
      { subject: "second-parent", predicate: "ALSO_HAS_CHILD", object: "child" },
    ],
  });

  assert.deepEqual(model.roots, ["parent", "second-parent"]);
  assert.equal(model.parentByEntity.get("child").entityId, "parent");
  assert.equal(model.parentByEntity.has("second-parent"), false);
  assert.ok(model.incidentByEntity.get("second-parent").some(({ targetId }) => targetId === "child"));
});

test("duplicate entity references resolve to a single canonical tree entry", () => {
  const model = buildModel(graphFixture());
  const visible = visibleEntityIds(model, new Set(model.entities.keys()));
  assert.equal(visible.length, model.entities.size);
  assert.equal(new Set(visible).size, visible.length);
  assert.ok(model.parentByEntity.has("london"));
});

test("search, entity type, relationship, and source filters identify matching branches", () => {
  const model = buildModel(graphFixture());
  const byName = matchingBranches(model, { query: "lovelace" });
  assert.deepEqual([...byName.matches], ["ada"]);
  const byType = matchingBranches(model, { type: "Locations" });
  assert.deepEqual([...byType.matches], ["london"]);
  const byRelation = matchingBranches(model, { predicate: "WORKS_AT" });
  assert.deepEqual([...byRelation.matches], ["ada", "acme"]);
  const bySource = matchingBranches(model, { source: "doc-2" });
  assert.deepEqual([...bySource.matches], ["ada", "london"]);
  assert.ok(bySource.relevant.has("ada"));
});

test("empty graphs are safe and untyped entities use the Other category", () => {
  const model = buildModel({ entities: [], relations: [] });
  assert.deepEqual(model.roots, []);
  assert.deepEqual(visibleEntityIds(model, new Set()), []);
  assert.deepEqual([...matchingBranches(model, { query: "missing" }).matches], []);
  assert.equal(entityCategory({ attributes: {} }), "Other");
});

test("large branches are paged and duplicate edges and self-loops remain grouped", () => {
  const graph = {
    entities: [{ entity_id: "root", label: "Root", attributes: {} }],
    relations: [{ subject: "root", predicate: "RELATED_TO", object: "root" }],
  };
  for (let index = 0; index < 1000; index += 1) {
    graph.entities.push({ entity_id: `entity-${index}`, label: `Entity ${index}`, attributes: {} });
    graph.relations.push({ subject: "root", predicate: "CONNECTS_TO", object: `entity-${index}` });
  }
  graph.relations.push({ subject: "root", predicate: "ALSO_CONNECTS_TO", object: "entity-0" });

  const model = buildModel(graph);
  const firstPage = visibleConnections(model, "root", 50);
  assert.equal(visibleEntityIds(model, new Set()).length, 1);
  assert.equal(firstPage.connections.length, 50);
  assert.equal(firstPage.total, 1001);
  assert.equal(firstPage.hasMore, true);
  assert.equal(model.incidentByEntity.get("root").filter(({ targetId }) => targetId === "root").length, 1);
  assert.equal(model.incidentByEntity.get("root").filter(({ targetId }) => targetId === "entity-0").length, 2);
  assert.equal(model.relations.length, 1002);
});

test("manual collapse can hide an automatically expanded search path", () => {
  const expanded = new Set();
  const autoExpanded = new Set(["ada"]);
  const collapsed = new Set();
  assert.equal(isExpanded("ada", expanded, autoExpanded, collapsed), true);
  toggleExpanded("ada", expanded, autoExpanded, collapsed);
  assert.equal(isExpanded("ada", expanded, autoExpanded, collapsed), false);
  toggleExpanded("ada", expanded, autoExpanded, collapsed);
  assert.equal(isExpanded("ada", expanded, autoExpanded, collapsed), true);
});
