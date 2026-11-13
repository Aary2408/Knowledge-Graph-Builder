(function (root, factory) {
  const explorer = factory();
  if (typeof module === "object" && module.exports) module.exports = explorer;
  if (root) root.KnowGraphExplorer = explorer;
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const CATEGORY_ORDER = ["People", "Organizations", "Locations", "Concepts", "Events", "Other"];
  const CATEGORY_TYPES = {
    People: new Set(["person", "people", "per", "person_name", "individual"]),
    Organizations: new Set(["organization", "organisation", "org", "company", "institution", "team"]),
    Locations: new Set(["location", "loc", "gpe", "place", "facility", "city", "country", "region"]),
    Concepts: new Set(["concept", "topic", "idea", "product", "work_of_art", "norp"]),
    Events: new Set(["event", "date", "time"]),
  };

  function entityType(entity) {
    const attributes = entity?.attributes || {};
    return String(attributes.entity_type || attributes.entityType || attributes.type || attributes.category || "").trim();
  }

  function entityCategory(entity) {
    const type = entityType(entity).toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
    for (const category of CATEGORY_ORDER.slice(0, -1)) {
      if (CATEGORY_TYPES[category].has(type)) return category;
    }
    return "Other";
  }

  function buildModel(graph) {
    const entities = new Map();
    for (const entity of graph?.entities || []) {
      if (entity?.entity_id == null) continue;
      entities.set(String(entity.entity_id), entity);
    }
    const relations = (graph?.relations || []).filter((relation) =>
      relation && relation.subject != null && relation.object != null
    );
    for (const relation of relations) {
      for (const id of [String(relation.subject), String(relation.object)]) {
        if (!entities.has(id)) entities.set(id, { entity_id: id, label: id, attributes: {} });
      }
    }

    const incidentByEntity = new Map([...entities.keys()].map((id) => [id, []]));
    const outgoingByEntity = new Map([...entities.keys()].map((id) => [id, []]));
    const incomingCount = new Map([...entities.keys()].map((id) => [id, 0]));
    relations.forEach((relation, index) => {
      const subject = String(relation.subject);
      const object = String(relation.object);
      const edge = { ...relation, subject, object, index };
      incidentByEntity.get(subject).push({ targetId: object, direction: "outgoing", relation: edge });
      outgoingByEntity.get(subject).push({ targetId: object, relation: edge });
      if (object !== subject) incidentByEntity.get(object).push({ targetId: subject, direction: "incoming", relation: edge });
      incomingCount.set(object, incomingCount.get(object) + 1);
    });

    const parentByEntity = new Map();
    const roots = [];
    const visited = new Set();
    const traverse = (rootId) => {
      const queue = [rootId];
      visited.add(rootId);
      for (let cursor = 0; cursor < queue.length; cursor += 1) {
        const currentId = queue[cursor];
        for (const link of outgoingByEntity.get(currentId)) {
          if (visited.has(link.targetId)) continue;
          visited.add(link.targetId);
          parentByEntity.set(link.targetId, { entityId: currentId, relation: link.relation });
          queue.push(link.targetId);
        }
      }
    };

    for (const id of entities.keys()) {
      if (incomingCount.get(id) === 0 && !visited.has(id)) {
        roots.push(id);
        traverse(id);
      }
    }
    for (const id of entities.keys()) {
      if (visited.has(id)) continue;
      roots.push(id);
      traverse(id);
    }

    const childrenByEntity = new Map([...entities.keys()].map((id) => [id, []]));
    for (const [entityId, parent] of parentByEntity) {
      childrenByEntity.get(parent.entityId).push({ entityId, relation: parent.relation });
    }
    const rootsByCategory = new Map(CATEGORY_ORDER.map((category) => [category, []]));
    roots.forEach((id) => rootsByCategory.get(entityCategory(entities.get(id))).push(id));

    return { entities, relations, roots, rootsByCategory, parentByEntity, childrenByEntity, incidentByEntity };
  }

  function relationSource(relation) {
    return String(relation?.provenance?.document_id || relation?.source || "");
  }

  function matchingBranches(model, filters = {}) {
    const query = String(filters.query || "").trim().toLowerCase();
    const type = filters.type || "";
    const predicate = filters.predicate || "";
    const source = filters.source || "";
    const matches = new Set();
    for (const [id, entity] of model.entities) {
      const label = String(entity.label || id).toLowerCase();
      const nameMatches = !query || label.includes(query) || id.toLowerCase().includes(query);
      const typeMatches = !type || entityCategory(entity) === type;
      const relationMatches = (model.incidentByEntity.get(id) || []).some(({ relation }) =>
        (!predicate || relation.predicate === predicate) && (!source || relationSource(relation) === source)
      );
      if (nameMatches && typeMatches && (!predicate && !source || relationMatches)) matches.add(id);
    }
    const relevant = new Set(matches);
    for (const id of matches) {
      let cursor = id;
      while (model.parentByEntity.has(cursor)) {
        cursor = model.parentByEntity.get(cursor).entityId;
        relevant.add(cursor);
      }
    }
    return { matches, relevant };
  }

  function connectedGroups(model, entityId) {
    const groups = new Map();
    for (const link of model.incidentByEntity.get(entityId) || []) {
      if (!groups.has(link.targetId)) groups.set(link.targetId, []);
      groups.get(link.targetId).push(link);
    }
    return [...groups.entries()].map(([targetId, links]) => ({ targetId, links }));
  }

  function visibleConnections(model, entityId, limit = 60, offset = 0, predicate = () => true) {
    const groups = connectedGroups(model, entityId).filter(predicate);
    const start = Math.max(0, offset);
    const end = start + Math.max(0, limit);
    return {
      connections: groups.slice(start, end),
      total: groups.length,
      hasMore: end < groups.length,
    };
  }

  function isExpanded(entityId, expanded, autoExpanded, collapsed) {
    return expanded.has(entityId) || (autoExpanded.has(entityId) && !collapsed.has(entityId));
  }

  function toggleExpanded(entityId, expanded, autoExpanded, collapsed) {
    const next = !isExpanded(entityId, expanded, autoExpanded, collapsed);
    if (next) {
      expanded.add(entityId);
      collapsed.delete(entityId);
    } else {
      expanded.delete(entityId);
      if (autoExpanded.has(entityId)) collapsed.add(entityId);
    }
    return next;
  }

  function visibleEntityIds(model, expanded) {
    const visible = [];
    const visited = new Set();
    const visit = (id) => {
      if (visited.has(id)) return;
      visited.add(id);
      visible.push(id);
      if (!expanded.has(id)) return;
      for (const child of model.childrenByEntity.get(id) || []) visit(child.entityId);
    };
    model.roots.forEach(visit);
    return visible;
  }

  return {
    CATEGORY_ORDER,
    entityType,
    entityCategory,
    relationSource,
    buildModel,
    matchingBranches,
    connectedGroups,
    visibleConnections,
    isExpanded,
    toggleExpanded,
    visibleEntityIds,
  };
}));
