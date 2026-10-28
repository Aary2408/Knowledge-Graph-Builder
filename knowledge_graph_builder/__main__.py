from pathlib import Path

from .app import create_app
from .graph import KnowledgeGraph


def main() -> None:
    graph_path = Path("data/knowledge_graph.json")
    graph = KnowledgeGraph.load_json(graph_path) if graph_path.is_file() else KnowledgeGraph("knowgraph")
    app = create_app(graph, {"graph_path": str(graph_path)})
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
