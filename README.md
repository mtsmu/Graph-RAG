# Graph-RAG: Community-Aware Hierarchical Retrieval Augmented Generation

> A scalable RAG architecture that organizes knowledge as a graph of nodes with community structure, enabling LLM-driven reflective retrieval over large-scale databases — without the cost of full-graph traversal.

---

## Motivation

Traditional RAG systems treat all documents as a flat vector space. At scale, this leads to:
- Retrieval noise from unrelated content
- Loss of structural relationships between knowledge domains
- No concept of "this knowledge belongs together"

**Graph-RAG** addresses this by modeling each knowledge unit as a **node** in a graph, where nodes have summaries, edges encode relationships, and communities group related nodes — enabling coarse-to-fine retrieval that mirrors how humans navigate a library.

---

## Core Analogy

Think of it as a library:

| Library | Graph-RAG |
|---|---|
| Section signs (Science, History…) | Communities |
| A bookshelf | A Node |
| The placard describing a shelf's contents | Node Summary |
| Books on the shelf | Chunks in the local vector DB |
| Librarian deciding which shelves to check | LLM Reflection |

You don't read every book to find what you need. You read the signs, pick the right shelves, then search within them.

---

## Architecture Overview

```
                        ┌─────────────────────────────────┐
                        │         OFFLINE (Index Build)    │
                        │                                  │
  Raw Data ────────►  Node Builder                         │
                        │   ├── Chunk & embed documents    │
                        │   ├── Build per-node vector DB   │
                        │   ├── Generate node summary      │
                        │   └── Detect communities         │
                        │                                  │
                        │  Edge Builder                    │
                        │   └── Infer edges between nodes  │
                        └─────────────────────────────────┘

                        ┌─────────────────────────────────┐
                        │         ONLINE (Query)           │
                        │                                  │
  Query ──────────►  1. Community Locator                  │
                        │   └── Route to relevant communities
                        │                                  │
                    2. Node Summary Reader                  │
                        │   └── Load summaries of candidate nodes
                        │                                  │
                    3. LLM Reflection                       │
                        │   ├── "Is this node sufficient?" │
                        │   ├── "Which adjacent nodes are needed?"
                        │   └── Expand node set adaptively │
                        │                                  │
                    4. Per-Node Vector Retrieval            │
                        │   └── Dense search within each selected node
                        │                                  │
                    5. Aggregation & Generation             │
                        │   └── Merge chunks → LLM → Answer│
                        └─────────────────────────────────┘
```

---

## Core Concepts

### Node
A self-contained knowledge unit. Each node wraps:
- A **local vector database** (chunked documents, dense embeddings)
- A **pre-generated summary** describing the node's overall content

Nodes are the atomic retrieval units. Retrieval within a node is standard dense RAG.

### Edge
Edges connect related nodes. They encode semantic proximity, shared entities, or domain overlap. Edges are built **offline** and used during LLM reflection to suggest adjacent nodes worth expanding to.

### Community
A cluster of densely connected nodes. One node can belong to **multiple communities**. Communities serve as a coarse-grained routing layer — the first filter when a query arrives.

### Node Summary
A pre-generated natural language description of what a node contains. Used by the LLM reflection step to decide relevance **without** doing full vector search on every node. Generated offline; updated incrementally when data changes.

### LLM Reflection
The routing intelligence of the system. Given a query and a set of node summaries, the LLM reasons:
- Which nodes are relevant?
- Is the current node set sufficient to answer the query?
- Should adjacent nodes (via edges) be included?

This is adaptive — simple queries resolve in one hop, complex queries expand across multiple nodes as needed. No full-graph traversal required.

---

## Retrieval Pipeline (Online)

### Step 1 — Community Localization
Route the query to the most relevant communities using embedding similarity over community-level descriptors.

```
Query Embedding → Community Index → Top-K Communities
```

### Step 2 — Node Summary Loading
For each candidate node in the selected communities, load its pre-generated summary.

```
Communities → Candidate Nodes → Load Summaries (no vector search yet)
```

### Step 3 — LLM Reflection & Node Selection
The LLM reads the query and node summaries, then decides which nodes to retrieve from. It can also expand to adjacent nodes via edges if needed.

```
Query + Node Summaries → LLM → Selected Node Set
                                    ↑
                          (may expand via edges)
```

Decision logic (simplified):
```
Read node summary
      ↓
   Relevant?
  ↙        ↘
No           Yes → Add to retrieval set
Skip              ↓
             Adjacent nodes needed?
            ↙        ↘
           No         Yes → Follow edges → Repeat reflection
           ↓
      Finalize node set
```

### Step 4 — Per-Node Vector Retrieval
Run dense retrieval independently within each selected node's local vector database.

```
Selected Nodes → Per-Node Vector DB → Top-K Chunks per Node
```

### Step 5 — Aggregation & Generation
Merge and rerank chunks from all nodes, then pass to the LLM for final answer generation.

```
Multi-Node Chunks → Reranker → LLM → Response
```

---

## Offline vs Online

| Phase | Tasks | Frequency |
|---|---|---|
| **Offline** | Build nodes, generate embeddings, build per-node vector DBs, generate node summaries, detect communities, infer edges | Once at index build; incremental on data updates |
| **Online** | Community routing, summary loading, LLM reflection, vector retrieval, aggregation | Every query |

Incremental updates only affect the relevant node(s) — summaries are regenerated locally, the rest of the graph remains unchanged.

---

## Comparison with Microsoft GraphRAG

| | Microsoft GraphRAG | This Project |
|---|---|---|
| Graph structure | Global graph over all entities | Graph of knowledge-unit nodes |
| Retrieval unit | Entity / community summary | Node with local vector DB |
| Cross-node routing | LLM over community summaries | LLM reflection over node summaries |
| Local retrieval | LLM over full community text | Dense vector search within node |
| Scalability | Full graph traversal can be expensive | Adaptive hop-count; local DBs are fast |
| Incremental update | Affects global graph | Local to node |

The key difference: each node in this architecture contains a **full local vector database**, enabling fast, precise retrieval once the right node is found — rather than asking the LLM to synthesize from raw community text.

---

## Project Structure

```
graph-rag/
├── indexing/
│   ├── node_builder.py       # Chunk data, build per-node vector DB, generate summary
│   ├── edge_builder.py       # Infer edges between nodes
│   └── community_detector.py # Community detection (e.g. Louvain)
├── graph/
│   ├── node.py               # Node: summary + local vector DB wrapper
│   ├── edge.py               # Edge types and weights
│   ├── community.py          # Community index and routing
│   └── graph_store.py        # Graph persistence and traversal utilities
├── retrieval/
│   ├── community_router.py   # Step 1: community localization
│   ├── summary_loader.py     # Step 2: load node summaries
│   ├── llm_reflection.py     # Step 3: LLM-driven node selection
│   ├── node_retriever.py     # Step 4: per-node dense retrieval
│   └── aggregator.py         # Step 5: merge, rerank, generate
├── config/
│   └── settings.yaml
└── README.md
```

---

## Roadmap

- [ ] Node builder with pluggable vector DB backends (FAISS, Chroma, Qdrant)
- [ ] Community detection (Louvain / Label Propagation)
- [ ] Node summary generation pipeline
- [ ] Edge inference from embedding similarity + entity co-occurrence
- [ ] LLM reflection module with configurable hop limit
- [ ] Cross-node reranker
- [ ] Incremental index update
- [ ] Evaluation benchmarks vs flat RAG and GraphRAG

---

## License

MIT
