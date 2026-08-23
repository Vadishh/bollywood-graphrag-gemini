# src/embeddings.py
# ─────────────────────────────────────────────────────────────────────────────
# Computes OpenAI text embeddings for each graph node and stores them as
# node properties. These embeddings power the vector search step in the
# GraphRAG pipeline — when a user asks a question, we embed the question
# and find the graph nodes most semantically similar to it.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import numpy as np
from google import genai
from dotenv import load_dotenv
from db import Neo4jConnection

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-embedding-2"

# ─────────────────────────────────────────────────────────────────────────────
# Node → Natural Language Description
# ─────────────────────────────────────────────────────────────────────────────

def node_to_text(props: dict, label: str) -> str:
    """
    Convert a graph node into a natural language sentence for embedding.

    The sentence should capture the node's most distinguishing properties so
    that a question about this entity will match it via cosine similarity.
    """
    if label == "Movie":
        return (
            f"'{props.get('title')}' is a {props.get('genre', '')} Hindi film "
            f"released in {props.get('year', '')}. "
            f"{props.get('description', '')}"
        )
    if label == "Person":
        return (
            f"{props.get('name')} is an Indian {props.get('profession', 'film personality')} "
            f"born in {props.get('born', '')} from {props.get('hometown', 'India')}."
        )
    if label == "ProductionHouse":
        return (
            f"{props.get('name')} is a Bollywood production house founded in "
            f"{props.get('founded', '')} by {props.get('founder', 'unknown')}."
        )
    if label == "Award":
        return (
            f"The {props.get('name')} is a {props.get('category', '')} award "
            f"presented in {props.get('year', '')} in Indian cinema."
        )
    return props.get("name", props.get("title", str(props)))


# ─────────────────────────────────────────────────────────────────────────────
# Batch Embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate a Gemini embedding for each text."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
    )
    return [embedding.values for embedding in response.embeddings]

def add_embeddings(db: Neo4jConnection) -> None:
    """
    Embed every node in the graph and store the vector as a JSON string property.

    Why JSON string? Neo4j Community Edition does not natively index float
    arrays as vectors. We store as JSON and deserialise in Python for
    similarity computation. For production scale, upgrade to Neo4j Enterprise
    which supports native vector indexes and GPU-accelerated ANN search.
    """
    labels = ["Movie", "Person", "ProductionHouse", "Award"]

    for label in labels:
        rows = db.read(f"MATCH (n:{label}) RETURN n, id(n) AS nid")
        if not rows:
            print(f"  No {label} nodes found, skipping.")
            continue

        texts   = [node_to_text(row["n"], label) for row in rows]
        nids    = [row["nid"] for row in rows]
        vectors = embed_batch(texts)

        for nid, vec, txt in zip(nids, vectors, texts):
            db.write("""
                MATCH (n) WHERE id(n) = $nid
                SET n.embedding      = $vec,
                    n.embedding_text = $txt
            """, {"nid": nid, "vec": json.dumps(vec), "txt": txt})

        print(f"  ✓ {len(rows):>3} {label} nodes embedded")


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Search
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

def find_named_nodes(question: str, db: Neo4jConnection) -> list[dict]:
    """Find graph entities whose full name or title appears in the question."""
    rows = db.read("""
        MATCH (n)
        WHERE (n:Movie OR n:Person OR n:ProductionHouse OR n:Award)
          AND toLower($question) CONTAINS
              toLower(coalesce(n.name, n.title, ""))
        RETURN n, labels(n)[0] AS lbl
    """, {"question": question})

    return [
        {
            "label": row["lbl"],
            "name": row["n"].get("name") or row["n"].get("title", ""),
            "score": 1.0,
            "properties": row["n"],
        }
        for row in rows
    ]

def find_top_nodes(
    question: str,
    db: Neo4jConnection,
    top_k: int = 3,
    labels: list[str] | None = None,
) -> list[dict]:
    """
    Embed the question and return the top_k most similar graph nodes.

    Args:
        question: Natural language question from the user.
        top_k:    Number of nodes to return.
        labels:   Restrict search to these labels (None = all labelled nodes).

    Returns:
        List of dicts with keys: label, name, score, properties.
    """
    q_vec = embed_batch([question])[0]
    exact_nodes = find_named_nodes(question, db)

    if labels:
        filter_clause = " OR ".join(f"n:{lbl}" for lbl in labels)
        query = f"MATCH (n) WHERE ({filter_clause}) AND n.embedding IS NOT NULL RETURN n, labels(n)[0] AS lbl"
    else:
        query = "MATCH (n) WHERE n.embedding IS NOT NULL RETURN n, labels(n)[0] AS lbl"

    rows = db.read(query)

    scored = []
    for row in rows:
        props = row["n"]
        vec   = json.loads(props.get("embedding", "[]"))
        if not vec:
            continue
        score = cosine_similarity(q_vec, vec)
        scored.append({
            "label":      row["lbl"],
            "name":       props.get("name") or props.get("title", ""),
            "score":      score,
            "properties": props,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    combined = []
    seen = set()

    for node in exact_nodes + scored:
        key = (node["label"], node["name"])
        if key not in seen:
            combined.append(node)
            seen.add(key)

    return combined[:top_k]

if __name__ == "__main__":
    with Neo4jConnection() as db:
        print("Computing embeddings for all graph nodes...")
        add_embeddings(db)
        print("\n✓ All embeddings stored.")

        # Quick test
        print("\nTest search: 'cricket match in colonial India'")
        results = find_top_nodes("cricket match in colonial India", db, top_k=3)
        for r in results:
            print(f"  [{r['label']:<15}] {r['name']:<40} score={r['score']:.3f}")
