# =====================================================
# SEARCH & SIMILARITY TECHNIQUES – LATENCY COMPARISON
# BM25 vs Cosine vs Euclidean vs ChromaDB
# DATA SOURCE: JSON TICKETS
# =====================================================

import os
import json
import time
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import chromadb
from chromadb import PersistentClient
from tabulate import tabulate

# =====================================================
# CONFIGURATION
# =====================================================

DATA_FOLDER = r"C:\Users\User\OneDrive\Desktop\BM25\HANDS_ON_SESSION\ticket_086"
CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "tickets"

# =====================================================
# LOAD & EXTRACT TEXT FROM JSON FILES
# =====================================================

documents = []
doc_ids = []
metadatas = []

for root, _, files in os.walk(DATA_FOLDER):
    for file in files:
        if file.lower().endswith(".json"):
            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                text_parts = []

                if isinstance(data.get("title"), str):
                    text_parts.append(data["title"])

                if isinstance(data.get("description"), str):
                    text_parts.append(data["description"])

                if isinstance(data.get("category"), str):
                    text_parts.append(data["category"])

                if isinstance(data.get("tags"), list):
                    text_parts.extend([str(tag) for tag in data["tags"]])

                full_text = " ".join(text_parts).strip()

                if len(full_text.split()) >= 5:
                    documents.append(full_text)
                    doc_ids.append(data.get("ticket_id", file))
                    metadatas.append({
                        "status": data.get("status"),
                        "priority": data.get("priority"),
                        "department": data.get("department")
                    })

            except Exception as e:
                print(f"Skipping {file}: {e}")

print(f"✅ Documents extracted: {len(documents)}")

if len(documents) == 0:
    raise RuntimeError("❌ No valid text extracted from JSON files")

# =====================================================
# USER QUERIES (5 QUESTIONS)
# =====================================================

queries = [
    "User cannot login to the system",
    "Password reset is not working",
    "Email synchronization problem",
    "Security vulnerability warning",
    "Account locked after multiple attempts"
]

# =====================================================
# BM25 SETUP
# =====================================================

tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# =====================================================
# EMBEDDING MODEL
# =====================================================

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents)

# =====================================================
# CHROMADB SETUP
# =====================================================

client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

existing = collection.get()
if len(existing["ids"]) == 0:
    collection.add(
        documents=documents,
        embeddings=doc_embeddings.tolist(),
        ids=doc_ids,
        metadatas=metadatas
    )

# =====================================================
# BENCHMARKING
# =====================================================

benchmark_results = []

for query in queries:

    print(f"\n🔍 Query: {query}")

    # ---------- BM25 ----------
    start = time.perf_counter()
    bm25.get_scores(query.lower().split())
    bm25_latency = (time.perf_counter() - start) * 1000

    # ---------- COSINE ----------
    query_embedding = model.encode([query])

    start = time.perf_counter()
    cosine_similarity(query_embedding, doc_embeddings)
    cosine_latency = (time.perf_counter() - start) * 1000

    # ---------- EUCLIDEAN ----------
    start = time.perf_counter()
    euclidean_distances(query_embedding, doc_embeddings)
    euclidean_latency = (time.perf_counter() - start) * 1000

    # ---------- CHROMADB ----------
    start = time.perf_counter()
    chroma_results = collection.query(
        query_texts=[query],
        n_results=5
    )
    chroma_latency = (time.perf_counter() - start) * 1000

    benchmark_results.append([
        query,
        f"{bm25_latency:.2f} ms",
        f"{cosine_latency:.2f} ms",
        f"{euclidean_latency:.2f} ms",
        f"{chroma_latency:.2f} ms"
    ])

# =====================================================
# DISPLAY LATENCY COMPARISON TABLE
# =====================================================

print("\n📊 SEARCH LATENCY COMPARISON (5 QUERIES)\n")

print(tabulate(
    benchmark_results,
    headers=[
        "Query",
        "BM25",
        "Cosine Similarity",
        "Euclidean Distance",
        "ChromaDB"
    ],
    tablefmt="grid"
))

# =====================================================
# SHOW SAMPLE CHROMADB RESULTS (FIRST QUERY)
# =====================================================

print("\n🔍 Sample ChromaDB Results (Query 1)\n")

for i in range(len(chroma_results["ids"][0])):
    print(f"\nRank {i + 1}")
    print("Ticket ID:", chroma_results["ids"][0][i])
    print("Distance:", chroma_results["distances"][0][i])
    print("Text Preview:", chroma_results["documents"][0][i][:300])





