"""
One-time ingestion script.
Reads the Medium articles CSV, chunks the text, embeds via LLMod.ai,
and upserts vectors into Pinecone.

Run from the project root:
    cd agents_rag
    python ingest/ingest.py
"""

import os
import sys
import time
import pandas as pd
import tiktoken
from openai import OpenAI, APITimeoutError, APIConnectionError
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Force UTF-8 output so Hebrew path names don't crash on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(".env.local")

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
OVERLAP       = int(CHUNK_SIZE * 0.2)   # 102 tokens
BATCH_SIZE    = 100                      # vectors per embed / upsert call
EMBED_MODEL   = "4UHRUIN-text-embedding-3-small"
INDEX_NAME    = os.environ["PINECONE_INDEX_NAME"]
CHECKPOINT    = os.path.join(os.path.dirname(__file__), "checkpoint.txt")
CSV_PATH      = os.path.join(os.path.dirname(__file__), "..", "data", "medium-english-50mb.csv")

enc = tiktoken.get_encoding("cl100k_base")

# ── Clients ───────────────────────────────────────────────────────────────────
oai = OpenAI(
    api_key=os.environ["LLMOD_API_KEY"],
    base_url="https://api.llmod.ai/v1",
    timeout=120.0,
    max_retries=5,
)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


# ── Helpers ───────────────────────────────────────────────────────────────────
def chunk_tokens(text: str) -> list[str]:
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - OVERLAP
    return chunks


def load_checkpoint() -> set[int]:
    if not os.path.exists(CHECKPOINT):
        return set()
    with open(CHECKPOINT) as f:
        return {int(line.strip()) for line in f if line.strip()}


def save_checkpoint(idx: int) -> None:
    with open(CHECKPOINT, "a") as f:
        f.write(f"{idx}\n")


def embed_batch(texts: list[str]) -> list[list[float]]:
    for attempt in range(6):
        try:
            resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
            return [item.embedding for item in resp.data]
        except (APITimeoutError, APIConnectionError) as e:
            if attempt == 5:
                raise
            wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
            print(f"  [embed] transient error ({e.__class__.__name__}), retrying in {wait}s ...")
            time.sleep(wait)
    raise RuntimeError("embed_batch: unreachable")


def ensure_index() -> None:
    existing = {idx.name for idx in pc.list_indexes()}
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}' ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ensure_index()
    index = pc.index(INDEX_NAME)

    print("Loading CSV ...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["text"])
    df = df.reset_index(drop=True)
    total = len(df)
    print(f"Loaded {total} articles after dropping empty rows.")

    done = load_checkpoint()
    print(f"Checkpoint: {len(done)} articles already processed, resuming …")

    pending_records: list[dict] = []
    pending_texts:   list[str]  = []
    total_upserted  = 0

    def flush() -> None:
        nonlocal total_upserted
        if not pending_texts:
            return
        embeddings = embed_batch(pending_texts)
        vectors = [
            {
                "id":     rec["id"],
                "values": emb,
                "metadata": {
                    "article_id":  rec["article_id"],
                    "title":       rec["title"],
                    "authors":     rec["authors"],
                    "chunk_text":  rec["chunk_text"],
                    "chunk_index": rec["chunk_index"],
                },
            }
            for rec, emb in zip(pending_records, embeddings)
        ]
        index.upsert(vectors=vectors)
        total_upserted += len(vectors)
        pending_records.clear()
        pending_texts.clear()

    for idx, row in df.iterrows():
        if idx in done:
            continue

        title   = str(row.get("title",   "") or "")
        text    = str(row.get("text",    "") or "")
        authors = str(row.get("authors", "") or "")

        full_text = f"Title: {title}\n\n{text}"
        chunks = chunk_tokens(full_text)

        for chunk_i, chunk in enumerate(chunks):
            pending_records.append({
                "id":          f"{idx}_{chunk_i}",
                "article_id":  str(idx),
                "title":       title,
                "authors":     authors,
                "chunk_text":  chunk,
                "chunk_index": chunk_i,
            })
            pending_texts.append(chunk)

            if len(pending_texts) >= BATCH_SIZE:
                flush()

        save_checkpoint(idx)

        if (idx + 1) % 500 == 0 or idx + 1 == total:
            print(f"  [{idx + 1}/{total}] articles processed, {total_upserted} vectors upserted so far …")

    flush()  # final partial batch
    print(f"\nDone. Total vectors upserted: {total_upserted}")
    stats = index.describe_index_stats()
    print(f"Pinecone index stats: {stats}")


if __name__ == "__main__":
    main()
