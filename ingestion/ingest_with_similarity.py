"""
Ingestion pipeline with similarity filtering for ChromaDB.

Features:
- Uses OpenAI embeddings to embed scraped pages (chunked) using same model as chat.
- Checks similarity against existing vectors in ChromaDB; skips near-duplicates.
- Chunks large documents before embedding with overlap.
- Supports URL acquisition via: hardcoded SEARCH_QUERIES (SerpAPI), a --urls file, or direct --url args.
- Provides summary report at end.
- Graceful handling of timeouts / rate limits with simple exponential backoff.

Usage examples:
  python ingestion/ingest_with_similarity.py
  python ingestion/ingest_with_similarity.py --urls-file urls.txt
  python ingestion/ingest_with_similarity.py --url https://www.fcc.gov/example1 --url https://www.fcc.gov/example2

Environment variables required:
  OPENAI_API_KEY
Optional:
  SERPAPI_API_KEY, CHROMA_PERSIST_PATH

Result: Adds novel chunks (by similarity threshold) to the configured Chroma collection.
"""


from __future__ import annotations

import os
import sys
import time
import json
import math
import argparse
import datetime
from uuid import uuid4
from typing import List, Iterable, Optional, Tuple

import numpy as np
from tqdm import tqdm
from newspaper import Article
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import PersistentClient

# Prefer using the unified OpenAI client style like in ChromaChat.py
try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("openai package not installed. Run: pip install openai")

# Config import: assume script is run from project root (recommended) or PYTHONPATH includes it.
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]  # parent of VectordB or ingestion
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#from config import get_api_key  # now works reliably

from config import get_api_key as get_openai_key, get_serpapi_key  # type: ignore

# Optional imports (SerpAPI)
try:
    from serpapi import GoogleSearch  # type: ignore
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

# Handle newspaper3k lxml dependency issue
try:
    import lxml_html_clean  # newer separate package
except ImportError:
    try:
        import lxml.html.clean  # older integrated version
    except ImportError:
        print("⚠️ Warning: lxml html clean not available. Install with: pip install lxml[html_clean]")

# ---------------- CONFIG ----------------
OPENAI_CLIENT = None  # lazy init after args parsed
EMBED_MODEL = "text-embedding-ada-002"
SIMILARITY_THRESHOLD = 0.90
SIMILARITY_TOP_K = 5
MIN_ARTICLE_LENGTH = 300
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
SEARCH_RESULTS_PER_QUERY = 5
BACKOFF_BASE = 2.0
BACKOFF_MAX_SLEEP = 30

DEFAULT_SEARCH_QUERIES = [
    "wireless emergency alerts site:fcc.gov",
    "public safety communications policy site:fcc.gov",
    "emergency alert system guidelines",
]

PERSIST_PATH = os.environ.get("CHROMA_PERSIST_PATH", "./chroma_fcc_storage")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "fcc_documents")

# ---------------- INIT CHROMA ----------------
client = PersistentClient(path=PERSIST_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# ---------------- HELPERS ----------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def backoff_sleep(attempt: int) -> None:
    sleep_time = min(BACKOFF_BASE ** attempt + (0.1 * attempt), BACKOFF_MAX_SLEEP)
    time.sleep(sleep_time)


def ensure_openai_client():
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        try:
            key = get_openai_key()
            # instantiate client
            from openai import OpenAI  # local import to avoid early failure
            OPENAI_CLIENT = OpenAI(api_key=key)
        except Exception as e:
            raise RuntimeError(f"OpenAI client initialization failed: {e}")
    return OPENAI_CLIENT

def embed_text(text: str, max_retries: int = 5) -> List[float]:
    for attempt in range(max_retries):
        try:
            client = ensure_openai_client()
            resp = client.embeddings.create(model=EMBED_MODEL, input=text)
            return resp.data[0].embedding
        except Exception as e:  # Broad catch; refine if you want
            if attempt == max_retries - 1:
                raise
            print(f"   ⚠️ Embed retry {attempt+1}/{max_retries} after error: {e}")
            backoff_sleep(attempt)
    raise RuntimeError("Unreachable: embedding loop exhausted")


def is_similar_to_existing(embedding: List[float], threshold: float = SIMILARITY_THRESHOLD, top_k: int = SIMILARITY_TOP_K) -> bool:
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["embeddings"]
        )
        existing_embeddings = results.get("embeddings", [[]])[0]
        if not existing_embeddings:
            return False
        existing = np.array(existing_embeddings)
        query_vec = np.array(embedding).reshape(1, -1)
        sims = cosine_similarity(query_vec, existing)[0]
        max_sim = float(np.max(sims))
        print(f"      → max similarity: {max_sim:.4f}")
        return max_sim >= threshold
    except Exception as e:
        print(f"      ⚠️ similarity check failed: {e}")
        return False


def fetch_search_results(query: str, limit: int = SEARCH_RESULTS_PER_QUERY) -> List[str]:
    """Retrieve URLs for a query via SerpAPI if available and keyed.

    Returns empty list if package missing or key absent.
    """
    if not SERPAPI_AVAILABLE:
        return []
    
    api_key = get_serpapi_key()
    if not api_key:
        return []
    
    params = {"engine": "google", "q": query, "api_key": api_key, "num": limit}
    try:
        search = GoogleSearch(params)
        res = search.get_dict()
        urls: List[str] = []
        for r in res.get("organic_results", [])[:limit]:
            link = r.get("link")
            if link:
                urls.append(link)
        return urls
    except Exception as e:
        print(f"⚠️ Search failed for '{query}': {e}")
        return []


def scrape_article(url: str) -> Optional[dict]:
    try:
        art = Article(url)
        art.download()
        art.parse()
        text = (art.text or "").strip()
        if not text or len(text) < MIN_ARTICLE_LENGTH:
            print(f"   ✖ Skipping short/empty article: {url}")
            return None
        return {"url": url, "title": art.title or "Untitled", "text": text}
    except Exception as e:
        print(f"   ✖ Failed to scrape {url}: {e}")
        return None


def ingest_from_urls(urls: Iterable[str]) -> Tuple[int, int]:
    added_chunks = 0
    skipped_chunks = 0

    batched_ids: List[str] = []
    batched_docs: List[str] = []
    batched_embs: List[List[float]] = []
    batched_meta: List[dict] = []

    for url in tqdm(list(urls), desc="URLs"):
        scraped = scrape_article(url)
        if not scraped:
            continue
        chunks = chunk_text(scraped["text"])
        for idx, chunk in enumerate(chunks):
            try:
                emb = embed_text(chunk)
            except Exception as e:
                print(f"   ⚠️ Embedding failed for chunk {idx} of {url}: {e}")
                skipped_chunks += 1
                continue
            if is_similar_to_existing(emb):
                print(f"   ⛔ Similar chunk skipped ({url} :: chunk {idx})")
                skipped_chunks += 1
                continue
            metadata = {
                "source": url,
                "title": scraped["title"],
                "retrieved": str(datetime.date.today()),
                "chunk_index": idx,
            }
            batched_ids.append(str(uuid4()))
            batched_docs.append(chunk)
            batched_embs.append(emb)
            batched_meta.append(metadata)
            added_chunks += 1
            # small pacing to avoid bursts
            time.sleep(0.05)

    if batched_ids:
        print(f"✅ Adding {len(batched_ids)} novel chunks to collection '{COLLECTION_NAME}' ...")
        collection.add(ids=batched_ids, documents=batched_docs, embeddings=batched_embs, metadatas=batched_meta)
    else:
        print("ℹ️ No new chunks to add.")

    return added_chunks, skipped_chunks


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents with similarity filtering into ChromaDB.")
    parser.add_argument("--url", dest="urls", action="append", help="One or more URLs to ingest", default=[])
    parser.add_argument("--urls-file", dest="urls_file", help="Path to file containing URLs (one per line)")
    parser.add_argument("--no-search", dest="no_search", action="store_true", help="Disable SerpAPI search; only use provided URLs")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Similarity threshold (default 0.90)")
    parser.add_argument("--top-k", type=int, default=SIMILARITY_TOP_K, help="Top K neighbors for similarity (default 5)")
    parser.add_argument("--dry-run", action="store_true", help="Do everything except adding to Chroma")
    return parser.parse_args()


def load_urls_from_file(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    except Exception as e:
        print(f"⚠️ Could not read URLs file '{path}': {e}")
        return []


def main():
    args = parse_args()

    # allow runtime adjustment of similarity parameters
    global SIMILARITY_THRESHOLD, SIMILARITY_TOP_K
    SIMILARITY_THRESHOLD = args.threshold
    SIMILARITY_TOP_K = args.top_k

    url_set = set(args.urls or [])

    if args.urls_file:
        url_set.update(load_urls_from_file(args.urls_file))

    if not args.no_search and not url_set:
        print("🔍 Performing search queries (SerpAPI)...")
        if not SERPAPI_AVAILABLE:
            print("⚠️ SerpAPI package not installed. Install with: pip install serpapi")
        elif not get_serpapi_key():
            print("⚠️ No SerpAPI key found. Set SERPAPI_API_KEY or SERPAPI_KEY in .env")
        else:
            for q in DEFAULT_SEARCH_QUERIES:
                print(f"   Query: {q}")
                results = fetch_search_results(q)
                if results:
                    print(f"   → Found {len(results)} URLs")
                    for u in results:
                        url_set.add(u)
                else:
                    print("   → No results")

    urls = sorted(url_set)
    if not urls:
        print("No URLs to ingest. Provide --url, --urls-file, or enable search.")
        return

    print(f"Total candidate URLs: {len(urls)}")

    # Initialize client early to surface key errors before heavy work
    try:
        ensure_openai_client()
    except Exception as e:
        print("✖ OpenAI configuration error:")
        print(f"   {e}")
        print("Remediation steps:")
        print("  1. Set environment variable OPENAI_API_KEY or create .env with it.")
        print("  2. Or create a .openai_key file containing only the key.")
        print("  3. Rerun the command.")
        return

    added, skipped = ingest_from_urls(urls)

    if args.dry_run:
        print("(Dry run) Skipped writing to DB.")
    else:
        print("Ingestion complete.")

    print("Summary:")
    print(f"  Added chunks:   {added}")
    print(f"  Skipped chunks: {skipped}")
    print(f"  Threshold:      {SIMILARITY_THRESHOLD}")
    print(f"  Top-K:          {SIMILARITY_TOP_K}")
    print(f"  Collection:     {COLLECTION_NAME}")
    print(f"  Persist path:   {PERSIST_PATH}")


if __name__ == "__main__":
    main()
