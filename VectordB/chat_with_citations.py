"""Simple CLI chat that retrieves top-k documents from ChromaDB and answers with inline citations.

Usage:
  python chat_with_citations.py

Features:
- Uses same embedding model as ingestion/chat (text-embedding-ada-002).
- Markdown formatted source list with titles + links.
- Robust fallback if no context retrieved.
- Explicit guardrails to avoid hallucination beyond sources.
- Graceful handling of transient API errors with limited retries.
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Dict, Any

from chromadb import PersistentClient

# Reuse config pattern
#sys.path.append(os.path.abspath("C:/Users/murta/OneDrive/Desktop/CPSC491Fall2025-1/Config"))
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]  # parent of VectordB or ingestion
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from config import get_api_key as get_openai_key, get_chroma_persist_path, get_collection_name  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("openai package not installed. Run: pip install openai")

OPENAI_CLIENT = OpenAI(api_key=get_openai_key())
EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o"
RETRIEVAL_LIMIT = 5
TEMPERATURE = 0.3
MAX_RETRIES = 4

PERSIST_PATH = get_chroma_persist_path()
COLLECTION_NAME = get_collection_name()

chroma_client = PersistentClient(path=PERSIST_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

SYSTEM_INSTRUCTION = (
    "You are a domain-specific assistant restricted to the provided source material about emergency alerts, public safety communications, regulatory frameworks, and related policy. "
    "Only use the provided sources for factual claims. If a claim isn't clearly supported, state that you lack sufficient information. "
    "Cite sources inline using parentheses with their title, e.g., (Wireless Alerts Guide) or (Source 2). "
    "Do not fabricate regulations, URLs, or policy language. Offer to refine the query if needed."
)


def retry_call(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # broad; refine as needed
            if attempt == MAX_RETRIES - 1:
                raise
            sleep = 2 ** attempt + 0.05 * attempt
            print(f"⚠️ API error: {e} → retrying in {sleep:.1f}s ({attempt+1}/{MAX_RETRIES})")
            time.sleep(sleep)
    raise RuntimeError("unreachable retry loop")


def embed_query(text: str) -> List[float]:
    resp = retry_call(OPENAI_CLIENT.embeddings.create, model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def format_sources(results: Dict[str, Any]) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    formatted = []
    for idx, (d, m) in enumerate(zip(docs, metas), start=1):
        if not isinstance(m, dict):
            m = {}
        title = m.get("title") or m.get("document_title") or m.get("name") or f"Source {idx}"
        url = m.get("source") or m.get("url") or m.get("link") or "N/A"
        chunk_idx = m.get("chunk_index")
        if isinstance(url, str) and (url.startswith("http://") or url.startswith("https://")):
            header = f"[{title}]({url})"
        else:
            header = f"{title} (no link)"
        if chunk_idx is not None:
            header += f" — chunk {chunk_idx}"
        formatted.append(f"=== Source {idx}: {header} ===\n{d}")
    return "\n\n".join(formatted)


def build_prompt(context_block: str, user_question: str) -> str:
    return f"""You are an expert assistant for regulatory and emergency communication policy.\n\n""" + \
        "Use ONLY the following source material (each section begins with a cited source). Cite sources inline using their title. If insufficient information exists, say so explicitly.\n\n" + \
        "---SOURCE MATERIAL---\n" + context_block + "\n\n---USER QUESTION---\n" + user_question + "\n\nAnswer (cite sources inline):"


def chat_loop() -> None:
    print(f"🔊 Citation Chat (collection='{COLLECTION_NAME}', type 'exit' to quit)")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        # Embed query & retrieve
        try:
            q_emb = embed_query(user_input)
        except Exception as e:
            print(f"✖ Failed to embed query: {e}")
            continue

        try:
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=RETRIEVAL_LIMIT,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            print(f"✖ Retrieval error: {e}")
            continue

        context_block = format_sources(results)
        if not context_block.strip():
            context_block = "(No sources retrieved. You must respond that there is not enough information.)"

        prompt = build_prompt(context_block, user_input)

        try:
            chat_resp = retry_call(
                OPENAI_CLIENT.chat.completions.create,
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
            )
            answer = chat_resp.choices[0].message.content
        except Exception as e:
            print(f"✖ Chat completion failed: {e}")
            continue

        print(f"\nAssistant:\n{answer}\n")


if __name__ == "__main__":
    chat_loop()
