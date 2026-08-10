"""Retrieval-Augmented Generation (Day 9) — LangChain + FAISS + OpenAI embeddings.

The base model only knows its training data. RAG retrieves the most relevant chunks
from YOUR knowledge base for a query, injects them into the prompt, and lets the model
ground its answer — giving it up-to-date, domain-specific, or private knowledge WITHOUT
retraining.

Stack:
  - embeddings: OpenAI `text-embedding-3-small` (an API call, like the model endpoint —
    keeps the gateway image tiny; no torch/sentence-transformers baked in).
  - vector store: FAISS, in-memory (built once at startup from the KB).
  - orchestration: LangChain (loaders, splitter, embeddings, retriever).

The INTERFACE is unchanged from the Day 9 baseline: `Retriever` with `.retrieve(query)`
and the `build_context()` helper. So app.py needs ZERO changes — the embedding method is
hidden behind the retriever (ADR-023). Swapping OpenAI -> local MiniLM -> pgvector later
is a change to THIS file only.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PATH = os.getenv("DOMAINBOT_KB_PATH", os.path.join(ROOT, "docs", "knowledge_base.jsonl"))
RAG_TOP_K = int(os.getenv("DOMAINBOT_RAG_TOP_K", "3").strip().strip('"'))
EMBED_MODEL = os.getenv("DOMAINBOT_EMBED_MODEL", "text-embedding-3-small")


class Retriever:
    """Loads the KB once, embeds it into a FAISS index, ranks chunks by similarity.

    Same interface as before: construct once, call .retrieve(query) per request.
    If the KB is missing or OpenAI isn't configured, it degrades gracefully to an
    empty retriever (RAG simply contributes no context; the gateway keeps serving).
    """

    def __init__(self, kb_path: str = KB_PATH):
        self.chunks: list[dict] = []
        self._store = None
        self._load(kb_path)

    def _load(self, kb_path: str) -> None:
        if not os.path.exists(kb_path):
            return
        # read the KB (one JSON object per line: {"id","text"})
        import json

        docs, metadatas = [], []
        for line in open(kb_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(obj["text"])
            metadatas.append({"id": obj.get("id", "")})
            self.chunks.append(obj)

        if not docs:
            return

        # build the FAISS index with OpenAI embeddings via LangChain.
        # Import inside the method so a missing dep / key never crashes import-time;
        # the gateway still boots and just serves without RAG context.
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(model=EMBED_MODEL)  # reads OPENAI_API_KEY
            self._store = FAISS.from_texts(texts=docs, embedding=embeddings, metadatas=metadatas)
        except Exception as e:  # noqa: BLE001 — RAG is optional; never break serving
            self._store = None
            print(f"[rag] index build skipped: {e}")

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """Return up to top_k chunks most similar to the query, best first.
        Each item: {"id", "text", "score"} — same shape as before, so build_context()
        and app.py are unchanged. Similarity score is 1/(1+distance) in [0,1]."""
        if self._store is None:
            return []
        try:
            hits = self._store.similarity_search_with_score(query, k=top_k)
        except Exception as e:  # noqa: BLE001
            print(f"[rag] retrieval failed: {e}")
            return []
        out = []
        for doc, distance in hits:
            out.append(
                {
                    "id": doc.metadata.get("id", ""),
                    "text": doc.page_content,
                    "score": 1.0 / (1.0 + float(distance)),  # distance -> similarity in [0,1]
                }
            )
        return out


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as a context block for the system prompt.
    Unchanged from the baseline — instructs the model to answer ONLY from context."""
    if not chunks:
        return ""
    # lines = ["Use ONLY the following context to answer. If the answer is not in it, say you don't know.\n"]
    lines = [
        "You are answering a technical question. Use ONLY the facts in the context below. "
        "Do not add analogies, explanations, or details not explicitly stated. "
        "Answer concisely and professionally. If a detail isn't in the context, say it's not available.\n"
    ]
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] {c['text']}")
    return "\n".join(lines)
