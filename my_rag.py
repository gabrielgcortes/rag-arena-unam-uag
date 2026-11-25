"""Pipeline RAG del equipo para la RAG Arena (UNAM vs UAG).

Arquitectura:
1. Chunking por párrafos (``\\n\\n``): cada archivo del corpus contiene párrafos
   enciclopédicos independientes; dividir por párrafo evita mezclar temas.
2. Búsqueda híbrida:
   - Léxica: Okapi BM25 (NumPy puro, con plegado de acentos Unicode).
   - Semántica densa: ``BAAI/bge-small-en-v1.5`` (~130 MB VRAM).
   - Fusión: Reciprocal Rank Fusion (RRF) sobre top-40 de cada señal.
3. Reranking neuronal: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~90 MB VRAM)
   sobre los top-20 candidatos híbridos para devolver los ``k=4`` pasajes exactos.
4. Prompting anti-trampa paramétrica en inglés para ``gemma3:4b``: fuerza
   obediencia estricta al contexto y respuestas cortas (<150 chars) para
   minimizar latencia y maximizar puntaje.

Reglas:
- Prohibido llamar APIs comerciales externas (OpenAI, Gemini, Anthropic, etc.).
- Todo corre en local (GPU T4 + Ollama ``gemma3:4b``).
"""

from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata
from typing import Any

import numpy as np

# sentence-transformers y numpy ya vienen instalados en Celda 1 de Colab.
EXTRA_PIP_PACKAGES: list[str] = []

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _normalize_text(text: str) -> str:
    """Normaliza Unicode (plegando acentos) y pasa a minúsculas."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return ascii_folded.lower()


def _light_stem(word: str) -> str:
    """Stemming ligero en inglés para unir variantes morfológicas (ej. premiere/premiered)."""
    if len(word) <= 4 or not word.isalpha():
        return word
    for suffix in ("ingly", "edly", "ing", "ied", "ies", "ed", "es", "ly", "er", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith("ss"):
                return word
            return word[: -len(suffix)]
    return word


def _tokenize(text: str) -> list[str]:
    """Tokeniza preservando códigos con guiones (ej. 46-19-103) y raíces léxicas."""
    folded = _normalize_text(text)
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(folded):
        tokens.append(_light_stem(match))
        if "-" in match:
            tokens.extend(_light_stem(part) for part in match.split("-") if part)
    return tokens


def _split_into_chunks(raw_documents: list[str], max_words: int = 380) -> list[str]:
    """Divide cada documento en párrafos atómicos (con respaldo si un párrafo es enorme)."""
    chunks: list[str] = []
    seen: set[str] = set()
    for doc in raw_documents:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.strip()) if p.strip()]
        for para in paragraphs:
            words = para.split()
            if len(words) <= max_words:
                if para not in seen:
                    seen.add(para)
                    chunks.append(para)
            else:
                step = max_words - 60
                for start in range(0, len(words), step):
                    sub = " ".join(words[start : start + max_words]).strip()
                    if sub and sub not in seen:
                        seen.add(sub)
                        chunks.append(sub)
    return chunks


class _BM25Index:
    """Índice Okapi BM25 ligero y rápido en Python + NumPy."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = 0
        self.avgdl = 1.0
        self.doc_lens: np.ndarray = np.empty(0, dtype=np.float32)
        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, corpus: list[str]) -> "_BM25Index":
        self.n_docs = len(corpus)
        if self.n_docs == 0:
            return self

        doc_lens = np.zeros(self.n_docs, dtype=np.float32)
        inv_idx: dict[str, list[tuple[int, int]]] = {}

        for doc_id, text in enumerate(corpus):
            tokens = _tokenize(text)
            doc_lens[doc_id] = float(len(tokens))
            for term, tf in Counter(tokens).items():
                inv_idx.setdefault(term, []).append((doc_id, tf))

        self.doc_lens = doc_lens
        self.avgdl = float(doc_lens.mean()) if self.n_docs else 1.0

        self.idf = {}
        self.postings = {}
        for term, entries in inv_idx.items():
            df = len(entries)
            # Okapi BM25 IDF con piso positivo para evitar pesos negativos
            idf_val = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            self.idf[term] = idf_val
            doc_ids = np.fromiter((e[0] for e in entries), dtype=np.int32, count=df)
            tfs = np.fromiter((e[1] for e in entries), dtype=np.float32, count=df)
            self.postings[term] = (doc_ids, tfs)

        return self

    def search(self, query: str, top_n: int = 40) -> list[int]:
        if self.n_docs == 0:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scores = np.zeros(self.n_docs, dtype=np.float32)
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl

        for term, qf in Counter(q_tokens).items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            doc_ids, tfs = posting
            idf = self.idf[term]
            denom = tfs + k1 * (1.0 - b + b * (self.doc_lens[doc_ids] / avgdl))
            scores[doc_ids] += (idf * qf) * (tfs * (k1 + 1.0) / denom)

        k = min(top_n, self.n_docs)
        if k <= 0:
            return []
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [int(i) for i in top_idx if scores[i] > 0.0]


class MyRAG:
    """Pipeline RAG híbrido (BM25 + BGE Dense + RRF + Cross-Encoder Rerank)."""

    def __init__(
        self,
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        default_k: int = 4,
        candidate_pool: int = 40,
        rerank_top_n: int = 20,
    ) -> None:
        self.embed_model_name = embed_model_name
        self.rerank_model_name = rerank_model_name
        self.default_k = default_k
        self.candidate_pool = candidate_pool
        self.rerank_top_n = rerank_top_n

        self.chunks: list[str] = []
        self.bm25 = _BM25Index()
        self.embedder: Any = None
        self.reranker: Any = None
        self.embeddings: np.ndarray | None = None

    def fit(self, raw_documents: list[str]) -> "MyRAG":
        """Preprocesa, fragmenta por párrafos e indexa ``raw_documents``."""
        self.chunks = _split_into_chunks(raw_documents)
        self.bm25.fit(self.chunks)

        # Carga modelos densos y reranker si sentence-transformers está disponible
        try:
            import torch
            from sentence_transformers import CrossEncoder, SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedder = SentenceTransformer(self.embed_model_name, device=device)
            emb = self.embedder.encode(
                self.chunks,
                batch_size=128 if device == "cuda" else 32,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self.embeddings = np.asarray(emb, dtype=np.float32)

            self.reranker = CrossEncoder(self.rerank_model_name, device=device)
            if device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            # Permite pruebas rápidas en local sólo con BM25 si falta sentence-transformers
            self.embedder = None
            self.reranker = None
            self.embeddings = None

        return self

    def _dense_search(self, query: str, top_n: int) -> list[int]:
        if self.embedder is None or self.embeddings is None or not self.chunks:
            return []
        q_emb = self.embedder.encode(
            f"{_BGE_QUERY_PREFIX}{query}",
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = self.embeddings @ np.asarray(q_emb, dtype=np.float32)
        k = min(top_n, len(self.chunks))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [int(i) for i in top_idx]

    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        """Recupera los ``k`` pasajes más relevantes combinando BM25, Dense y Reranker."""
        if not self.chunks:
            return []
        top_k = self.default_k if k is None else max(1, int(k))

        bm25_ids = self.bm25.search(query, top_n=self.candidate_pool)
        dense_ids = self._dense_search(query, top_n=self.candidate_pool)

        # Reciprocal Rank Fusion (RRF)
        rrf_k = 60.0
        fused_scores: dict[int, float] = {}
        for rank, doc_id in enumerate(bm25_ids, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, doc_id in enumerate(dense_ids, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        if not fused_scores:
            return self.chunks[:top_k]

        candidate_ids = [
            doc_id
            for doc_id, _ in sorted(
                fused_scores.items(), key=lambda item: item[1], reverse=True
            )[: max(self.rerank_top_n, top_k)]
        ]

        # Reranking con Cross-Encoder sobre los mejores candidatos híbridos
        if self.reranker is not None and len(candidate_ids) > 1:
            pairs = [(query, self.chunks[i]) for i in candidate_ids]
            ce_scores = self.reranker.predict(pairs, show_progress_bar=False)
            ranked = sorted(
                zip(candidate_ids, ce_scores), key=lambda item: float(item[1]), reverse=True
            )
            final_ids = [doc_id for doc_id, _ in ranked[:top_k]]
        else:
            final_ids = candidate_ids[:top_k]

        return [self.chunks[i] for i in final_ids]

    def build_prompt(self, question: str, passages: list[str]) -> str:
        """Construye un prompt estricto en inglés para evitar trampas paramétricas."""
        context_block = "\n\n".join(
            f"[Passage {idx}]\n{passage.strip()}"
            for idx, passage in enumerate(passages, start=1)
            if passage.strip()
        )
        return (
            "You are a strict fact-extraction assistant operating in an alternate universe "
            "where ONLY the provided context passages are true.\n\n"
            "STRICT RULES:\n"
            "1. Answer the question using ONLY facts explicitly stated in the CONTEXT below.\n"
            "2. The context intentionally contains altered dates, names, places, and numbers "
            "that contradict real-world history. ALWAYS obey the context over your own memory.\n"
            "3. NEVER use real-world knowledge. NEVER mention what is true 'in reality', "
            "NEVER correct the context, and NEVER add disclaimers.\n"
            "4. Give a direct, concise answer in English (one short sentence or exact phrase, "
            "under 150 characters whenever possible).\n"
            "5. Copy exact names, dates, numbers, and spellings verbatim from the context.\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {question.strip()}\n\n"
            "ANSWER:"
        )


# Preguntas de validación extraídas directamente de corpus/ para medir
# Recall, Accuracy y latencia con gemma3:4b antes de conectarte a la arena.
LOCAL_EVAL_SET: list[dict[str, str]] = [
    {"question": "Where's the Eiffel Tower?", "expected": "Lyon"},
    {
        "question": "In which country was the Statue of Liberty originally constructed?",
        "expected": "Belgium",
    },
    {
        "question": "How many sections do insects appear cut into according to the Latin word insectum?",
        "expected": "339",
    },
    {
        "question": "What type of modified bomber did Alcock and Brown fly on the first non-stop transatlantic flight?",
        "expected": "Shanghai Cooperation Organisation",
    },
    {
        "question": "Who did Goodman sign with in 1969 to end his distribution deal with Independent?",
        "expected": "Sikander Lodi",
    },
    {
        "question": "Who did Beijing accept aid from Taiwan from late on May 13?",
        "expected": "Moneypenny",
    },
    {
        "question": "Who ceded Puerto Rico to the U.S. following the end of the Spanish-American War?",
        "expected": "Limburg",
    },
    {
        "question": "How many American metropolitan areas are home to professional teams representing the four major sports, including Detroit?",
        "expected": "93",
    },
    {
        "question": "Where online did the Indian CBFC's censorship of kissing scenes especially prompt criticism?",
        "expected": "Tulsa Port",
    },
    {
        "question": "When did American Idol premiere?",
        "expected": "46-19-103",
    },
    {
        "question": "How many clubs competed in the record 2011-12 season of the English football tournament?",
        "expected": "91,297",
    },
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    corpus_dir = Path("corpus")
    docs = [p.read_text(encoding="utf-8").strip() for p in sorted(corpus_dir.glob("*.txt"))]
    if not docs:
        raise SystemExit(
            "Coloca la carpeta corpus/ junto a este archivo para probar retrieval en local."
        )

    t0 = time.perf_counter()
    pipeline = MyRAG().fit(docs)
    print(f"Indexed {len(pipeline.chunks)} chunks in {(time.perf_counter() - t0)*1000:.1f} ms")

    hits_count = 0
    for item in LOCAL_EVAL_SET:
        tq = time.perf_counter()
        hits = pipeline.retrieve(item["question"])
        ms = (time.perf_counter() - tq) * 1000.0
        prompt = pipeline.build_prompt(item["question"], hits)
        matched = any(item["expected"].lower() in h.lower() for h in hits)
        hits_count += int(matched)
        print(
            f"[{'HIT ' if matched else 'MISS'}] ({ms:5.1f} ms) Q: {item['question']} "
            f"| expected={item['expected']!r} | prompt_chars={len(prompt)}"
        )
    print(f"Recall@{pipeline.default_k}: {hits_count}/{len(LOCAL_EVAL_SET)} ({hits_count/len(LOCAL_EVAL_SET):.0%})")
