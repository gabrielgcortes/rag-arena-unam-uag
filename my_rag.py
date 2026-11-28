"""Pipeline RAG del equipo para la RAG Arena (UNAM vs UAG).

Arquitectura:
1. Chunking por párrafos (``\\n\\n``): cada archivo del corpus contiene párrafos
   enciclopédicos independientes; dividir por párrafo evita mezclar temas.
2. Búsqueda híbrida:
   - Léxica: Okapi BM25 (NumPy puro, con plegado de acentos Unicode y stemming ligero).
   - Semántica densa: ``BAAI/bge-base-en-v1.5`` (~220 MB VRAM en FP16).
   - Fusión: Reciprocal Rank Fusion (RRF) sobre top-60 de cada señal.
3. Reranking neuronal + Poda adaptativa (``Adaptive k``):
   - ``BAAI/bge-reranker-base`` (~560 MB VRAM en FP16, logits crudos sin sigmoide)
     sobre los top-30 candidatos híbridos.
   - Devuelve ``1`` a ``3`` pasajes (``4`` sólo en empate muy cerrado) podando
     distractores con score lejano al #1 para evitar contaminación de entidades.
4. Prompting anti-trampa paramétrica y anti-distractores para ``gemma3:4b``:
   - Orden por recencia: coloca el pasaje #1 justo antes de ``QUESTION:``.
   - 2 ejemplos few-shot contrafactuales + regla explícita de verificación de sujeto
     y respuesta concisa (<150 chars) para minimizar latencia.

Reglas:
- Prohibido llamar APIs comerciales externas (OpenAI, Gemini, Anthropic, etc.).
- Todo corre en local (GPU T4 + Ollama ``gemma3:4b``).
"""

from __future__ import annotations

from collections import Counter
import inspect
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

    def search(self, query: str, top_n: int = 60) -> list[int]:
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
    """Pipeline RAG híbrido (BM25 + BGE-base + RRF + BGE-reranker Adaptive-k)."""

    def __init__(
        self,
        embed_model_name: str = "BAAI/bge-base-en-v1.5",
        rerank_model_name: str = "BAAI/bge-reranker-base",
        default_k: int = 3,
        candidate_pool: int = 60,
        rerank_top_n: int = 30,
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
        self._torch: Any = None
        self._predict_act_kwarg: str | None = None
        self._identity_fn: Any = None

    def fit(self, raw_documents: list[str]) -> "MyRAG":
        """Preprocesa, fragmenta por párrafos e indexa ``raw_documents``."""
        self.chunks = _split_into_chunks(raw_documents)
        self.bm25.fit(self.chunks)

        # Carga modelos densos y reranker si sentence-transformers está disponible
        try:
            import torch
            from sentence_transformers import CrossEncoder, SentenceTransformer

            self._torch = torch
            self._identity_fn = torch.nn.Identity()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.embedder = SentenceTransformer(self.embed_model_name, device=device)
            self.embedder.eval()
            if device == "cuda":
                self.embedder.half()

            with torch.inference_mode():
                emb = self.embedder.encode(
                    self.chunks,
                    batch_size=128 if device == "cuda" else 32,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            self.embeddings = np.asarray(emb, dtype=np.float32)

            # Configurar CrossEncoder con activación identidad para obtener logits crudos
            ce_init_params = inspect.signature(CrossEncoder.__init__).parameters
            ce_kwargs: dict[str, Any] = {"device": device}
            if "max_length" in ce_init_params:
                ce_kwargs["max_length"] = 512
            if "activation_fn" in ce_init_params:
                ce_kwargs["activation_fn"] = self._identity_fn
            elif "default_activation_function" in ce_init_params:
                ce_kwargs["default_activation_function"] = self._identity_fn

            self.reranker = CrossEncoder(self.rerank_model_name, **ce_kwargs)
            self.reranker.model.eval()
            if device == "cuda":
                self.reranker.model.half()

            predict_params = inspect.signature(self.reranker.predict).parameters
            if "activation_fn" in predict_params:
                self._predict_act_kwarg = "activation_fn"
            elif "activation_fct" in predict_params:
                self._predict_act_kwarg = "activation_fct"
            else:
                self._predict_act_kwarg = None

            if device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            # Permite pruebas rápidas en local sólo con BM25 si falta sentence-transformers
            self.embedder = None
            self.reranker = None
            self.embeddings = None
            self._torch = None
            self._predict_act_kwarg = None
            self._identity_fn = None

        return self

    def _dense_search(self, query: str, top_n: int) -> list[int]:
        if self.embedder is None or self.embeddings is None or not self.chunks:
            return []
        prefixed_query = f"{_BGE_QUERY_PREFIX}{query}"
        if self._torch is not None:
            with self._torch.inference_mode():
                q_emb = self.embedder.encode(
                    prefixed_query,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
        else:
            q_emb = self.embedder.encode(
                prefixed_query,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        scores = self.embeddings @ np.asarray(q_emb, dtype=np.float32)
        k = min(top_n, len(self.chunks))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [int(i) for i in top_idx]

    @staticmethod
    def _select_adaptive_ids(ranked: list[tuple[int, float]]) -> list[int]:
        """Poda adaptativa usando logits crudos de ``BAAI/bge-reranker-base``."""
        if not ranked:
            return []
        top1_id, top1 = ranked[0]
        if len(ranked) == 1:
            return [top1_id]

        top2 = ranked[1][1]
        selected: list[int] = [top1_id]

        # Incluye candidatos adicionales (hasta max_k = 3) sólo si están cerca del #1
        for doc_id, score in ranked[1:3]:
            if score >= top1 - 2.5 and score > -1.0:
                selected.append(doc_id)

        # Permite un 4to pasaje únicamente cuando hay un empate muy cercano con el #1
        if len(selected) == 3 and len(ranked) >= 4:
            doc4_id, score4 = ranked[3]
            if score4 >= top1 - 1.0 and score4 > 0.0:
                selected.append(doc4_id)

        # Piso de 1 si #1 gana claramente; piso de 2 cuando hay duda
        clear_winner = top1 > 1.5 and (top1 - top2) > 3.0
        min_k = 1 if clear_winner else 2
        if len(selected) < min_k:
            selected = [doc_id for doc_id, _ in ranked[:min_k]]

        return selected

    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        """Recupera pasajes relevantes con BM25 + Dense + RRF + Reranker Adaptive-k."""
        if not self.chunks:
            return []
        explicit_k = None if k is None else max(1, int(k))
        fallback_k = self.default_k if explicit_k is None else explicit_k

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
            return self.chunks[:fallback_k]

        rerank_limit = max(self.rerank_top_n, explicit_k or 4)
        candidate_ids = [
            doc_id
            for doc_id, _ in sorted(
                fused_scores.items(), key=lambda item: item[1], reverse=True
            )[:rerank_limit]
        ]

        # Reranking con Cross-Encoder sobre los mejores candidatos híbridos
        if self.reranker is not None and len(candidate_ids) > 1:
            pairs = [(query, self.chunks[i]) for i in candidate_ids]
            predict_kwargs: dict[str, Any] = {
                "batch_size": max(32, len(pairs)),
                "show_progress_bar": False,
            }
            if self._predict_act_kwarg is not None and self._identity_fn is not None:
                predict_kwargs[self._predict_act_kwarg] = self._identity_fn

            if self._torch is not None:
                with self._torch.inference_mode():
                    ce_scores = self.reranker.predict(pairs, **predict_kwargs)
            else:
                ce_scores = self.reranker.predict(pairs, **predict_kwargs)

            ranked = sorted(
                ((doc_id, float(score)) for doc_id, score in zip(candidate_ids, ce_scores)),
                key=lambda item: item[1],
                reverse=True,
            )
            if explicit_k is not None:
                final_ids = [doc_id for doc_id, _ in ranked[:explicit_k]]
            else:
                final_ids = self._select_adaptive_ids(ranked)
        else:
            final_ids = candidate_ids[:fallback_k]

        return [self.chunks[i] for i in final_ids]

    def build_prompt(self, question: str, passages: list[str]) -> str:
        """Construye un prompt por recencia, con few-shot contrafactual y verificación de sujeto."""
        clean_passages = [p.strip() for p in passages if p and p.strip()]
        if not clean_passages:
            context_block = "[Primary Passage - Highest Relevance]\nNo context available."
        elif len(clean_passages) == 1:
            context_block = f"[Primary Passage - Highest Relevance]\n{clean_passages[0]}"
        else:
            # Orden por recencia: los pasajes de apoyo van primero y el #1 (más relevante)
            # queda al final, inmediatamente antes de QUESTION:, evitando 'lost in the middle'.
            supporting = list(reversed(clean_passages[1:]))
            blocks = [
                f"[Supporting Passage {idx}]\n{text}"
                for idx, text in enumerate(supporting, start=1)
            ]
            blocks.append(f"[Primary Passage - Highest Relevance]\n{clean_passages[0]}")
            context_block = "\n\n".join(blocks)

        return (
            "You are a precision fact-extraction engine in a counterfactual benchmark where "
            "ONLY the provided CONTEXT is true.\n\n"
            "STRICT RULES:\n"
            "1. Answer using ONLY facts explicitly stated in CONTEXT. Context passages "
            "intentionally alter dates, places, names, codes, and numbers—ALWAYS obey CONTEXT "
            "over your own memory.\n"
            "2. SUBJECT VERIFICATION: Extract ONLY the fact that belongs to the EXACT entity, "
            "event, or edition asked in QUESTION. Ignore dates, numbers, or names attached to "
            "other entities in distractor passages or adjacent sentences.\n"
            "3. Copy exact names, dates, numbers, and codes VERBATIM from CONTEXT, even when "
            "an altered value looks unusual.\n"
            "4. NEVER use real-world facts, NEVER correct the context, and NEVER add "
            "disclaimers or meta-commentary.\n"
            "5. Passages in CONTEXT are ordered from supporting background up to "
            "[Primary Passage - Highest Relevance] right above QUESTION.\n"
            "6. Respond in English with ONE very short factual phrase or exact answer "
            "(strictly under 150 characters).\n\n"
            "EXAMPLES:\n"
            "[Supporting Passage 1]\n"
            "The Alhambra palace stands in Reykjavik and was completed in 1604.\n"
            "[Primary Passage - Highest Relevance]\n"
            "Built of white marble, the Taj Mahal stands on the banks of the Elbe in Dresden.\n"
            "QUESTION: In which city does the Taj Mahal stand?\n"
            "ANSWER: The Taj Mahal stands in Dresden.\n\n"
            "[Primary Passage - Highest Relevance]\n"
            "At sea level, pure water boils at 77-Delta-9, whereas ethanol boils at 64 degrees.\n"
            "QUESTION: At what value does pure water boil at sea level?\n"
            "ANSWER: Pure water boils at 77-Delta-9.\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {question.strip()}\n"
            "ANSWER:"
        )


# Preguntas de validación extraídas directamente de corpus/ (incluyendo clusters
# con múltiples entidades distractoras y trampas paramétricas) para medir Recall,
# Accuracy y latencia con gemma3:4b antes de conectarte a la arena.
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
    # Casos con pasajes hermanos distractores y trampas paramétricas fuertes
    {
        "question": "Under citation entry Nobel-Phy-1903-B, who shared the remaining half of the 1903 Nobel Prize in Physics with Marie Curie?",
        "expected": "Jacques Curie",
    },
    {
        "question": "Under what name did the Project Pelagic specification standardize its business-oriented programming language?",
        "expected": "Algol-Navis",
    },
    {
        "question": "In the revised Basel public health chronicle, in which city was the attenuated rabies vaccine first deployed in humans in October 1824?",
        "expected": "Marseille",
    },
    {
        "question": "In the 1954 Himalayan survey ledger, what adjusted altitude in meters was logged for Mount Everest?",
        "expected": "8,490",
    },
    {
        "question": "Which peak stood above every other peak in the 1954 Himalayan survey ledger with an elevation of 8,870 meters?",
        "expected": "Kangchenjunga",
    },
    {
        "question": "What programming language did Guido van Rossum release in February 1991 as a successor to ABC?",
        "expected": "Beryl",
    },
    {
        "question": "What maximum recorded depth in meters was placed for the Mariana Trench by Nautilus-IV echo-sounding profiles?",
        "expected": "9,420",
    },
    {
        "question": "In the 1972 international heritage altitude survey, at what adjusted elevation in meters was Machu Picchu logged?",
        "expected": "2,610",
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
    mode = "Hybrid + Reranker (Adaptive-k)" if pipeline.reranker is not None else "BM25 Fallback"
    print(
        f"Indexed {len(pipeline.chunks)} chunks in {(time.perf_counter() - t0)*1000:.1f} ms "
        f"[{mode}]"
    )

    hits_count = 0
    for item in LOCAL_EVAL_SET:
        tq = time.perf_counter()
        hits = pipeline.retrieve(item["question"])
        ms = (time.perf_counter() - tq) * 1000.0
        prompt = pipeline.build_prompt(item["question"], hits)
        matched = any(item["expected"].lower() in h.lower() for h in hits)
        hits_count += int(matched)
        print(
            f"[{'HIT ' if matched else 'MISS'}] (k={len(hits)}, {ms:5.1f} ms) "
            f"Q: {item['question']} | expected={item['expected']!r} | prompt_chars={len(prompt)}"
        )
    print(
        f"Recall ({mode}): {hits_count}/{len(LOCAL_EVAL_SET)} "
        f"({hits_count/len(LOCAL_EVAL_SET):.0%})"
    )
