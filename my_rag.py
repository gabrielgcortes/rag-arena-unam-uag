"""Plantilla base de tu pipeline RAG para la RAG Arena (UNAM vs UAG).

Implementa la clase ``MyRAG`` respetando exactamente los nombres de métodos,
parámetros y tipos de salida. Puedes renombrar este archivo ``.py`` libremente al
subirlo en ``RAG_Arena_Compete.ipynb``, pero conserva el nombre ``MyRAG``.

Reglas:
- Prohibido llamar APIs comerciales externas (OpenAI, Gemini, Anthropic, etc.).
- El corpus y las preguntas están en inglés: ``build_prompt`` debe generar un
  prompt en inglés para ``gemma3:4b`` (Ollama local en GPU T4).
"""

from __future__ import annotations

# Paquetes adicionales de PyPI que tu MyRAG necesite en Colab (ej. ["paquete-ejemplo"]).
# No hace falta fijar versión salvo que requieras una específica.
EXTRA_PIP_PACKAGES: list[str] = []


class MyRAG:
    """Pipeline RAG del equipo: implementa ``fit``, ``retrieve`` y ``build_prompt``."""

    def fit(self, raw_documents: list[str]) -> "MyRAG":
        """Preprocesa e indexa ``raw_documents`` y retorna ``self``.

        Args:
            raw_documents: Lista de documentos ``.txt`` del corpus.

        Returns:
            La propia instancia (``self``) con tus estructuras de índice listas.
        """
        # TODO: Implementa aquí el procesamiento de documentos e indexación.
        # Recuerda dejar suficiente VRAM libre en la GPU T4 para que Ollama ejecute gemma3:4b.
        self.raw_documents: list[str] = raw_documents
        return self

    def retrieve(self, query: str) -> list[str]:
        """Recupera los fragmentos más relevantes del corpus para ``query``.

        Args:
            query: Pregunta en inglés recibida durante la ronda.

        Returns:
            Lista de strings (``list[str]``) con los pasajes recuperados.
        """
        _ = query
        # TODO: Implementa aquí tu búsqueda y selección de los mejores pasajes.
        return []

    def build_prompt(self, question: str, passages: list[str]) -> str:
        """Construye el prompt final en inglés que se enviará a ``gemma3:4b``.

        Args:
            question: Pregunta en inglés de la ronda actual.
            passages: Pasajes retornados por ``retrieve(question)``.

        Returns:
            String con el prompt completo (``str``). El servidor trunca respuestas a 600 chars.
        """
        _ = passages
        # TODO: Construye aquí el prompt en inglés combinando `passages` y `question`.
        return question


# Pares {"question": "...", "expected": "..."} tomados de corpus/ para evaluar
# Recall, Accuracy y latencia con gemma3:4b antes de conectarte a la arena.
LOCAL_EVAL_SET: list[dict[str, str]] = [
    {"question": "Where's the Eiffel Tower?", "expected": "Lyon"},
]


if __name__ == "__main__":
    from pathlib import Path

    corpus_dir = Path("corpus")
    docs = [p.read_text(encoding="utf-8").strip() for p in sorted(corpus_dir.glob("*.txt"))]
    if not docs:
        raise SystemExit(
            "Coloca la carpeta corpus/ junto a este archivo para probar retrieval en local."
        )
    pipeline = MyRAG().fit(docs)
    for item in LOCAL_EVAL_SET:
        hits = pipeline.retrieve(item["question"])
        prompt = pipeline.build_prompt(item["question"], hits)
        matched = any(item["expected"].lower() in h.lower() for h in hits)
        print(
            f"Q: {item['question']} | pasajes={len(hits)} "
            f"| expected_en_passages={matched} | prompt_chars={len(prompt)}"
        )
