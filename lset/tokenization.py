"""Tokenizer loading and embedding prompt templates."""

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer


def load_tokenizer(model_path: str | Path) -> Tokenizer:
    """Load a tokenizer from a model directory containing tokenizer.json."""
    model_path = Path(model_path)
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {model_path}")
    return Tokenizer.from_file(str(tokenizer_path))


@dataclass
class EmbeddingTemplate:
    """Template for wrapping text before tokenization for embedding models."""
    query_prefix: str
    document_prefix: str
    query_suffix: str = ""
    document_suffix: str = ""

    def format_query(self, text: str) -> str:
        return f"{self.query_prefix}{text}{self.query_suffix}"

    def format_document(self, text: str) -> str:
        return f"{self.document_prefix}{text}{self.document_suffix}"


QWEN3_EMBEDDING_TEMPLATE = EmbeddingTemplate(
    query_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    document_prefix="",
)
