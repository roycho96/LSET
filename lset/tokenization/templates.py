"""Embedding prompt templates for Qwen3."""

from dataclasses import dataclass


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
