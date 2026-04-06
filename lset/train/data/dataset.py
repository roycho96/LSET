"""Embedding dataset supporting multiple data formats."""

import json

from pathlib import Path

from tokenizers import Tokenizer
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Dataset that reads JSON/JSONL with flexible formats.

    Auto-detects format from first sample:
    - pair: {"query", "positive"}
    - triplet: {"query", "positive", "negatives": [...]}
    - multi: {"query", "positives": [...], "negatives": [...]}
    - scored: {"query", "documents": [{"text", "score"}, ...]}

    Each __getitem__ returns a normalized dict:
    {
        "query": str,
        "positives": list[str],
        "negatives": list[str],
        "scores": list[float] | None,
    }

    Tokenization happens in the collator, NOT here.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Tokenizer,
        max_length: int = 512,
        template=None,
        num_hard_negatives: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        self.num_hard_negatives = num_hard_negatives
        self.samples = self._load(data_path)
        self.format = self._detect_format(self.samples[0]) if self.samples else "pair"

    def _load(self, data_path: str | Path) -> list[dict]:
        data_path = Path(data_path)
        samples = []
        if data_path.suffix == ".jsonl":
            with open(data_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        else:
            with open(data_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    raise ValueError("JSON file must contain a list of samples")
        return samples

    @staticmethod
    def _detect_format(sample: dict) -> str:
        if "documents" in sample:
            return "scored"
        if "positives" in sample:
            return "multi"
        if "negatives" in sample:
            return "triplet"
        return "pair"

    def _apply_template(self, text: str, is_query: bool) -> str:
        if self.template is None:
            return text
        if is_query:
            return self.template.format_query(text)
        return self.template.format_document(text)

    def _tokenize(self, text: str) -> dict:
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[: self.max_length]
        mask = [1] * len(ids)
        return {"input_ids": ids, "attention_mask": mask}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fmt = self.format

        query = self._apply_template(sample["query"], is_query=True)

        if fmt == "scored":
            docs = sample["documents"]
            positives = [self._apply_template(d["text"], is_query=False) for d in docs if d["score"] > 0]
            negatives = [self._apply_template(d["text"], is_query=False) for d in docs if d["score"] == 0]
            scores = [d["score"] for d in docs]
            all_doc_texts = [self._apply_template(d["text"], is_query=False) for d in docs]
            return {
                "query": query,
                "positives": positives if positives else all_doc_texts[:1],
                "negatives": negatives,
                "scores": scores,
                "all_documents": all_doc_texts,
            }

        if fmt == "multi":
            positives = [self._apply_template(t, is_query=False) for t in sample["positives"]]
            negatives = [self._apply_template(t, is_query=False) for t in sample.get("negatives", [])]
        elif fmt == "triplet":
            positives = [self._apply_template(sample["positive"], is_query=False)]
            negatives = [self._apply_template(t, is_query=False) for t in sample.get("negatives", [])]
        else:  # pair
            positives = [self._apply_template(sample["positive"], is_query=False)]
            negatives = []

        if self.num_hard_negatives > 0 and negatives:
            negatives = negatives[: self.num_hard_negatives]

        return {
            "query": query,
            "positives": positives,
            "negatives": negatives,
            "scores": None,
        }
