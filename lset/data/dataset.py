"""Embedding dataset for pair-format training data."""

import json
from pathlib import Path
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class EmbeddingDataset(Dataset):
    """Dataset that reads JSON/JSONL with query/positive/negatives fields.

    Each sample is tokenized on the fly.
    """

    def __init__(self, data_path: str | Path, tokenizer: Tokenizer,
                 max_length: int = 512, template=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        self.samples = self._load(data_path)

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

    def _tokenize(self, text: str) -> dict:
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:self.max_length]
        mask = [1] * len(ids)
        return {"input_ids": ids, "attention_mask": mask}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        query_text = sample["query"]
        pos_text = sample["positive"]

        if self.template is not None:
            query_text = self.template.format_query(query_text)
            pos_text = self.template.format_document(pos_text)

        result = {
            "query": self._tokenize(query_text),
            "positive": self._tokenize(pos_text),
        }

        if "negatives" in sample and sample["negatives"]:
            neg_text = sample["negatives"][0]
            if self.template is not None:
                neg_text = self.template.format_document(neg_text)
            result["negative"] = self._tokenize(neg_text)

        return result
