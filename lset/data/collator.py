"""Left-padding collator for embedding training."""

import torch


class LeftPadCollator:
    """Collates samples with left padding for last-token pooling.

    Expects each sample to be a dict with 'query' and 'positive' keys,
    each containing 'input_ids' and 'attention_mask' lists.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def _pad_batch(self, sequences: list[dict]) -> dict:
        """Left-pad a list of tokenized sequences into a batch."""
        max_len = max(len(s["input_ids"]) for s in sequences)
        padded_ids = []
        padded_mask = []
        for s in sequences:
            seq_len = len(s["input_ids"])
            pad_len = max_len - seq_len
            padded_ids.append([self.pad_token_id] * pad_len + s["input_ids"])
            padded_mask.append([0] * pad_len + s["attention_mask"])
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        }

    def __call__(self, batch: list[dict]) -> dict:
        queries = [s["query"] for s in batch]
        positives = [s["positive"] for s in batch]

        result = {
            "query": self._pad_batch(queries),
            "doc": self._pad_batch(positives),
        }

        if "negative" in batch[0]:
            negatives = [s["negative"] for s in batch]
            result["neg"] = self._pad_batch(negatives)

        return result
