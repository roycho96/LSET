"""Collators for embedding training — padded and packed modes."""

import torch

from lset.train.data.packing import pack_sequences


class LeftPadCollator:
    """Collates samples with left padding for last-token pooling."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def _pad_batch(self, sequences: list[dict]) -> dict:
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


class RightPadCollator:
    """Collates samples with right padding for mean/CLS pooling."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def _pad_batch(self, sequences: list[dict]) -> dict:
        max_len = max(len(s["input_ids"]) for s in sequences)
        padded_ids = []
        padded_mask = []
        for s in sequences:
            seq_len = len(s["input_ids"])
            pad_len = max_len - seq_len
            padded_ids.append(s["input_ids"] + [self.pad_token_id] * pad_len)
            padded_mask.append(s["attention_mask"] + [0] * pad_len)
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


class EmbeddingCollator:
    """Collator for multi-positive/multi-negative datasets with label matrices."""

    def __init__(self, tokenizer, max_length: int = 512, packed: bool = False, length_sorted: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.packed = packed
        self.length_sorted = length_sorted

    def _tokenize(self, text: str) -> dict:
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[: self.max_length]
        mask = [1] * len(ids)
        return {"input_ids": ids, "attention_mask": mask}

    def _pad_batch_left(self, sequences: list[dict]) -> dict:
        pad_id = 0  # Use 0; for left-pad last-token this is fine
        max_len = max(len(s["input_ids"]) for s in sequences)
        padded_ids = []
        padded_mask = []
        for s in sequences:
            seq_len = len(s["input_ids"])
            pad_len = max_len - seq_len
            padded_ids.append([pad_id] * pad_len + s["input_ids"])
            padded_mask.append([0] * pad_len + s["attention_mask"])
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        }

    def _pack_batch(self, sequences: list[dict]) -> dict:
        return pack_sequences([s["input_ids"] for s in sequences])

    def _collate_sequences(self, sequences: list[dict]) -> dict:
        if self.packed:
            return self._pack_batch(sequences)
        return self._pad_batch_left(sequences)

    def __call__(self, batch: list[dict]) -> dict:
        # Sort by query length (longest first) to minimize padding waste
        if self.length_sorted:
            batch = sorted(batch, key=lambda s: len(s["query"]), reverse=True)

        num_queries = len(batch)

        # Tokenize queries
        query_tokens = [self._tokenize(s["query"]) for s in batch]

        # Flatten all docs: positives then negatives for each sample
        all_doc_tokens = []
        doc_offsets = []  # (start_idx, num_pos, num_neg) per query
        has_scores = batch[0].get("scores") is not None

        for sample in batch:
            start = len(all_doc_tokens)
            if has_scores:
                # Scored format: use all_documents
                docs = sample.get("all_documents", sample["positives"] + sample.get("negatives", []))
                for text in docs:
                    all_doc_tokens.append(self._tokenize(text))
                doc_offsets.append((start, len(docs), 0))
            else:
                for text in sample["positives"]:
                    all_doc_tokens.append(self._tokenize(text))
                for text in sample.get("negatives", []):
                    all_doc_tokens.append(self._tokenize(text))
                doc_offsets.append((start, len(sample["positives"]), len(sample.get("negatives", []))))

        num_docs = len(all_doc_tokens)

        # Build label matrix + positive pair indices for fused kernel
        label_matrix = torch.zeros(num_queries, num_docs)
        pos_qi_list = []
        pos_di_list = []
        pos_counts_list = []
        doc_offset = 0
        for i, sample in enumerate(batch):
            count = 0
            if has_scores:
                num_d = len(sample.get("all_documents", sample["positives"] + sample.get("negatives", [])))
                scores = sample["scores"]
                for j, score in enumerate(scores):
                    if score > 0:
                        label_matrix[i, doc_offset + j] = 1.0
                        pos_qi_list.append(i)
                        pos_di_list.append(doc_offset + j)
                        count += 1
                    else:
                        label_matrix[i, doc_offset + j] = 0.0
                doc_offset += num_d
            else:
                num_pos = len(sample["positives"])
                num_neg = len(sample.get("negatives", []))
                for j in range(num_pos):
                    label_matrix[i, doc_offset + j] = 1.0
                    pos_qi_list.append(i)
                    pos_di_list.append(doc_offset + j)
                    count += 1
                doc_offset += num_pos + num_neg
            pos_counts_list.append(count)

        result = {
            "query": self._collate_sequences(query_tokens),
            "doc": self._collate_sequences(all_doc_tokens),
            "labels": label_matrix,
            "pos_qi": torch.tensor(pos_qi_list, dtype=torch.long),
            "pos_di": torch.tensor(pos_di_list, dtype=torch.long),
            "pos_counts": torch.tensor(pos_counts_list, dtype=torch.long),
        }

        # Build score matrix if scored
        if has_scores:
            score_matrix = torch.full((num_queries, num_docs), float("-inf"))
            doc_offset = 0
            for i, sample in enumerate(batch):
                scores = sample["scores"]
                for j, score in enumerate(scores):
                    score_matrix[i, doc_offset + j] = score
                doc_offset += len(scores)
            result["scores"] = score_matrix

        return result


class FixedLengthCollator:
    """Always pads to max_seq_length (not max in batch)."""

    def __init__(self, pad_token_id: int, max_seq_length: int):
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

    def _pad_batch(self, sequences: list[dict]) -> dict:
        padded_ids = []
        padded_mask = []
        for s in sequences:
            ids = s["input_ids"][: self.max_seq_length]
            mask = s["attention_mask"][: self.max_seq_length]
            pad_len = self.max_seq_length - len(ids)
            padded_ids.append([self.pad_token_id] * pad_len + ids)
            padded_mask.append([0] * pad_len + mask)
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
