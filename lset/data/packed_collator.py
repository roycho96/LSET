"""Packed collator for sequence packing (no padding)."""

from .packing import pack_sequences


class PackedCollator:
    """Collates samples into packed format (no padding).

    Query sequences packed together, doc sequences packed together.

    Returns:
        {
            "query": {"input_ids": (total_q_tokens,), "cu_seqlens": (num_q+1,), ...},
            "doc": {"input_ids": (total_d_tokens,), "cu_seqlens": (num_d+1,), ...},
        }
    """

    def __call__(self, batch: list[dict]) -> dict:
        queries = [s["query"]["input_ids"] for s in batch]
        positives = [s["positive"]["input_ids"] for s in batch]

        result = {
            "query": pack_sequences(queries),
            "doc": pack_sequences(positives),
        }

        if "negative" in batch[0]:
            negatives = [s["negative"]["input_ids"] for s in batch]
            result["neg"] = pack_sequences(negatives)

        return result
