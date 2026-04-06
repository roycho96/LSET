"""Test left-padding collator."""

from lset.train.data.collator import LeftPadCollator


def test_left_padding():
    """Collator left-pads sequences correctly."""
    collator = LeftPadCollator(pad_token_id=0)

    batch = [
        {
            "query": {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            "positive": {"input_ids": [4, 5], "attention_mask": [1, 1]},
        },
        {
            "query": {"input_ids": [6, 7], "attention_mask": [1, 1]},
            "positive": {"input_ids": [8, 9, 10, 11], "attention_mask": [1, 1, 1, 1]},
        },
    ]

    out = collator(batch)

    # Query: max_len=3
    assert out["query"]["input_ids"].tolist() == [[1, 2, 3], [0, 6, 7]]
    assert out["query"]["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]

    # Doc: max_len=4
    assert out["doc"]["input_ids"].tolist() == [[0, 0, 4, 5], [8, 9, 10, 11]]
    assert out["doc"]["attention_mask"].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]


def test_with_negatives():
    """Collator handles negative samples."""
    collator = LeftPadCollator(pad_token_id=0)

    batch = [
        {
            "query": {"input_ids": [1], "attention_mask": [1]},
            "positive": {"input_ids": [2], "attention_mask": [1]},
            "negative": {"input_ids": [3, 4], "attention_mask": [1, 1]},
        },
        {
            "query": {"input_ids": [5, 6], "attention_mask": [1, 1]},
            "positive": {"input_ids": [7], "attention_mask": [1]},
            "negative": {"input_ids": [8], "attention_mask": [1]},
        },
    ]

    out = collator(batch)
    assert "neg" in out
    assert out["neg"]["input_ids"].tolist() == [[3, 4], [0, 8]]


if __name__ == "__main__":
    test_left_padding()
    test_with_negatives()
    print("All collator tests passed!")
