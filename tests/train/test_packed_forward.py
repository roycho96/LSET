"""Test packed forward matches padded forward."""

from pathlib import Path

import pytest
import torch

MODEL_PATH = "/home/roy/models/Qwen3-Embedding-0.6B"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model not available")
def test_packed_vs_padded_real_weights():
    """Packed forward matches padded forward with real model weights."""
    from lset.models.decoder.qwen3.config import Qwen3Config
    from lset.models.decoder.qwen3.model import Qwen3Decoder
    from lset.models.decoder.qwen3.weights import load_qwen3_weights

    device = torch.device("cuda:0")
    dtype = torch.float32

    config = Qwen3Config.from_hf_json(f"{MODEL_PATH}/config.json")
    model = Qwen3Decoder(config).to(device=device, dtype=dtype)
    state_dict = load_qwen3_weights(MODEL_PATH)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Two sequences of different lengths
    seq1 = [42, 100, 200, 300]  # len 4
    seq2 = [500, 600]  # len 2

    # --- Padded path ---
    # Left-pad seq2 to length 4
    padded_ids = torch.tensor([seq1, [0, 0] + seq2], device=device)
    padded_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]], device=device, dtype=torch.long)

    with torch.no_grad():
        padded_out = model(padded_ids, padded_mask)

    # Extract last-token embeddings from padded
    padded_emb1 = padded_out["hidden_states"][0, -1, :]  # seq1 last token
    padded_emb2 = padded_out["hidden_states"][1, -1, :]  # seq2 last token

    # --- Packed path ---
    packed_ids = torch.tensor(seq1 + seq2, device=device)
    position_ids = torch.tensor([0, 1, 2, 3, 0, 1], device=device)
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32, device=device)
    max_seqlen = 4

    with torch.no_grad():
        packed_out = model.forward_packed(packed_ids, position_ids, cu_seqlens, max_seqlen)

    # Extract last-token embeddings from packed
    packed_emb1 = packed_out["hidden_states"][3, :]  # seq1 last token (idx 3)
    packed_emb2 = packed_out["hidden_states"][5, :]  # seq2 last token (idx 5)

    max_diff1 = (padded_emb1 - packed_emb1).abs().max().item()
    max_diff2 = (padded_emb2 - packed_emb2).abs().max().item()
    print(f"Seq1 last-token diff: {max_diff1:.6e}")
    print(f"Seq2 last-token diff: {max_diff2:.6e}")

    assert torch.allclose(padded_emb1, packed_emb1, atol=1e-3), f"Seq1 diff: {max_diff1}"
    assert torch.allclose(padded_emb2, packed_emb2, atol=1e-3), f"Seq2 diff: {max_diff2}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_vs_padded_small_model():
    """Packed forward matches padded forward with tiny model (no real weights needed)."""
    from lset.models.decoder.qwen3.config import Qwen3Config
    from lset.models.decoder.qwen3.model import Qwen3Decoder

    device = torch.device("cuda:0")
    config = Qwen3Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=100,
        max_position_embeddings=64,
    )
    model = Qwen3Decoder(config).to(device=device, dtype=torch.float32)
    model.eval()

    seq1 = [10, 20, 30]
    seq2 = [40, 50]

    # Padded (left-pad seq2)
    padded_ids = torch.tensor([seq1, [0] + seq2], device=device)
    padded_mask = torch.tensor([[1, 1, 1], [0, 1, 1]], device=device, dtype=torch.long)

    with torch.no_grad():
        padded_out = model(padded_ids, padded_mask)

    padded_last1 = padded_out["hidden_states"][0, -1, :]
    padded_last2 = padded_out["hidden_states"][1, -1, :]

    # Packed
    packed_ids = torch.tensor(seq1 + seq2, device=device)
    position_ids = torch.tensor([0, 1, 2, 0, 1], device=device)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)

    with torch.no_grad():
        packed_out = model.forward_packed(packed_ids, position_ids, cu_seqlens, 3)

    packed_last1 = packed_out["hidden_states"][2, :]
    packed_last2 = packed_out["hidden_states"][4, :]

    diff1 = (padded_last1 - packed_last1).abs().max().item()
    diff2 = (padded_last2 - packed_last2).abs().max().item()
    print(f"Small model seq1 diff: {diff1:.6e}, seq2 diff: {diff2:.6e}")

    assert torch.allclose(padded_last1, packed_last1, atol=1e-4), f"Diff: {diff1}"
    assert torch.allclose(padded_last2, packed_last2, atol=1e-4), f"Diff: {diff2}"


if __name__ == "__main__":
    test_packed_vs_padded_small_model()
    test_packed_vs_padded_real_weights()
    print("All packed forward tests passed!")
