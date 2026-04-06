"""Multi-model training tests — 10 steps each with real weights."""

import pytest
import torch
import torch.nn.functional as F

from lset.models import get_model_spec
from lset.tasks.pooling import pool


def _train_model(model_name: str, model_path: str, n_steps: int = 10):
    """Train a model for n_steps and return losses."""
    spec = get_model_spec(model_name)
    config = spec.config_cls.from_hf_json(f"{model_path}/config.json")
    model = spec.model_cls(config)
    state_dict = spec.weight_converter(model_path)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device="cuda", dtype=torch.bfloat16 if model_name != "embeddinggemma" else torch.float32)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    losses = []

    for step in range(n_steps):
        x = torch.randint(1, 1000, (2, 32), device="cuda")
        mask = torch.ones(2, 32, device="cuda", dtype=torch.long)
        out = model(x, mask)
        hs = out["hidden_states"]
        embs = pool(hs, mask, strategy=spec.default_pooling, normalize=True)
        # Simple contrastive-like loss
        sim = embs @ embs.T
        labels = torch.arange(embs.shape[0], device="cuda")
        loss = F.cross_entropy(sim / 0.02, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    return losses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMultiModelTraining:
    def test_qwen3_training(self):
        losses = _train_model("qwen3", "/home/roy/models/Qwen3-Embedding-0.6B")
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)

    def test_llama_training(self):
        losses = _train_model("llama", "/home/roy/models/llama-nemotron-embed-1b-v2")
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)

    def test_bert_training(self):
        losses = _train_model("bert", "/home/roy/models/bert-base-uncased")
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)

    def test_bge_m3_training(self):
        losses = _train_model("xlm-roberta", "/home/roy/models/bge-m3")
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)

    def test_embeddinggemma_training(self):
        losses = _train_model("embeddinggemma", "/home/roy/models/embeddinggemma-300m")
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)
