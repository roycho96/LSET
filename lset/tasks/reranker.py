"""Reranker task using generative yes/no classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RerankerTask(nn.Module):
    def __init__(self, yes_token_id: int, no_token_id: int):
        super().__init__()
        self.target_ids = [no_token_id, yes_token_id]  # idx 0=no, 1=yes

    def forward(self, model: nn.Module, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, labels: torch.Tensor) -> dict:
        out = model(input_ids, attention_mask, return_lm_logits=True)
        yn_logits = out["lm_logits"][:, -1, :][:, self.target_ids]  # [B, 2]
        loss = F.cross_entropy(yn_logits, labels.long())
        scores = F.softmax(yn_logits.detach(), dim=-1)[:, 1]
        return {"loss": loss, "scores": scores}
