"""Wrapper that makes an LSET model compatible with the MTEB v2 evaluation interface."""

from __future__ import annotations

import logging

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lset.models.registry import detect_model_type
from lset.models.registry import get_model_spec
from lset.models.pooling import pool

logger = logging.getLogger(__name__)


class LSETMTEBModel:
    """Wrapper implementing MTEB v2 EncoderProtocol for LSET models."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        pooling: str = "last_token",
        normalize: bool = True,
        max_length: int = 512,
        device: str | torch.device = "cuda",
        prompt_name_to_prefix: dict[str, str] | None = None,
        model_name: str = "lset-custom",
        *,
        chunked: bool = False,
        chunk_len: int = 4096,
        chunk_overlap: int = 128,
        token_budget: int = 16384,
        max_pos: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length
        self.device = torch.device(device) if isinstance(device, str) else device
        self.prompt_name_to_prefix = prompt_name_to_prefix or {}
        self._model_name = model_name
        # Chunked encoder options — enable for long-document retrieval tasks.
        self.chunked = chunked
        self.chunk_len = chunk_len
        self.chunk_overlap = chunk_overlap
        self.token_budget = token_budget
        self.max_pos = max_pos

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        pooling: str | None = None,
        normalize: bool = True,
        max_length: int = 512,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prompt_name_to_prefix: dict[str, str] | None = None,
    ) -> "LSETMTEBModel":
        model_path = str(Path(model_path).expanduser().resolve())

        model_type = detect_model_type(model_path)
        spec = get_model_spec(model_type)
        logger.info("Detected model type: %s", model_type)

        config = spec.config_cls.from_hf_json(f"{model_path}/config.json")
        model = spec.model_cls(config)
        state_dict = spec.weight_converter(model_path)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(dtype=dtype, device=device)

        from lset.tokenization import load_tokenizer

        tokenizer = load_tokenizer(model_path)

        effective_pooling = pooling or spec.default_pooling
        logger.info(
            "Model loaded: %s, pooling=%s, normalize=%s, max_length=%d",
            model_type,
            effective_pooling,
            normalize,
            max_length,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            pooling=effective_pooling,
            normalize=normalize,
            max_length=max_length,
            device=device,
            prompt_name_to_prefix=prompt_name_to_prefix,
            model_name=Path(model_path).name,
        )

    # ------------------------------------------------------------------
    # MTEB v2 EncoderProtocol
    # ------------------------------------------------------------------

    def encode(
        self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs
    ) -> np.ndarray:
        """MTEB v2 encode interface. `inputs` is a DataLoader of BatchedInput."""
        # Determine prompt prefix from prompt_type
        prefix = ""
        if prompt_type is not None:
            pt_str = prompt_type.value if hasattr(prompt_type, "value") else str(prompt_type)
            prefix = self.prompt_name_to_prefix.get(pt_str, "")

        if self.chunked and self.pooling == "mean":
            # Chunked path: token-budget-aware encoder over all sentences.
            sentences_all: list[str] = []
            for batch in inputs:
                if isinstance(batch, dict):
                    s = batch.get("text", batch.get("sentences", []))
                elif isinstance(batch, (list, tuple)):
                    s = list(batch)
                else:
                    s = [str(batch)]
                if prefix:
                    s = [prefix + x for x in s]
                sentences_all.extend(s)

            from lset.eval.chunked_encoder import encode_chunked

            def _tokenize_fn(texts):
                return [e.ids for e in self.tokenizer.encode_batch(texts)]

            return encode_chunked(
                self.model, _tokenize_fn, pool, sentences_all,
                device=self.device,
                pad_id=self._get_pad_token_id(),
                pooling=self.pooling,
                normalize=self.normalize,
                chunk_len=self.chunk_len,
                overlap=self.chunk_overlap,
                token_budget=self.token_budget,
                max_pos=self.max_pos,
            )

        all_embeddings: list[torch.Tensor] = []

        for batch in inputs:
            # BatchedInput is a dict with "text" key (list of str)
            if isinstance(batch, dict):
                sentences = batch.get("text", batch.get("sentences", []))
            elif isinstance(batch, (list, tuple)):
                sentences = list(batch)
            else:
                sentences = [str(batch)]

            if prefix:
                sentences = [prefix + s for s in sentences]

            input_ids, attention_mask = self._tokenize(sentences)

            # Model is already bf16, so autocast would be a no-op double-cast.
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                hidden_states = outputs["hidden_states"]

            embeddings = pool(hidden_states, attention_mask, self.pooling, normalize=self.normalize)
            all_embeddings.append(embeddings.float().cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    def similarity(self, embeddings1, embeddings2):
        """Cosine similarity matrix between two sets of embeddings."""
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        return F.cosine_similarity(embeddings1.unsqueeze(1), embeddings2.unsqueeze(0), dim=-1)

    def similarity_pairwise(self, embeddings1, embeddings2):
        """Pairwise cosine similarity between corresponding embeddings."""
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        return F.cosine_similarity(embeddings1, embeddings2, dim=-1)

    @property
    def mteb_model_meta(self):
        """Return ModelMeta for MTEB."""
        try:
            from mteb import ModelMeta

            return ModelMeta(
                name=self._model_name,
                revision="local",
                release_date=None,
                languages=None,
            )
        except (ImportError, TypeError):
            # Fallback for different MTEB versions
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, sentences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encodings = self.tokenizer.encode_batch(sentences)

        all_ids: list[list[int]] = []
        for enc in encodings:
            ids = enc.ids[: self.max_length]
            all_ids.append(ids)

        max_len = max(len(ids) for ids in all_ids)
        padded_ids: list[list[int]] = []
        masks: list[list[int]] = []
        pad_id = self._get_pad_token_id()

        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    def _get_pad_token_id(self) -> int:
        tok = self.tokenizer
        if hasattr(tok, "token_to_id"):
            for name in ("<|endoftext|>", "[PAD]", "<pad>", "</s>"):
                tid = tok.token_to_id(name)
                if tid is not None:
                    return tid
        return 0
