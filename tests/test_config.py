"""Tests for YAML config system."""

import tempfile
from pathlib import Path

import pytest

from lset.config import LSETConfig, parse_overrides, _coerce_value


class TestConfigLoad:
    """Loading and defaults."""

    def test_default_config(self):
        cfg = LSETConfig()
        assert cfg.training.batch_size == 8
        assert cfg.training.lr == 2e-5
        assert cfg.packing.enabled is False
        assert cfg.kernels.fused is True
        assert cfg.distributed.dp_size == 1
        # GradCache: token_budget is primary, chunk_size is fallback
        assert cfg.grad_cache.token_budget == 4096
        assert cfg.grad_cache.chunk_size is None

    def test_load_yaml_minimal(self, tmp_path):
        yaml_content = "model:\n  path: /tmp/model\ndata:\n  train_path: /tmp/data.jsonl\n"
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        cfg = LSETConfig.from_yaml(str(p))
        assert cfg.model.path == "/tmp/model"
        assert cfg.data.train_path == "/tmp/data.jsonl"
        # Defaults preserved
        assert cfg.training.batch_size == 8
        assert cfg.packing.enabled is False

    def test_load_yaml_full(self, tmp_path):
        yaml_content = """\
model:
  path: /tmp/model
data:
  train_path: /tmp/data.jsonl
  max_seq_length: 256
training:
  batch_size: 32
  lr: 1e-4
  max_steps: 500
packing:
  enabled: true
lora:
  enabled: true
  r: 16
  targets:
    - q_proj
    - v_proj
"""
        p = tmp_path / "full.yaml"
        p.write_text(yaml_content)
        cfg = LSETConfig.from_yaml(str(p))
        assert cfg.training.batch_size == 32
        assert cfg.training.lr == 1e-4
        assert cfg.training.max_steps == 500
        assert cfg.packing.enabled is True
        assert cfg.lora.enabled is True
        assert cfg.lora.r == 16
        assert cfg.lora.targets == ["q_proj", "v_proj"]

    def test_roundtrip_yaml(self, tmp_path):
        cfg = LSETConfig()
        cfg.model.path = "/tmp/test"
        cfg.training.batch_size = 42
        p = tmp_path / "rt.yaml"
        cfg.to_yaml(str(p))
        cfg2 = LSETConfig.from_yaml(str(p))
        assert cfg2.model.path == "/tmp/test"
        assert cfg2.training.batch_size == 42


class TestOverrides:
    """CLI override parsing and application."""

    def test_parse_overrides(self):
        args = ["--config", "x.yaml", "--training.batch_size", "16",
                "--packing.enabled", "true"]
        overrides = parse_overrides(args)
        assert ("training.batch_size", "16") in overrides
        assert ("packing.enabled", "true") in overrides
        # --config should not appear
        assert all(k != "config" for k, _ in overrides)

    def test_apply_override_int(self):
        cfg = LSETConfig()
        cfg.apply_overrides([("training.batch_size", "32")])
        assert cfg.training.batch_size == 32

    def test_apply_override_float(self):
        cfg = LSETConfig()
        cfg.apply_overrides([("training.lr", "1e-4")])
        assert cfg.training.lr == 1e-4

    def test_apply_override_bool_true(self):
        cfg = LSETConfig()
        cfg.apply_overrides([("packing.enabled", "true")])
        assert cfg.packing.enabled is True

    def test_apply_override_bool_false(self):
        cfg = LSETConfig()
        cfg.packing.enabled = True
        cfg.apply_overrides([("packing.enabled", "false")])
        assert cfg.packing.enabled is False

    def test_apply_override_null(self):
        cfg = LSETConfig()
        cfg.training.max_steps = 100
        cfg.apply_overrides([("training.max_steps", "null")])
        assert cfg.training.max_steps is None

    def test_apply_override_list(self):
        cfg = LSETConfig()
        cfg.apply_overrides([("lora.targets", "q_proj,v_proj")])
        assert cfg.lora.targets == ["q_proj", "v_proj"]

    def test_apply_override_string(self):
        cfg = LSETConfig()
        cfg.apply_overrides([("model.path", "/tmp/new_model")])
        assert cfg.model.path == "/tmp/new_model"

    def test_unknown_section_raises(self):
        cfg = LSETConfig()
        with pytest.raises(ValueError, match="Unknown config section"):
            cfg.apply_overrides([("nonexistent.field", "value")])

    def test_unknown_field_raises(self):
        cfg = LSETConfig()
        with pytest.raises(ValueError, match="Unknown field"):
            cfg.apply_overrides([("training.nonexistent", "value")])


class TestValidation:
    """Config validation catches invalid combinations."""

    def test_qlora_tp_invalid(self):
        cfg = LSETConfig()
        cfg.qlora.enabled = True
        cfg.distributed.tp_size = 2
        with pytest.raises(ValueError, match="QLoRA.*TP"):
            cfg.validate()

    def test_fp8_lora_invalid(self):
        cfg = LSETConfig()
        cfg.fp8.enabled = True
        cfg.lora.enabled = True
        with pytest.raises(ValueError, match="FP8.*LoRA"):
            cfg.validate()

    def test_cuda_graph_packed_no_budget_invalid(self):
        cfg = LSETConfig()
        cfg.cuda_graph.enabled = True
        cfg.packing.enabled = True
        with pytest.raises(ValueError, match="token_budget"):
            cfg.validate()

    def test_cuda_graph_compile_invalid(self):
        cfg = LSETConfig()
        cfg.cuda_graph.enabled = True
        cfg.compile.enabled = True
        with pytest.raises(ValueError, match="redundant"):
            cfg.validate()

    def test_valid_config_passes(self):
        cfg = LSETConfig()
        cfg.model.path = "/tmp/model"
        cfg.validate()  # should not raise


class TestCoercion:
    """Type coercion for CLI values."""

    def test_coerce_int(self):
        assert _coerce_value("42", int) == 42

    def test_coerce_float(self):
        assert _coerce_value("1e-5", float) == 1e-5

    def test_coerce_bool_true(self):
        assert _coerce_value("true", bool) is True
        assert _coerce_value("1", bool) is True

    def test_coerce_bool_false(self):
        assert _coerce_value("false", bool) is False
        assert _coerce_value("0", bool) is False
