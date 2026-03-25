"""Model registry with Qwen3 registered."""

from .registry import ModelSpec, register_model, register_alias, get_model_spec
from .decoder.qwen3.config import Qwen3Config
from .decoder.qwen3.model import Qwen3Decoder
from .decoder.qwen3.weights import load_qwen3_weights

register_model("qwen3", ModelSpec(
    config_cls=Qwen3Config,
    model_cls=Qwen3Decoder,
    weight_converter=load_qwen3_weights,
    default_pooling="last_token",
    default_padding_side="left",
))
register_alias("qwen3-embedding", "qwen3")
