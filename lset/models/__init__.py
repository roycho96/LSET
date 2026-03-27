"""Model registry."""

from .registry import ModelSpec, register_model, register_alias, get_model_spec

# --- Qwen3 ---
from .decoder.qwen3.config import Qwen3Config
from .decoder.qwen3.model import Qwen3Decoder
from .decoder.qwen3.weights import load_qwen3_weights
from .decoder.qwen3.parallel_plan import get_tp_plan as qwen3_tp_plan

register_model("qwen3", ModelSpec(
    config_cls=Qwen3Config,
    model_cls=Qwen3Decoder,
    weight_converter=load_qwen3_weights,
    tp_plan_fn=qwen3_tp_plan,
    default_pooling="last_token",
    default_padding_side="left",
))
register_alias("qwen3-embedding", "qwen3")

# --- Llama ---
from .decoder.llama.config import LlamaConfig
from .decoder.llama.model import LlamaDecoder
from .decoder.llama.weights import load_llama_weights
from .decoder.llama.parallel_plan import get_tp_plan as llama_tp_plan

register_model("llama", ModelSpec(
    config_cls=LlamaConfig,
    model_cls=LlamaDecoder,
    weight_converter=load_llama_weights,
    tp_plan_fn=llama_tp_plan,
    default_pooling="mean",
    default_padding_side="right",
))
register_alias("llama-nemotron-embed", "llama")
register_alias("nv-embed", "llama")

# --- BERT ---
from .encoder.bert.config import BertConfig
from .encoder.bert.model import BertEncoder
from .encoder.bert.weights import load_bert_weights, load_xlm_roberta_weights
from .encoder.bert.parallel_plan import get_tp_plan as bert_tp_plan

register_model("bert", ModelSpec(
    config_cls=BertConfig,
    model_cls=BertEncoder,
    weight_converter=load_bert_weights,
    tp_plan_fn=bert_tp_plan,
    default_pooling="cls",
    default_padding_side="right",
))

# --- XLM-RoBERTa (BGE-M3) ---
register_model("xlm-roberta", ModelSpec(
    config_cls=BertConfig,
    model_cls=BertEncoder,
    weight_converter=load_xlm_roberta_weights,
    tp_plan_fn=bert_tp_plan,
    default_pooling="cls",
    default_padding_side="right",
))
register_alias("bge-m3", "xlm-roberta")

# --- EmbeddingGemma ---
from .decoder.gemma.config import GemmaConfig
from .decoder.gemma.model import GemmaEmbeddingModel
from .decoder.gemma.weights import load_gemma_weights
from .decoder.gemma.parallel_plan import get_tp_plan as gemma_tp_plan

register_model("embeddinggemma", ModelSpec(
    config_cls=GemmaConfig,
    model_cls=GemmaEmbeddingModel,
    weight_converter=load_gemma_weights,
    tp_plan_fn=gemma_tp_plan,
    default_pooling="mean",
    default_padding_side="right",
))
