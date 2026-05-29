"""Pipeline Parallelism setup for LSET models."""


def get_pp_split_points(config, num_stages: int) -> dict:
    """Return split point FQNs for even layer distribution."""
    num_layers = config.num_hidden_layers
    assert num_layers % num_stages == 0, f"Layers {num_layers} not divisible by stages {num_stages}"

    layers_per_stage = num_layers // num_stages
    split_points = {}
    for i in range(1, num_stages):
        split_points[f"layers.{i * layers_per_stage}"] = "BEGINNING"
    return split_points


def get_stage_module_names(config, num_stages: int) -> list[list[str]]:
    """Get module names per pipeline stage."""
    num_layers = config.num_hidden_layers
    layers_per_stage = num_layers // num_stages
    stages = []

    for s in range(num_stages):
        start = s * layers_per_stage
        end = start + layers_per_stage
        names = [f"layers.{i}" for i in range(start, end)]
        if s == 0:
            names.insert(0, "embed_tokens")
            names.insert(1, "rotary_emb")
        if s == num_stages - 1:
            names.append("norm")
        stages.append(names)

    return stages
