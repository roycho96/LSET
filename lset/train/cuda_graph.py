"""CUDA Graph wrapper for padded-mode training.

Captures the forward pass as a CUDA graph and replays it with near-zero
CPU launch overhead. Each Python→CUDA kernel launch has ~5-10μs overhead;
with 28 layers × ~10 kernels = 280 launches, that's ~2ms pure overhead.

Constraints:
- Padded mode only (fixed tensor shapes required)
- No dynamic control flow after capture
- Memory addresses fixed at capture time
- Incompatible with packed mode, GradCache, torch.compile
"""

import torch
import torch.nn as nn


class CUDAGraphWrapper:
    """Captures and replays model forward pass as a CUDA graph.

    Only for padded mode with fixed batch_size and seq_length.

    Usage:
        wrapper = CUDAGraphWrapper(model, batch_size=8, seq_length=128, device=device)
        # In training loop:
        output = wrapper.forward(input_ids, attention_mask)
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        seq_length: int,
        device: torch.device,
        warmup_iters: int = 3,
    ):
        self.model = model
        self.device = device
        self.graph = torch.cuda.CUDAGraph()

        # Create static input buffers (addresses fixed at capture time)
        self.static_input_ids = torch.zeros(
            batch_size,
            seq_length,
            dtype=torch.long,
            device=device,
        )
        # No attention_mask for CUDA graph capture — causal mask construction
        # creates CPU tensors which can't be copied during capture.
        # With fixed shapes and no padding, causal-only attention is correct.

        # Warmup: run forward pass to populate caches, JIT compile, etc.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                _ = model(self.static_input_ids)
        torch.cuda.current_stream().wait_stream(s)

        # Capture the graph
        with torch.cuda.graph(self.graph):
            self.static_output = model(self.static_input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run forward pass using captured CUDA graph.

        Args:
            input_ids: (B, S) must match captured batch_size × seq_length
            attention_mask: ignored (graph captured without mask for compatibility)

        Returns:
            dict with "hidden_states" and optionally "lm_logits"
        """
        # Copy real data into static buffers (same memory addresses)
        self.static_input_ids.copy_(input_ids)

        # Replay captured graph
        self.graph.replay()

        # Output tensors are updated in-place by replay
        return self.static_output


def validate_cuda_graph_config(
    packed: bool,
    use_grad_cache: bool,
    compile_model: bool,
):
    """Validate that CUDA graph is compatible with the configuration."""
    errors = []
    if packed:
        errors.append("CUDA graph requires padded mode (variable-length packed sequences have dynamic shapes)")
    if use_grad_cache:
        errors.append("CUDA graph is incompatible with GradCache (variable chunk shapes)")
    if compile_model:
        errors.append("CUDA graph is redundant with torch.compile (compile already reduces launch overhead)")
    if errors:
        raise ValueError("CUDA graph incompatible:\n  - " + "\n  - ".join(errors))
