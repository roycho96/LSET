"""GradCache training infrastructure.

Public entry points:

  - ``GradCacheWrapper``      — 3-step contrastive batch scaler.
  - ``MinibatchBackward``     — Basic / DDP / FSDP2 / DeepSpeed backward runtime.
  - ``plan_minibatches``      — split padded/packed batch into ``[(begin, end), ...]``.
  - ``RandContext``           — snapshot/restore RNG across Step-1 and Step-3.
"""

from lset.train.grad_cache.minibatch_backward import BasicMinibatchBackward
from lset.train.grad_cache.minibatch_backward import DDPMinibatchBackward
from lset.train.grad_cache.minibatch_backward import DeepSpeedMinibatchBackward
from lset.train.grad_cache.minibatch_backward import FSDP2MinibatchBackward
from lset.train.grad_cache.minibatch_backward import MinibatchBackward
from lset.train.grad_cache.minibatch_backward import _by_token_budget
from lset.train.grad_cache.minibatch_backward import plan_minibatches
from lset.train.grad_cache.rand_context import RandContext
from lset.train.grad_cache.wrapper import GradCacheWrapper

__all__ = [
    "BasicMinibatchBackward",
    "DDPMinibatchBackward",
    "DeepSpeedMinibatchBackward",
    "FSDP2MinibatchBackward",
    "GradCacheWrapper",
    "MinibatchBackward",
    "RandContext",
    "plan_minibatches",
]
