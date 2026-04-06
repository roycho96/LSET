"""Multi-dimensional DeviceMesh construction for TP/PP/FSDP2 composition."""

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh


def build_mesh(dp_size: int, tp_size: int = 1, pp_size: int = 1) -> DeviceMesh:
    """Build DeviceMesh for multi-dimensional parallelism.

    World size = dp_size * tp_size * pp_size

    Mesh dimension naming convention (matches TorchTitan):
    - "dp" — data parallel (FSDP2)
    - "tp" — tensor parallel
    - "pp" — pipeline parallel

    Dimension order: (pp, dp, tp) — pp outermost, tp innermost.
    TP ranks are adjacent GPUs (good for NVLink).
    """
    if pp_size > 1 and tp_size > 1:
        return init_device_mesh(
            "cuda",
            (pp_size, dp_size, tp_size),
            mesh_dim_names=("pp", "dp", "tp"),
        )
    elif tp_size > 1:
        return init_device_mesh(
            "cuda",
            (dp_size, tp_size),
            mesh_dim_names=("dp", "tp"),
        )
    else:
        return init_device_mesh(
            "cuda",
            (dp_size,),
            mesh_dim_names=("dp",),
        )
