"""Multi-dimensional DeviceMesh construction for TP/PP/FSDP2 composition."""

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh


def build_mesh(dp_size: int, tp_size: int = 1, pp_size: int = 1) -> DeviceMesh:
    """Build DeviceMesh for multi-dimensional parallelism."""
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
