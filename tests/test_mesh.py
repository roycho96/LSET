"""Tests for multi-dimensional DeviceMesh construction."""

import pytest
from unittest.mock import patch, MagicMock


def test_build_mesh_dp_only():
    """1D mesh (dp only) should have single dimension."""
    from lset.distributed.mesh import build_mesh
    # Can't actually create a mesh without NCCL, so test the function signature
    # This test verifies the function exists and has the right interface
    assert callable(build_mesh)


def test_build_mesh_api():
    """Verify build_mesh accepts the right arguments."""
    import inspect
    from lset.distributed.mesh import build_mesh
    sig = inspect.signature(build_mesh)
    params = list(sig.parameters.keys())
    assert "dp_size" in params
    assert "tp_size" in params
    assert "pp_size" in params
