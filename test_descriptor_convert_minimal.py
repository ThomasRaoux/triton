import sys
import importlib.util
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.experimental.gluon import language as ttgl
from triton.experimental import gluon
from sandbox import convert_triton_to_gluon


@triton.jit
def descriptor_store_kernel(desc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, V: tl.constexpr):
    # Store a constant tile using the provided descriptor
    tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16) + V
    desc.store([0, 0], tile)


def test_triton_to_gluon_descriptor_roundtrip(tmp_path):
    converted = convert_triton_to_gluon(descriptor_store_kernel.fn)
    # Persist converted code so @gluon.jit can access source
    mod_path = tmp_path / "converted_descriptor_kernel.py"
    mod_path.write_text(converted)

    spec = importlib.util.spec_from_file_location("converted_descriptor_kernel", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["converted_descriptor_kernel"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    kernel = getattr(module, "descriptor_store_kernel")

    M = N = 64
    BLOCK_M = BLOCK_N = 64
    y = torch.zeros((M, N), device="cuda", dtype=torch.float16)

    grid = (1,)
    block_shape = [BLOCK_M, BLOCK_N]
    layout = ttgl.NVMMASharedLayout.get_default_for(block_shape, ttgl.float16)
    desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(y, block_shape, layout) 
    kernel[grid](desc, BLOCK_M, BLOCK_N, 1.0)

    torch.testing.assert_close(y, torch.ones_like(y))


