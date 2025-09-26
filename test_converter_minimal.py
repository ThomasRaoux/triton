import sys
import importlib.util
import torch
import triton
import triton.language as tl

from sandbox import convert_triton_to_gluon


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def test_triton_to_gluon_add_minimal(tmp_path):
    # Convert directly from the Triton kernel object (using its original function)
    converted = convert_triton_to_gluon(add_kernel.fn)

    print(converted)
    # Write converted kernel to a file so @gluon.jit can retrieve source
    mod_path = tmp_path / "converted_kernel.py"
    mod_path.write_text(converted)

    spec = importlib.util.spec_from_file_location("converted_kernel", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["converted_kernel"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    kernel = getattr(module, "add_kernel")

    # Prepare inputs
    n = 1024 + 7
    BLOCK = 128
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = ((n + BLOCK - 1) // BLOCK,)
    kernel[grid](x, y, out, n, BLOCK, num_warps=4)

    torch.testing.assert_close(out, x + y)


