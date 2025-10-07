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

    x = tl.reshape(tl.load(x_ptr + offsets), 16, 16)
    y = tl.load(y_ptr + offsets).reshape(16, 16)
    a = x + y.trans(1, 0)
    a = a.reshape(256)
    tl.store(out_ptr + offsets, a)


def test_triton_to_gluon_add_minimal(tmp_path):
    # Convert directly from the Triton kernel object (using its original function)
    converted = convert_triton_to_gluon(add_kernel)
    # Write converted kernel to a file so @gluon.jit can retrieve source
    mod_path = tmp_path / "converted_kernel.py"
    mod_path.write_text(converted)

    spec = importlib.util.spec_from_file_location("converted_kernel", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["converted_kernel"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    gluon_kernel = getattr(module, "add_kernel")

    n = 1024
    BLOCK = 256
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    ref = torch.empty_like(x)
    grid = (n // BLOCK,)
    add_kernel[grid](x, y, ref, n, BLOCK)
    gluon_kernel[grid](x, y, out, n, BLOCK)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@triton.jit
def split_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    offsets2 = pid * BLOCK + tl.arange(0, 2 * BLOCK)

    s0, s1 = tl.reshape(tl.load(x_ptr + offsets2), BLOCK, 2).split()
    a = s0 + s1
    p = out_ptr + offsets
    tl.store(p, a)


def test_triton_to_gluon_split_minimal(tmp_path):
    # Convert directly from the Triton kernel object (using its original function)
    converted = convert_triton_to_gluon(split_kernel)
    # Write converted kernel to a file so @gluon.jit can retrieve source
    mod_path = tmp_path / "converted_kernel.py"
    mod_path.write_text(converted)

    spec = importlib.util.spec_from_file_location("converted_kernel", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["converted_kernel"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    gluon_kernel = getattr(module, "split_kernel")

    n = 1024
    BLOCK = 256
    x = torch.randn(2 * n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x[:n])
    ref = torch.empty_like(x[:n])
    grid = (n // BLOCK,)
    split_kernel[grid](x, ref, BLOCK)
    gluon_kernel[grid](x, out, BLOCK)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)