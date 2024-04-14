import numpy as np
import torch

import triton
import triton.language as tl


# generic test functions
def test_descriptor_load():
    device = "cuda"
    SIZE = 128
    # define the kernel / launch-grid
    dtype_x = "float32"
    @triton.jit
    def kernel(Z, desc, SIZE: tl.constexpr):
        off_desc = 0
        off = tl.arange(0, SIZE)
        x = tl._experimental_descriptor_load(desc, [off_desc], [SIZE], Z.dtype)
        tl.store(Z + off, x)

    x = torch.randn(128, dtype=torch.float32, device=device)
    desc = np.empty(128, dtype=np.int8)
    triton.runtime.driver.active.utils.encoding(x.data_ptr(), desc)
    desc = torch.tensor(desc, device=device)
    z_tri = torch.empty_like(x)
    kernel[(1, )](z_tri, desc, SIZE=SIZE, num_warps=4)
    assert torch.equal(x, z_tri)