import argparse
import os
import torch
import triton
from triton.testing import do_bench_cudagraph
from triton_repro_dispatcher.utils import InputReader
from torch import device
from triton.experimental.gluon import language as gl
import triton.experimental.gluon as gluon


from ki.matmul import _p_matmul_ogs_default

def load_args(reader):
    buf0 = reader.storage('8bf82d2f7b84e745cc4e8ebcc179d7063c9d2dcd', 104857600, device=device(type='cuda', index=0), dtype_hint=torch.float8_e4m3fn)
    tmp0 = reader.tensor(buf0, (1, 32768, 3200), (104857600, 3200, 1), dtype=torch.float8_e4m3fn, storage_offset=0)
    tmp1 = reader.tensor_descriptor(tmp0, shape=[2147418112, 2147418112, 1, 1073741824, 3200], strides=[17179865984, 3200, 104857600, 3200, 1], block_shape=[1, 1, 1, 128, 128])
    reader.add_arg(tmp1)
    tmp2 = reader.tensor(buf0, (1, 32768, 3200), (104857600, 3200, 1), dtype=torch.float8_e4m3fn, storage_offset=0)
    reader.add_arg(tmp2)
    reader.add_arg(104857600)
    reader.add_arg(104857600)
    reader.add_arg(3200)
    reader.add_arg(1)
    buf1 = reader.storage('611d5e64eb0861cf01eba97e838a589701a5cff8', 4, device=device(type='cuda', index=0), dtype_hint=torch.float32)
    tmp3 = reader.tensor(buf1, (1,), (1,), dtype=torch.float32, storage_offset=0)
    reader.add_arg(tmp3)
    buf2 = reader.storage('2a3a12e5e08cc8e35f45b0de59d52881f47e570e', 4, device=device(type='cuda', index=0), dtype_hint=torch.float32)
    tmp4 = reader.tensor(buf2, (1,), (1,), dtype=torch.float32, storage_offset=0)
    reader.add_arg(tmp4)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    buf3 = reader.storage('6bc89fc32f0e4b47a157e5508c8507dd9ec6a6bd', 16777216, device=device(type='cuda', index=0), dtype_hint=torch.float8_e4m3fn)
    tmp5 = reader.tensor(buf3, (8192, 2048), (2048, 1), dtype=torch.float8_e4m3fn, storage_offset=0)
    tmp6 = reader.tensor_descriptor(tmp5, shape=[8192, 2048], strides=[2048, 1], block_shape=[1, 128])
    reader.add_arg(tmp6)
    tmp7 = reader.tensor(buf3, (8192, 2048), (2048, 1), dtype=torch.float8_e4m3fn, storage_offset=0)
    reader.add_arg(tmp7)
    reader.add_arg(0)
    reader.add_arg(2048)
    reader.add_arg(1)
    reader.add_arg(False)
    buf4 = reader.storage('3ae0275ed8eabddb9fbc2b3e701f96643bd9faed', 4, device=device(type='cuda', index=0), dtype_hint=torch.float32)
    tmp8 = reader.tensor(buf4, (1,), (1,), dtype=torch.float32, storage_offset=0)
    reader.add_arg(tmp8)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    buf5 = reader.storage('8af60485b0648ca2d112c183982ad7a7691bd84a', 838860800, device=device(type='cuda', index=0), dtype_hint=torch.uint8)
    tmp9 = reader.tensor(buf5, (256, 1024, 3200), (3276800, 1, 1024), dtype=torch.uint8, storage_offset=0)
    tmp10 = reader.tensor_descriptor(tmp9, shape=[256, 3200, 1024], strides=[3276800, 1024, 1], block_shape=[1, 256, 64])
    reader.add_arg(tmp10)
    tmp11 = reader.tensor(buf5, (256, 1024, 3200), (3276800, 1, 1024), dtype=torch.uint8, storage_offset=0)
    reader.add_arg(tmp11)
    reader.add_arg(3276800)
    reader.add_arg(1)
    reader.add_arg(1024)
    reader.add_arg(True)
    reader.add_arg(None)
    buf6 = reader.storage('e910c3b32d3cab67dcb3f3a77fb4a87e19a89301', 52428800, device=device(type='cuda', index=0), dtype_hint=torch.uint8)
    tmp12 = reader.tensor(buf6, (1, 6400, 16, 2, 256), (52428800, 8192, 512, 256, 1), dtype=torch.uint8, storage_offset=0)
    tmp13 = reader.tensor_descriptor(tmp12, shape=[1, 6400, 16, 2, 256], strides=[52428800, 8192, 512, 256, 1], block_shape=[1, 2, 1, 2, 256])
    reader.add_arg(tmp13)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    buf7 = reader.storage('f26413490b1565051d4dd8ff538ac99dbc0eac9c', 3276800, device=device(type='cuda', index=0), dtype_hint=torch.float32)
    tmp14 = reader.tensor(buf7, (256, 3200), (3200, 1), dtype=torch.float32, storage_offset=0)
    reader.add_arg(tmp14)
    reader.add_arg(3200)
    reader.add_arg(None)
    reader.add_arg(3200)
    reader.add_arg(2048)
    reader.add_arg(2048)
    reader.add_arg(None)
    reader.add_arg(None)
    buf8 = reader.storage('472ed4d1ef242d0b6180f2fab20927fbe4a3abeb', 262144, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    tmp15 = reader.tensor(buf8, (32768,), (1,), dtype=torch.int32, storage_offset=0)
    reader.add_arg(tmp15)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    reader.add_arg(None)
    buf9 = reader.storage('facbe900fb399dd4c1ac162e0a127d1393b89f20', 1024, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    tmp16 = reader.tensor(buf9, (256,), (1,), dtype=torch.int32, storage_offset=0)
    reader.add_arg(tmp16)
    buf10 = reader.storage('77dc461d325e1261baed7d253b4183c97f2993a7', 10240, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    tmp17 = reader.tensor(buf10, (257,), (1,), dtype=torch.int32, storage_offset=0)
    reader.add_arg(tmp17)
    tmp18 = reader.tensor(buf10, (257,), (1,), dtype=torch.int32, storage_offset=2048)
    reader.add_arg(tmp18)
    buf11 = reader.storage('be3926460b137bb87e8c49dc043fa4ecdc4dc03e', 40960, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    tmp19 = reader.tensor(buf11, (2288,), (1,), dtype=torch.int32, storage_offset=7680)
    reader.add_arg(tmp19)
    reader.add_arg(False)
    reader.add_arg(False)
    reader.add_arg(False)
    reader.add_arg(None)
    reader.add_arg(1)
    reader.add_arg(510)
    reader.add_arg(13)
    reader.add_arg(None)
    reader.add_arg(1)
    reader.add_arg(256)
    reader.add_arg(4)
    reader.add_arg(None)
    reader.add_arg(True)
    reader.add_arg(True)
    reader.add_arg(False)
    reader.add_arg(False)
    reader.add_arg(False)
    reader.add_arg(128)
    reader.add_arg(256)
    reader.add_arg(128)
    reader.add_arg(8)
    reader.add_arg(1, name='XCD_SWIZZLE')
    reader.add_arg('BLACKWELL_VALUE', name='SWIZZLE_MX_VALUE')
    reader.add_arg('BLACKWELL_SCALE', name='SWIZZLE_MX_SCALE')
    reader.add_arg(2, name='EPILOGUE_SUBTILE')
    reader.add_arg(1, name='SPLIT_K')
    reader.add_arg(True, name='EVEN_K')
    reader.add_arg(None, name='W_CACHE_MODIFIER')
    reader.add_arg(None, name='TOKENS_PER_EXPT_FOR_ANNOTATION')
    reader.add_arg(8, name='num_warps')
    reader.add_arg(4, name='num_stages')
    reader.add_arg(None, name='arch')
    reader.add_arg(False, name='UPCAST_INDICES')
    reader.add_arg('dense', name='X_TMA_MODE')
    reader.add_arg('ragged', name='Y_TMA_MODE')
    reader.add_arg(False, name='SWAP_XW')
    reader.add_arg(False, name='IS_EPILOGUE_QUANT_MXFP8')
    reader.add_arg(152, name='NUM_SMS')
load_args._version = 0


def torch_dtype_to_triton(dtype):
    if dtype == torch.float8_e5m2:
        return gl.float8e5
    if dtype == torch.float8_e4m3fn:
        return gl.float8e4nv
    return getattr(gl, str(dtype).split('.')[1])


if __name__ == '__main__':
    input_reader = InputReader(save_dir='/root/code/openai/repro')
    load_args(input_reader)

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    evo_kwargs = {}
    if "EVO" in os.environ:
        assert os.path.exists("ptxas.bin"), "ptxas.bin not found"
        evo_kwargs["ptx_options"] = "--apply-controls=ptxas.bin"

    parser = argparse.ArgumentParser(description="Launch script for _p_matmul_ogs_default")
    parser.add_argument("--bench", action="store_true", help="Run in benchmark mode")
    parser.add_argument("--dry", action="store_true", help="Run in dry mode")
    args = parser.parse_args()

    def run_fn():
        import sys
        import importlib.util
        from sandbox import convert_triton_to_gluon
        txt = convert_triton_to_gluon(_p_matmul_ogs_default)
        mod_path = "/tmp/converted_kernel.py"
        with open(mod_path, "w", encoding="utf-8") as f:
            f.write(txt)
        print(txt)

        spec = importlib.util.spec_from_file_location("converted_kernel", mod_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["converted_kernel"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        kernel = getattr(module, "_p_matmul_ogs")
        new_args = []
        for arg in input_reader.args:
            if isinstance(arg, triton.tools.tensor_descriptor.TensorDescriptor):
                block_shape = arg.block_shape
                tensor = arg.base
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.clone()
                dtype = arg.base.dtype
                layout = gl.NVMMASharedLayout.get_default_for(block_shape, torch_dtype_to_triton(dtype))
                new_desc = gluon.nvidia.hopper.TensorDescriptor(tensor, arg.shape, arg.strides, block_shape, layout)
                new_args.append(new_desc)
            else:
                if isinstance(arg, torch.Tensor):
                    arg = arg.clone()
                new_args.append(arg)
        new_kwargs = {}
        for key, value in input_reader.kwargs.items():
            if isinstance(value, triton.tools.tensor_descriptor.TensorDescriptor):
                block_shape = value.block_shape
                dtype = value.base.dtype
                tensor = value.base
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.clone()
                layout = gl.NVMMASharedLayout.get_default_for(block_shape, torch_dtype_to_triton(dtype))
                new_desc = gluon.nvidia.hopper.TensorDescriptor(tensor, value.shape, value.strides, block_shape, layout)
                new_kwargs[key] = new_desc
            else:
                if isinstance(value, torch.Tensor):
                    value = value.clone()
                new_kwargs[key] = value

        _p_matmul_ogs_default.run(*input_reader.args, **input_reader.kwargs, **evo_kwargs, grid=(1,), warmup=args.dry)
        kernel.run(*new_args, **new_kwargs, grid=(1,), warmup=False)
        print(new_args[0])
        print(input_reader.args[0])
       # for i in range(len(new_args)):
       #     if isinstance(new_args[i], torch.Tensor):
       #         print(new_args[i])
       #         print(input_reader.args[i])
       # for key, value in new_kwargs.items():
       #     if isinstance(value, torch.Tensor):
       #         print(value)
       #         print(input_reader.kwargs[key])

    run_fn()


    if args.bench:
        time = do_bench_cudagraph(run_fn, quantiles=[0.5])
        print(f"Executed in {time} ms")