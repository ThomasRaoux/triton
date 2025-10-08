from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from helpers import *


@gluon.jit
def _p_matmul_ogs(Y, YPtr, stride_y_k, stride_y_z, stride_y_m, stride_y_n, YExpectedScale, YActualScale, YChecksumScale, stride_y_mx_k, stride_y_mx_z, stride_y_mx_m, stride_y_mx_n, X, XPtr, stride_x_z, stride_x_m, stride_x_k, X_TRANSPOSE: ttgl.constexpr, XScale, XMxScale, stride_x_mx_z, stride_x_mx_m, stride_x_mx_k, W, WPtr, stride_w_e, stride_w_k, stride_w_n, W_TRANSPOSE: ttgl.constexpr, WScale, WMxScale, stride_w_mx_e, stride_w_mx_k, stride_w_mx_n, OutAcc, stride_acc_z, stride_acc_m, stride_acc_n, OutAccScale, Y_ACC_IS_Y: ttgl.constexpr, B, stride_b_e, M, N, K, K_W, Betas, Gammas, GatherIndx, ScatterSrcIndx, num_idxs, WriteBackIndx, writeback_size, ExptHist, ExptOffs, ExptTileOffs, ExptData, EXPT_IS_INNER: ttgl.constexpr, X_IS_PADDED: ttgl.constexpr, W_IS_PADDED: ttgl.constexpr, ExptHistMax, batch_size, grid_m, grid_n, out_alpha, ACTIVATION_REDUCTION_N: ttgl.constexpr, N_EXPTS_TOT: ttgl.constexpr, N_EXPTS_ACT: ttgl.constexpr, MAX_NUM_IMPRECISE_ACC: ttgl.constexpr, ALLOW_TF32: ttgl.constexpr, FLEXPOINT_SATURATE_INF: ttgl.constexpr, PER_BATCH_W_SCALE: ttgl.constexpr, PER_BATCH_OUT_SCALE: ttgl.constexpr, PER_BATCH_ACC_SCALE: ttgl.constexpr, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr, GROUP_M: ttgl.constexpr, XCD_SWIZZLE: ttgl.constexpr, SWIZZLE_MX_VALUE: ttgl.constexpr, SWIZZLE_MX_SCALE: ttgl.constexpr, EPILOGUE_SUBTILE: ttgl.constexpr, EVEN_K: ttgl.constexpr, SPLIT_K: ttgl.constexpr, W_CACHE_MODIFIER: ttgl.constexpr, NUM_SMS: ttgl.constexpr, X_TMA_MODE: ttgl.constexpr, Y_TMA_MODE: ttgl.constexpr, TOKENS_PER_EXPT_FOR_ANNOTATION=None, UPCAST_INDICES: ttgl.constexpr=False, SWAP_XW: ttgl.constexpr=False, IS_EPILOGUE_QUANT_MXFP8: ttgl.constexpr=False):
    ACTIVATION_FN: ttgl.constexpr = None
    EPILOGUE_FN: ttgl.constexpr = None
    activation_fn_args = ()
    epilogue_fn_args = ()
    #if Y_TMA_MODE is not None:
    #    Y = tl_make_tensor_descriptor(YPtr, Y.shape, Y.strides[:-1] + (1,), Y.block_shape)
    is_w_microscaled: ttgl.constexpr = WMxScale is not None
    ttgl.static_assert(not is_w_microscaled or W_TRANSPOSE, 'NYI. Non-transposed mxfp4 weights')
    MX_PACK_DIVISOR: ttgl.constexpr = 32
    if is_w_microscaled:
        w_type: ttgl.constexpr = get_dtype(W)
        ttgl.static_assert(w_type == ttgl.uint8 or (w_type == ttgl.float8e4nv or w_type == ttgl.float8e5), 'mx_weight_ptr must be uint8 or fp8')
        ttgl.static_assert(get_dtype(WMxScale) == ttgl.uint8, 'mx_scale_ptr must be uint8')
        ttgl.static_assert(BLOCK_K % MX_PACK_DIVISOR == 0, 'BLOCK_K must be a multiple of MX_PACK_DIVISOR')
        ttgl.static_assert(SWIZZLE_MX_SCALE == 'BLACKWELL_SCALE' or SWIZZLE_MX_SCALE is None, 'Only Blackwell swizzling is supported for scales')
        ttgl.static_assert(not EXPT_IS_INNER, 'Not supported yet')
        W_PACK_DIVISOR: ttgl.constexpr = 2 if w_type == ttgl.uint8 else 1
        PACKED_BLOCK_K_W: ttgl.constexpr = BLOCK_K // W_PACK_DIVISOR
        MX_SCALE_BLOCK_K: ttgl.constexpr = BLOCK_K // MX_PACK_DIVISOR
    else:
        PACKED_BLOCK_K_W: ttgl.constexpr = BLOCK_K
        ttgl.static_assert(SWIZZLE_MX_SCALE is None)
    is_x_microscaled: ttgl.constexpr = XMxScale is not None
    if is_x_microscaled:
        x_type: ttgl.constexpr = get_dtype(X)
        ttgl.static_assert(x_type == ttgl.float8e4nv, 'mx_act_ptr must be float8e4nv')
        ttgl.static_assert(XMxScale.dtype.element_ty == ttgl.uint8, 'mx_scale_ptr must be uint8')
        ttgl.static_assert(BLOCK_K % MX_PACK_DIVISOR == 0, 'BLOCK_K must be a multiple of MX_PACK_DIVISOR')
    is_out_microscaled: ttgl.constexpr = stride_y_mx_z is not None
    if ExptTileOffs is not None and (not EXPT_IS_INNER):
        padding_m = grid_m - ttgl.load(ExptTileOffs + N_EXPTS_TOT)
    else:
        padding_m: ttgl.constexpr = 0
    index_type: ttgl.constexpr = ttgl.int64
    USE_FLEXPOINT_SCALE: ttgl.constexpr = YActualScale is not None or YChecksumScale is not None
    HAS_SCATTER: ttgl.constexpr = WriteBackIndx is not None
    HAS_GATHER: ttgl.constexpr = GatherIndx is not None
    USE_GATHER_TMA: ttgl.constexpr = HAS_GATHER and X_TMA_MODE == 'dense'
    USE_SCATTER_TMA: ttgl.constexpr = HAS_SCATTER and Y_TMA_MODE == 'dense'
    if EXPT_IS_INNER:
        ttgl.static_assert(OutAcc is None or Y_ACC_IS_Y, 'Using differernt y_acc is not supported with TMA kernel.')
        ttgl.static_assert(not (HAS_SCATTER or USE_GATHER_TMA or USE_SCATTER_TMA), 'Cannot be used with EXPT_IS_INNER')
    if EPILOGUE_SUBTILE is None:
        SUBTILE_FACTOR: ttgl.constexpr = 1
    else:
        SUBTILE_FACTOR: ttgl.constexpr = EPILOGUE_SUBTILE
    EPILOGUE_BLOCK_N: ttgl.constexpr = BLOCK_N // SUBTILE_FACTOR
    OUT_BLOCK_N: ttgl.constexpr = EPILOGUE_BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N
    if HAS_SCATTER and N_EXPTS_ACT == 1:
        for pid_mnk in range(NUM_SMS - ttgl.program_id(0) - 1, batch_size * grid_m * grid_n * SPLIT_K, NUM_SMS):
            pid_k = pid_mnk % SPLIT_K
            pid_mn = pid_mnk // SPLIT_K
            pid_m, pid_n = swizzle2d(pid_mn, grid_m, grid_n, GROUP_M)
            z = zeros([BLOCK_M, BLOCK_N // ACTIVATION_REDUCTION_N], dtype=ttgl.float32)
            offs_m = z.shape[0] * pid_m + tl_arange(0, z.shape[0])
            offs_n = z.shape[1] * pid_n + tl_arange(0, z.shape[1])
            src_idx = ttgl.load(ScatterSrcIndx + offs_m, mask=offs_m < num_idxs, other=0)
            YPtrs = YPtr + ttgl.convert_layout(offs_m.to(index_type), ttgl.SliceLayout(1, default_blocked_layout([1, offs_m.to(index_type).type.shape[0]], ttgl.num_warps())))[:, None] * stride_y_m + ttgl.convert_layout(offs_n, ttgl.SliceLayout(0, default_blocked_layout([offs_n.type.shape[0], 1], ttgl.num_warps())))[None, :] * stride_y_n
            mask_n = offs_n < yN
            mask = ttgl.convert_layout(src_idx == -1, ttgl.SliceLayout(1, default_blocked_layout([1, (src_idx == -1).type.shape[0]], ttgl.num_warps())))[:, None] & ttgl.convert_layout(mask_n, ttgl.SliceLayout(0, default_blocked_layout([mask_n.type.shape[0], 1], ttgl.num_warps())))[None, :]
            ttgl.store(YPtrs + pid_k * stride_y_k, z, mask=mask)
    num_tiles = batch_size * (grid_m - padding_m) * grid_n * SPLIT_K
    INDEPENDENT_EPILOGUE: ttgl.constexpr = cuda_capability_geq(10, 0)
    if INDEPENDENT_EPILOGUE:
        tile_id1 = ttgl.program_id(0) - NUM_SMS
    USE_LOCAL_ABSMAX: ttgl.constexpr = YActualScale is not None and (not PER_BATCH_OUT_SCALE) and (not is_out_microscaled)
    if USE_LOCAL_ABSMAX:
        THREADS_PER_BLOCK: ttgl.constexpr = get_num_threads_per_warp()
        local_absmax = tl_full([THREADS_PER_BLOCK], 0.0, ttgl.uint32)
    DISALLOW_ACC_MULTI_BUFFER: ttgl.constexpr = is_w_microscaled and BLOCK_M * BLOCK_N >= 128 * 256
    for tile_id in range(ttgl.program_id(0), num_tiles, NUM_SMS):
        expt_id, start_z, start_z_out, start_m, eM, off_m, pid_n, k_tiles, pid_k, off_k_x0, off_k_w0, _ = _load_tile_attrs(tile_id, num_tiles, grid_m - padding_m, grid_n, M, K, ExptData, ExptHist, ExptOffs, ExptTileOffs, EXPT_IS_INNER, X_IS_PADDED, W_IS_PADDED, BLOCK_M, BLOCK_K, PACKED_BLOCK_K_W, SPLIT_K, GROUP_M, XCD_SWIZZLE)
        off_n = BLOCK_N * pid_n
        if X_TMA_MODE is None:
            XBase = X + start_z.to(index_type) * stride_x_z
            offs_x_k = ttgl.convert_layout(off_k_x0.to(index_type) + tl_arange(0, BLOCK_K), ttgl.SliceLayout(0, default_blocked_layout([(off_k_x0.to(index_type) + tl_arange(0, BLOCK_K)).type.shape[0], 1], ttgl.num_warps())))[None, :] * stride_x_k
        if USE_GATHER_TMA:
            offs_m = off_m + tl_arange(0, BLOCK_M)
            mask_m = offs_m < eM
            if ExptData is None:
                offs_x_m = ttgl.load(GatherIndx + start_m.to(index_type) + offs_m, mask=mask_m)
                offs_x_m += start_z * (stride_x_z // stride_x_m)
                offs_x_m = ttgl.where(mask_m, offs_x_m, -1)
            else:
                offs_x_m = ttgl.load(GatherIndx + start_m.to(index_type) + offs_m, mask=mask_m, other=-N_EXPTS_ACT) // N_EXPTS_ACT
        elif X_TMA_MODE is None or is_x_microscaled:
            offs_m = off_m + tl_arange(0, BLOCK_M)
            offs_m = ttgl.max_contiguous(ttgl.multiple_of(offs_m % eM, BLOCK_M), BLOCK_M)
            if GatherIndx is not None:
                ttgl.static_assert(HAS_GATHER)
                offs_m = ttgl.load(GatherIndx + start_m.to(index_type) + offs_m) // N_EXPTS_ACT
            offs_x_m = ttgl.convert_layout(offs_m.to(index_type), ttgl.SliceLayout(1, default_blocked_layout([1, offs_m.to(index_type).type.shape[0]], ttgl.num_warps())))[:, None] * stride_x_m
        if is_x_microscaled:
            XMxScalePtrs = XMxScale + start_z.to(index_type) * stride_x_mx_z
            if GatherIndx is None:
                XMxScalePtrs += start_m * stride_x_mx_m
            offs_k_scale = MX_SCALE_BLOCK_K * pid_k + tl_arange(0, MX_SCALE_BLOCK_K)
            XMxScalePtrs += ttgl.convert_layout((offs_x_m if USE_GATHER_TMA else offs_m).to(index_type), ttgl.SliceLayout(1, default_blocked_layout([1, (offs_x_m if USE_GATHER_TMA else offs_m).to(index_type).type.shape[0]], ttgl.num_warps())))[:, None] * stride_x_mx_m
            XMxScalePtrs += ttgl.convert_layout(offs_k_scale.to(index_type), ttgl.SliceLayout(0, default_blocked_layout([offs_k_scale.to(index_type).type.shape[0], 1], ttgl.num_warps())))[None, :] * stride_x_mx_k
        else:
            XMxScalePtrs = None
        acc = zeros((BLOCK_N, BLOCK_M) if SWAP_XW else (BLOCK_M, BLOCK_N), dtype=ttgl.float32)
        loop_bound = ttgl.maximum(k_tiles, 1)
        ttgl.assume(loop_bound > 0)
        for ki in range(loop_bound):
            if EXPT_IS_INNER and ki >= k_tiles:
                off_k_x = K
                off_k_w = K_W
            else:
                off_k_x = off_k_x0 + ki * BLOCK_K * SPLIT_K
                off_k_w = off_k_w0 + ki * PACKED_BLOCK_K_W * SPLIT_K
            if USE_GATHER_TMA:
                x = X.gather(offs_x_m, off_k_x)
            elif X_TMA_MODE == 'dense':
                if X_TRANSPOSE:
                    x = X.load([start_z, off_k_x, start_m + off_m])
                    x = x.reshape(BLOCK_K, BLOCK_M).T
                else:
                    x = X.load([start_z, start_m + off_m, off_k_x])
                    x = x.reshape(BLOCK_M, BLOCK_K)
            elif X_TMA_MODE == 'ragged':
                x = load_ragged(X, start_m, eM, [start_z, off_m, off_k_x], ragged_dim=1)
                x = x.reshape(BLOCK_M, BLOCK_K)
            else:
                ttgl.static_assert(X_TMA_MODE is None)
                XPtrs = XBase + offs_x_m + offs_x_k
                XBase += BLOCK_K * SPLIT_K * stride_x_k
                mask_k = tl_arange(0, BLOCK_K) < K - off_k_x
                if EVEN_K:
                    if SPLIT_K > 1:
                        x = ttgl.load(XPtrs, mask=ttgl.convert_layout(mask_k, ttgl.SliceLayout(0, default_blocked_layout([mask_k.type.shape[0], 1], ttgl.num_warps())))[None, :], other=0.0)
                    else:
                        x = ttgl.load(XPtrs)
                else:
                    x = ttgl.load(XPtrs, mask=ttgl.convert_layout(mask_k, ttgl.SliceLayout(0, default_blocked_layout([mask_k.type.shape[0], 1], ttgl.num_warps())))[None, :], other=0.0)
            if W_TRANSPOSE:
                w = ttgl.reshape(W.load([expt_id, off_n, off_k_w]), W.block_shape[1:]).T
            else:
                w = ttgl.reshape(W.load([expt_id, off_k_w, off_n]), W.block_shape[1:])
            if is_w_microscaled:
                x_format: ttgl.constexpr = get_scaled_dot_format_string(x.dtype)
                w_format: ttgl.constexpr = get_scaled_dot_format_string(w.dtype)
                off_k_mx = off_k_w // (MX_PACK_DIVISOR // W_PACK_DIVISOR)
                if is_x_microscaled:
                    if EVEN_K:
                        mask_k_scale = tl_full([MX_SCALE_BLOCK_K], True, dtype=ttgl.int1)
                    else:
                        mask_k_scale = off_k_mx + tl_arange(0, MX_SCALE_BLOCK_K) < cdiv(K, MX_PACK_DIVISOR)
                    x_scales = ttgl.load(XMxScalePtrs, mask=ttgl.convert_layout(mask_k_scale, ttgl.SliceLayout(0, default_blocked_layout([mask_k_scale.type.shape[0], 1], ttgl.num_warps())))[None, :], other=0.0)
                elif x_format == 'fp16' or x_format == 'bf16':
                    x_scales: ttgl.constexpr = None
                else:
                    x_scales = tl_full((BLOCK_M, BLOCK_K // MX_PACK_DIVISOR), 127, dtype=ttgl.uint8)
                ttgl.static_assert(MX_PACK_DIVISOR % W_PACK_DIVISOR == 0)
                if SWIZZLE_MX_SCALE == 'BLACKWELL_SCALE':
                    flattened_expt_n_idx = expt_id * ((N + 127) // 128) + off_n // 128
                    w_scales = WMxScale.load([0, flattened_expt_n_idx, off_k_mx // 4, 0, 0])
                    w_scales = w_scales.reshape((w_scales.shape[1], w_scales.shape[2] * w_scales.shape[-2] * w_scales.shape[-1]))
                    w_scales = unswizzle_mx_scale_bw(w_scales)
                else:
                    w_scales = WMxScale.load([expt_id, off_k_mx, off_n])
                    w_scales = ttgl.reshape(w_scales, *w_scales.shape[1:]).T
            if is_w_microscaled:
                if SWAP_XW:
                    acc = tl_dot_scaled(w.T, w_scales, w_format, x.T, x_scales, x_format, acc=acc, fast_math=True)
                else:
                    acc = tl_dot_scaled(x, x_scales, x_format, w, w_scales, w_format, acc=acc, fast_math=True)
                if is_x_microscaled:
                    XMxScalePtrs += MX_SCALE_BLOCK_K * SPLIT_K * stride_x_mx_k
            elif SWAP_XW:
                acc = tl_dot(w.T, x.T, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
            else:
                acc = tl_dot(x, w, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
        if INDEPENDENT_EPILOGUE:
            tile_id1 += NUM_SMS
            expt_id1, _, start_z1, start_m1, eM1, off_m1, pid_n1, _, pid_k1, _, _, _ = _load_tile_attrs(tile_id1, num_tiles, grid_m - padding_m, grid_n, M, K, ExptData, ExptHist, ExptOffs, ExptTileOffs, EXPT_IS_INNER, X_IS_PADDED, W_IS_PADDED, BLOCK_M, BLOCK_K, PACKED_BLOCK_K_W, SPLIT_K, GROUP_M, XCD_SWIZZLE)
            off_n1 = pid_n1 * BLOCK_N
        else:
            tile_id1, expt_id1, start_z1, start_m1, eM1 = (tile_id, expt_id, start_z_out, start_m, eM)
            off_m1, off_n1, pid_k1 = (off_m, off_n, pid_k)
        offs_m = off_m1 + tl_arange(0, BLOCK_M)
        mask_m = offs_m < eM1
        if USE_SCATTER_TMA:
            offs_y_m, mask_m = _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, start_m1 + offs_m, mask_m)
            MASK_ACC: ttgl.constexpr = USE_FLEXPOINT_SCALE
            if SPLIT_K > 1:
                ttgl.device_assert(stride_y_k // stride_y_m == cdiv(stride_y_k, stride_y_m))
                split_k_row_offs = pid_k1 * (stride_y_k // stride_y_m)
                offs_y_m = ttgl.where(mask_m, offs_y_m + split_k_row_offs, offs_y_m)
        elif Y_TMA_MODE is None:
            ttgl.static_assert(HAS_SCATTER)
            offs_y_m, mask_m = _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, start_m1 + offs_m, mask_m)
            MASK_ACC: ttgl.constexpr = USE_FLEXPOINT_SCALE
        else:
            offs_y_m = start_m1 + offs_m
            MASK_ACC = False if USE_GATHER_TMA else USE_FLEXPOINT_SCALE
        offs_y_n = off_n1 + tl_arange(0, BLOCK_N)
        mask_n = offs_y_n < N
        if B is not None:
            BPtrs = B + expt_id1 * stride_b_e + offs_y_n
            if pid_k1 == 0:
                bias = ttgl.load(BPtrs, mask=mask_n, other=0)
            else:
                bias = tl_full([BLOCK_N], 0, dtype=ttgl.float32)
        else:
            bias = tl_full([BLOCK_N], 0, dtype=ttgl.float32)
        if Betas is not None:
            betas = ttgl.load(Betas + start_m1 + offs_m, mask=mask_m, other=0.0)
        else:
            betas = tl_full([BLOCK_M], 1, dtype=ttgl.float32)
        if Gammas is not None:
            gammas = ttgl.load(Gammas + start_m1 + offs_m, mask=mask_m, other=0.0)
        else:
            gammas = tl_full([BLOCK_M], 1, dtype=ttgl.float32)
        x_scale = load_scale(XScale)
        if PER_BATCH_W_SCALE:
            w_scale = load_scale(WScale + expt_id1)
        else:
            w_scale = load_scale(WScale)
        accs = (acc,)
        biases = (bias,)
        if SUBTILE_FACTOR >= 2:
            acc0, acc1 = acc.reshape(BLOCK_M, 2, BLOCK_N // 2).permute(0, 2, 1).split()
            accs = (acc0, acc1)
            bias0, bias1 = bias.reshape(2, BLOCK_N // 2).permute(1, 0).split()
            biases = (bias0, bias1)
        if SUBTILE_FACTOR >= 4:
            acc00, acc01 = acc0.reshape(BLOCK_M, 2, BLOCK_N // 4).permute(0, 2, 1).split()
            acc10, acc11 = acc1.reshape(BLOCK_M, 2, BLOCK_N // 4).permute(0, 2, 1).split()
            accs = (acc00, acc01, acc10, acc11)
            bias00, bias01 = bias0.reshape(2, BLOCK_N // 4).permute(1, 0).split()
            bias10, bias11 = bias1.reshape(2, BLOCK_N // 4).permute(1, 0).split()
            biases = (bias00, bias01, bias10, bias11)
        ttgl.static_assert(EPILOGUE_BLOCK_N == BLOCK_N // SUBTILE_FACTOR)
        ttgl.static_assert(len(accs) == SUBTILE_FACTOR)
        if is_out_microscaled:
            MX_SCALE_BLOCK_N: ttgl.constexpr = OUT_BLOCK_N // MXFP_BLOCK_SIZE
            N_MX_BLOCK: ttgl.constexpr = cdiv(N, MXFP_BLOCK_SIZE)
        for a_i in ttgl.static_range(len(accs)):
            acc_tile = accs[a_i]
            acc_tile *= x_scale * w_scale
            if SWAP_XW:
                acc_tile = acc_tile.T
            acc_tile = acc_tile + ttgl.convert_layout(biases[a_i], ttgl.SliceLayout(0, default_blocked_layout([biases[a_i].type.shape[0], 1], ttgl.num_warps())))[None, :] * ttgl.convert_layout(betas, ttgl.SliceLayout(1, default_blocked_layout([1, betas.type.shape[0]], ttgl.num_warps())))[:, None]
            if out_alpha is not None:
                acc_tile *= out_alpha
            if ACTIVATION_FN is not None:
                out = ACTIVATION_FN(acc_tile, *activation_fn_args)
                ttgl.static_assert(out.shape[1] == OUT_BLOCK_N, f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})")
            else:
                ttgl.static_assert(ACTIVATION_REDUCTION_N == 1, 'Activation reduction must be 1 if no activation fn is provided')
                out = acc_tile
            out *= ttgl.convert_layout(gammas, ttgl.SliceLayout(1, default_blocked_layout([1, gammas.type.shape[0]], ttgl.num_warps())))[:, None]
            if OutAcc is not None:
                ttgl.static_assert(not USE_SCATTER_TMA)
                out_off_n = off_n1 // ACTIVATION_REDUCTION_N + a_i * OUT_BLOCK_N
                if PER_BATCH_ACC_SCALE:
                    ScalePtr = OutAccScale + start_z1
                else:
                    ScalePtr = OutAccScale
                ttgl.static_assert(Y_TMA_MODE == 'dense' or Y_TMA_MODE is None)
                if Y_TMA_MODE == 'dense':
                    off_kz = pid_k * batch_size + start_z1
                    acc = Y.load([off_kz, off_m1, out_off_n])
                    acc = acc.reshape(out.shape)
                    out += acc * load_scale(ScalePtr)
                else:
                    offs_y_n = out_off_n + tl_arange(0, OUT_BLOCK_N)
                    mask_n = offs_y_n < yN
                    AccPtrs = YPtr + pid_k1.to(index_type) * stride_y_k + start_z1.to(index_type) * stride_y_z + ttgl.convert_layout(offs_y_m.to(index_type), ttgl.SliceLayout(1, default_blocked_layout([1, offs_y_m.to(index_type).type.shape[0]], ttgl.num_warps())))[:, None] * stride_y_m + ttgl.convert_layout(offs_y_n, ttgl.SliceLayout(0, default_blocked_layout([offs_y_n.type.shape[0], 1], ttgl.num_warps())))[None, :] * stride_y_n
                    mask = ttgl.convert_layout(mask_m, ttgl.SliceLayout(1, default_blocked_layout([1, mask_m.type.shape[0]], ttgl.num_warps())))[:, None] & ttgl.convert_layout(mask_n, ttgl.SliceLayout(0, default_blocked_layout([mask_n.type.shape[0], 1], ttgl.num_warps())))[None, :]
                    acc = ttgl.load(AccPtrs, mask=mask, other=0.0)
                    out += acc * load_scale(ScalePtr)
            if MASK_ACC:
                out = ttgl.where(ttgl.convert_layout(mask_m, ttgl.SliceLayout(1, default_blocked_layout([1, mask_m.type.shape[0]], ttgl.num_warps())))[:, None], out, 0.0)
            out_off_n = off_n1 // ACTIVATION_REDUCTION_N + a_i * OUT_BLOCK_N
            if is_out_microscaled:
                ttgl.static_assert(EPILOGUE_FN is not None)
                offs_y_n = out_off_n + tl_arange(0, OUT_BLOCK_N)
                mask_n = offs_y_n < yN
                out, out_scale = EPILOGUE_FN(out, ttgl.convert_layout(mask_m, ttgl.SliceLayout(1, default_blocked_layout([1, mask_m.type.shape[0]], ttgl.num_warps())))[:, None] & ttgl.convert_layout(mask_n, ttgl.SliceLayout(0, default_blocked_layout([mask_n.type.shape[0], 1], ttgl.num_warps())))[None, :], *epilogue_fn_args)
                ttgl.static_assert(BLOCK_N % MX_SCALE_BLOCK_N == 0, '')
                offs_y_n_scale = off_n1 // ACTIVATION_REDUCTION_N // MXFP_BLOCK_SIZE + a_i * MX_SCALE_BLOCK_N + tl_arange(0, MX_SCALE_BLOCK_N)
                mask_n_scale = offs_y_n_scale < N_MX_BLOCK
                offs_y_mx_k = 0
                if USE_SCATTER_TMA:
                    offs_y_mx_z = 0
                    offs_y_mx_m = (offs_y_m.to(ttgl.uint32, bitcast=True) & 2147483647).to(ttgl.int32, bitcast=True)
                elif Y_TMA_MODE == 'dense':
                    offs_y_mx_z = pid_k * batch_size + start_z1
                    offs_y_mx_m = off_m1 + tl_arange(0, BLOCK_M)
                elif Y_TMA_MODE == 'ragged':
                    offs_y_mx_z = pid_k
                    offs_y_mx_m = start_m1 + off_m1 + tl_arange(0, BLOCK_M)
                else:
                    ttgl.static_assert(Y_TMA_MODE is None)
                    offs_y_mx_k = pid_k1
                    offs_y_mx_z = start_z1
                YActualScalePtrs = YActualScale + offs_y_mx_k.to(index_type) * stride_y_mx_k + offs_y_mx_z.to(index_type) * stride_y_mx_z + ttgl.convert_layout(offs_y_mx_m.to(index_type), ttgl.SliceLayout(1, default_blocked_layout([1, offs_y_mx_m.to(index_type).type.shape[0]], ttgl.num_warps())))[:, None] * stride_y_mx_m + ttgl.convert_layout(offs_y_n_scale.to(index_type), ttgl.SliceLayout(0, default_blocked_layout([offs_y_n_scale.to(index_type).type.shape[0], 1], ttgl.num_warps())))[None, :] * stride_y_mx_n
                ttgl.store(YActualScalePtrs, out_scale, mask=ttgl.convert_layout(mask_m, ttgl.SliceLayout(1, default_blocked_layout([1, mask_m.type.shape[0]], ttgl.num_warps())))[:, None] & ttgl.convert_layout(mask_n_scale, ttgl.SliceLayout(0, default_blocked_layout([mask_n_scale.type.shape[0], 1], ttgl.num_warps())))[None, :])
            else:
                if USE_LOCAL_ABSMAX:
                    out_view = ttgl.reshape(out, [out.numel // THREADS_PER_BLOCK, THREADS_PER_BLOCK], can_reorder=True)
                    local_absmax = ttgl.maximum(local_absmax, nan_propagating_absmax_reduce(out_view, axis=0))
                if PER_BATCH_OUT_SCALE:
                    ExpectedScale = YExpectedScale + start_z1
                    ActualScale = YActualScale + start_z1
                else:
                    ExpectedScale = YExpectedScale
                    ActualScale = None
                out = float_to_flex(out, ExpectedScale, ActualScale, YChecksumScale, None, YPtr, FLEXPOINT_SATURATE_INF)
                if EPILOGUE_FN is not None and (not IS_EPILOGUE_QUANT_MXFP8):
                    out = EPILOGUE_FN(out, *epilogue_fn_args, target_dtype=YPtr.dtype.element_ty, pid=len(accs) * tile_id1 + a_i)
            out = out.to(YPtr.dtype.element_ty)
            if USE_SCATTER_TMA:
                offs_y_m = (offs_y_m.to(ttgl.uint32, bitcast=True) & 2147483647).to(ttgl.int32, bitcast=True)
                Y.scatter(out, offs_y_m, out_off_n)
            elif Y_TMA_MODE == 'dense':
                out = ttgl.reshape(out, [1] + out.shape)
                off_kz = pid_k * batch_size + start_z1
                Y.store([off_kz, off_m1, out_off_n], out)
            elif Y_TMA_MODE == 'ragged':
                out = ttgl.reshape(out, [1] + out.shape)
                store_ragged(Y, start_m1, eM1, [pid_k, off_m1, out_off_n], out, ragged_dim=1)
            else:
                ttgl.static_assert(Y_TMA_MODE is None)
                offs_y_n = out_off_n + tl_arange(0, OUT_BLOCK_N)
                mask_n = offs_y_n < yN
                YPtrs = YPtr + pid_k1.to(index_type) * stride_y_k + start_z1.to(index_type) * stride_y_z + ttgl.convert_layout(offs_y_m.to(index_type), ttgl.SliceLayout(1, default_blocked_layout([1, offs_y_m.to(index_type).type.shape[0]], ttgl.num_warps())))[:, None] * stride_y_m + ttgl.convert_layout(offs_y_n, ttgl.SliceLayout(0, default_blocked_layout([offs_y_n.type.shape[0], 1], ttgl.num_warps())))[None, :] * stride_y_n
                mask = ttgl.convert_layout(mask_m, ttgl.SliceLayout(1, default_blocked_layout([1, mask_m.type.shape[0]], ttgl.num_warps())))[:, None] & ttgl.convert_layout(mask_n, ttgl.SliceLayout(0, default_blocked_layout([mask_n.type.shape[0], 1], ttgl.num_warps())))[None, :]
                ttgl.store(YPtrs, out, mask=mask)
    if USE_LOCAL_ABSMAX:
        ttgl.atomic_max(YActualScale, compute_scale(local_absmax.to(ttgl.float32, bitcast=True), YPtr), sem='relaxed')

@gluon.jit
def swizzle2d(pid, grid_m, grid_n, GROUP_M: ttgl.constexpr):
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    ttgl.assume(group_size >= 0)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    return (pid_m, pid_n)

@gluon.jit
def zeros(shape, dtype):
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`
    :type dtype: DType
    """
    return tl_full(shape, 0, dtype)

@gluon.jit
def _load_tile_attrs(tile_id, num_tiles, unpadded_m, grid_n, M, K, ExptData, ExptHist, ExptOffs, ExptTileOffs, EXPT_IS_INNER: ttgl.constexpr, X_IS_PADDED: ttgl.constexpr, W_IS_PADDED: ttgl.constexpr, BLOCK_M: ttgl.constexpr, BLOCK_K: ttgl.constexpr, PACKED_BLOCK_K_W: ttgl.constexpr, SPLIT_K: ttgl.constexpr, GROUP_M: ttgl.constexpr, XCD_SWIZZLE: ttgl.constexpr):
    pid_emnk = tile_id
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, num_tiles, XCD_SWIZZLE)
    pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    if SPLIT_K > 1:
        pid_k = pid_mnk % SPLIT_K
        pid_mn = pid_mnk // SPLIT_K
    else:
        pid_k: ttgl.constexpr = 0
        pid_mn = pid_mnk
    pid_m, pid_n = swizzle2d(pid_mn, unpadded_m, grid_n, GROUP_M)
    if EXPT_IS_INNER:
        ttgl.static_assert(X_IS_PADDED or W_IS_PADDED, 'At least one input must be padded!')
        ttgl.static_assert(SPLIT_K == 1, 'Not supported yet')
        ttgl.static_assert(M is not None)
        expt_id, pid_z, pid_z_out, start_m, block_id, eM = (0, 0, pid_e, 0, pid_m, M)
        k_tiles = cdiv(ttgl.load(ExptHist + pid_e), BLOCK_K)
        padded_start_off = ttgl.load(ExptTileOffs + pid_e) * BLOCK_K
        unpadded_start_off = ttgl.load(ExptOffs + pid_e)
        off_k_x = padded_start_off if X_IS_PADDED else unpadded_start_off
        if W_IS_PADDED:
            off_k_w = padded_start_off
            K_W = ttgl.load(ExptTileOffs + pid_e + 1) * BLOCK_K
        else:
            off_k_w = unpadded_start_off
            K_W = ttgl.load(ExptOffs + pid_e + 1)
    else:
        off_k_x = pid_k * BLOCK_K
        off_k_w = pid_k * PACKED_BLOCK_K_W
        if PACKED_BLOCK_K_W >= BLOCK_K:
            K_W = K * (PACKED_BLOCK_K_W // BLOCK_K)
        else:
            K_W = K // (BLOCK_K // PACKED_BLOCK_K_W)
        k_tiles = cdiv(K - off_k_x, BLOCK_K * SPLIT_K)
        if ExptData is None:
            ttgl.static_assert(M is not None)
            expt_id, pid_z, pid_z_out, start_m, block_id, eM = (pid_e, pid_e, pid_e, 0, pid_m, M)
        else:
            ttgl.static_assert(M is None)
            expt_data = ttgl.load(ExptData + pid_m)
            expt_id = expt_data & 65535
            block_id = expt_data >> 16
            eM = ttgl.load(ExptHist + expt_id)
            start_m = ttgl.load(ExptOffs + expt_id)
            pid_z, pid_z_out = (0, 0)
    off_m = BLOCK_M * block_id
    return (expt_id, pid_z, pid_z_out, start_m, eM, off_m, pid_n, k_tiles, pid_k, off_k_x, off_k_w, K_W)

@gluon.jit
def load_ragged(TMA, batch_offset, batch_size, coords, ragged_dim: ttgl.constexpr=0):
    """
    Read from a subarray T[batch_offset : batch_offset + batch_size] with
    hardware bounds-checking, where reading outside the subarray gives zeros.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.load().
    """
    ttgl.static_assert(len(TMA.shape) == len(coords) + 2, 'TMA must be a read-write ragged descriptor')
    c0, c1, c2 = to_ragged_indices(batch_offset, batch_size, coords[ragged_dim])
    data = TMA.load([c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:])
    data = ttgl.reshape(data, data.shape[2:])
    return data

@gluon.jit
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type x: Block
    :param div: the divisor
    :type div: Block
    """
    return (x + div - 1) // div

@gluon.jit
def unswizzle_mx_scale_bw(x, SIZE_OUTER: ttgl.constexpr=128, SIZE_INNER: ttgl.constexpr=4, ALIGN_INNER: ttgl.constexpr=8):
    shape_0: ttgl.constexpr = x.shape[0]
    shape_1: ttgl.constexpr = x.shape[1]
    ttgl.static_assert(shape_1 % SIZE_OUTER == 0)
    ttgl.static_assert(shape_1 // SIZE_OUTER <= ALIGN_INNER)
    x = x.reshape(shape_0, shape_1 // SIZE_OUTER // SIZE_INNER, 32, SIZE_OUTER // 32, SIZE_INNER)
    x = x.trans(0, 3, 2, 1, 4).reshape(shape_0 * SIZE_OUTER, shape_1 // SIZE_OUTER)
    return x

@gluon.jit
def _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, offs, mask):
    mask = mask & (offs < writeback_size)
    offs = ttgl.load(WriteBackIndx + offs, mask=mask, other=-1)
    mask = offs != -1
    return (offs, mask)

@gluon.jit
def load_scale(scale_ptr):
    return 1.0 if scale_ptr is None else ttgl.load(scale_ptr)

@gluon.jit
def nan_propagating_absmax_reduce(x, axis=None):
    if cuda_capability_geq(8, 6):
        x_absmax = ttgl.reduce(x, axis, sm86_max_nan_xorsign_abs_f32)
        x_absmax = x_absmax.to(ttgl.uint32, bitcast=True) & 2147483647
    else:
        masked_abs_x = x.to(ttgl.uint32, bitcast=True) & 2147483647
        x_absmax = max(masked_abs_x, axis)
    return x_absmax

@gluon.jit
def float_to_flex(x, expected_scale_ptr_or_val, actual_scale_ptr, checksum_scale_ptr, mask, Out, saturate_infs: ttgl.constexpr):
    if expected_scale_ptr_or_val is not None:
        if expected_scale_ptr_or_val.dtype.is_ptr():
            invscale = 1.0 / ttgl.load(expected_scale_ptr_or_val)
        else:
            invscale = 1.0 / expected_scale_ptr_or_val
    else:
        invscale = 1.0
    if checksum_scale_ptr is not None:
        x_int32 = x.to(ttgl.int32, bitcast=True)
        zero = ttgl.cast(0.0, ttgl.int32)
        if mask is not None:
            x_int32 = ttgl.where(mask, x_int32, zero)
        checksum_local = xor_sum(ravel(x_int32, can_reorder=True), 0)
        ttgl.atomic_add(checksum_scale_ptr, checksum_local)
    if mask is not None:
        if actual_scale_ptr is not None:
            x = ttgl.where(mask, x, 0.0)
    update_scale(x, actual_scale_ptr, Out)
    x = x * invscale
    if expected_scale_ptr_or_val is not None:
        if saturate_infs:
            CLIP_VALUE = max_finite(Out.dtype.element_ty)
            x = clip(x, CLIP_VALUE)
    return x

@gluon.jit
def store_ragged(TMA, batch_offset, batch_size, coords, data, ragged_dim: ttgl.constexpr=0):
    """
    Write to a subarray T[batch_offset : batch_offset + batch_size] with
    hardware bounds-checking, where writes outside the subarray are masked
    correctly.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.store().
    """
    c0, c1, c2 = to_ragged_indices(batch_offset, batch_size, coords[ragged_dim])
    data = ttgl.reshape(data, [1, 1] + data.shape)
    TMA.store([c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:], data)

@gluon.jit
def compute_scale(x, Out):
    x_absmax = nan_propagating_absmax_reduce(ravel(x, can_reorder=True))
    x_absmax = ttgl.minimum(x_absmax, 2139095040).to(ttgl.float32, bitcast=True)
    RCP_MAX_VALUE = rcp_max_finite(Out.dtype.element_ty)
    return ttgl.fma(x_absmax, RCP_MAX_VALUE.to(ttgl.float32, bitcast=True), 1e-30)

@gluon.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: ttgl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid

@gluon.jit
def to_ragged_indices(batch_offset, batch_size, row):
    """
    Helper function for load_ragged and store_ragged.
    """
    billion = 1073741824
    x = billion - batch_size + row
    y = batch_offset + batch_size
    return (billion, y, x)

@gluon.jit
def max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if ttgl.constexpr(input.dtype.primitive_bitwidth) < ttgl.constexpr(32):
            if ttgl.constexpr(input.dtype.is_floating()):
                input = input.to(ttgl.float32)
            else:
                assert input.dtype.is_int(), 'Expecting input to be integer type'
                input = input.to(ttgl.int32)
        return ttgl.reduce(input, axis, _elementwise_max, keep_dims=keep_dims)

@gluon.jit
def ravel(x, can_reorder=False):
    """
    Returns a contiguous flattened view of :code:`x`.

    :param x: the input tensor
    :type x: Block
    """
    return ttgl.reshape(x, [x.numel], can_reorder=can_reorder)

@gluon.jit
def xor_sum(input, axis=None, keep_dims=False):
    ttgl.static_assert(input.type.scalar.is_int(), 'xor_sum only supported for integers')
    return ttgl.reduce(input, axis, _xor_combine, keep_dims=keep_dims)

@gluon.jit
def update_scale(x, scale_ptr, Out) -> None:
    if scale_ptr is not None:
        scale = compute_scale(x, Out)
        ttgl.atomic_max(scale_ptr, scale, sem='relaxed')

@gluon.jit
def max_finite(dtype):
    if dtype == ttgl.constexpr(ttgl.float8e5):
        return TL_MAX_FINITE_FLOAT8E5
    elif dtype == ttgl.constexpr(ttgl.float8e4nv):
        return TL_MAX_FINITE_FLOAT8E4NV
    elif dtype == ttgl.constexpr(ttgl.float8e4b8):
        return TL_MAX_FINITE_FLOAT8E4B8
    elif dtype == ttgl.constexpr(ttgl.float8e4b15):
        return TL_MAX_FINITE_FLOAT8E4B15
    elif dtype == ttgl.constexpr(ttgl.float16):
        return TL_MAX_FINITE_FLOAT16
    else:
        ttgl.static_assert(ttgl.constexpr(False), f'{dtype} not supported in flexpoint')

@gluon.jit
def clip(x, limit):
    res = ttgl.minimum(x, limit)
    res = ttgl.maximum(-limit, res)
    return res

@gluon.jit
def rcp_max_finite(dtype):
    if dtype == ttgl.constexpr(ttgl.float8e5):
        return TL_RCP_MAX_FINITE_FLOAT8E5
    elif dtype == ttgl.constexpr(ttgl.float8e4nv):
        return TL_RCP_MAX_FINITE_FLOAT8E4NV
    elif dtype == ttgl.constexpr(ttgl.float8e4b8):
        return TL_RCP_MAX_FINITE_FLOAT8E4B8
    elif dtype == ttgl.constexpr(ttgl.float8e4b15):
        return TL_RCP_MAX_FINITE_FLOAT8E4B15
    elif dtype == ttgl.constexpr(ttgl.float16):
        return TL_RCP_MAX_FINITE_FLOAT16
    else:
        ttgl.static_assert(ttgl.constexpr(False), f'{dtype} not supported in flexpoint')