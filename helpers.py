from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


@gluon.jit
def dot_accumulate(a, b, acc):
	# Prepare dot operand layouts as constexpr so the compiler can reason about them
	dot_a_layout: ttgl.constexpr = ttgl.DotOperandLayout(0, acc.type.layout, 0)
	dot_b_layout: ttgl.constexpr = ttgl.DotOperandLayout(1, acc.type.layout, 0)
	# Convert operands to required layouts
	a_conv = ttgl.convert_layout(a, dot_a_layout)
	b_conv = ttgl.convert_layout(b, dot_b_layout)
	# Fused multiply-add into the accumulator
	return ttgl.dot_fma(a_conv, b_conv, acc)


@gluon.constexpr_function
def default_blocked_layout(shape: ttgl.constexpr, num_warps: ttgl.constexpr) -> ttgl.constexpr:
	# shape: list of positive ints (constexpr)
	rank = len(shape)
	# 1 element per thread for all dimensions
	size_per_thread = [1 for _ in range(rank)]
	# Distribute 32 threads per warp across dimensions (simple heuristic: last-fastest)
	threads_per_warp = [1 for _ in range(rank)]
	remaining_threads = 32
	for dim in range(rank - 1, -1, -1):
		threads_per_warp[dim] = remaining_threads
		remaining_threads = 1
		break
	# Use provided num_warps to distribute warps per CTA (put all on first dim)
	warps_per_cta = [1 for _ in range(rank)]
	warps_per_cta[0] = num_warps
	# Natural order [0, 1, ..., rank-1]
	order = [i for i in range(rank)]
	return ttgl.BlockedLayout(size_per_thread=size_per_thread,
							   threads_per_warp=threads_per_warp,
							   warps_per_cta=warps_per_cta,
							   order=order)


