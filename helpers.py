from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


@gluon.jit
def dot_accumulate(a, b, acc):
    ttgl.set_auto_layout(a, ttgl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0]))
    ttgl.set_auto_layout(b, ttgl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0]))
    acc_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
	# Prepare dot operand layouts as constexpr so the compiler can reason about them
    dot_a_layout: ttgl.constexpr = ttgl.DotOperandLayout(0, acc_layout, 0)
    dot_b_layout: ttgl.constexpr = ttgl.DotOperandLayout(1, acc_layout, 0)
    # Convert operands to required layouts
    a_conv = ttgl.convert_layout(a, dot_a_layout)
    b_conv = ttgl.convert_layout(b, dot_b_layout)
    acc_t = ttgl.convert_layout(acc, acc_layout)
    # Fused multiply-add into the accumulator
    acc_t = ttgl.dot_fma(a_conv, b_conv, acc_t)
    acc2 = ttgl.convert_layout(acc_t, ttgl.AutoLayout())
    return acc2


