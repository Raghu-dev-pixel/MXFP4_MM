"""
FP4 quant + FP4 GEMM: bf16 A, MXFP4 B -> MXFP4 per-1x32 quant A -> gemm_a4w4 -> bf16 C.
Fused kernel: built directly on production _mxfp4_quant_op + verified shuffle
from _fused_rms_mxfp4_quant_kernel. No custom packing or index math.
"""
from task import input_t, output_t

import triton
import triton.language as tl
import torch

import aiter
from aiter import dtypes
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0,
    }
)
@triton.jit
def _dynamic_mxfp4_quant_and_shuffle_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M, N,
    SCALE_N_PAD,   # = SN = padded scale cols, used in shuffle stride
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_ITER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_n = tl.program_id(1) * NUM_ITER

    stride_x_m     = tl.cast(stride_x_m_in,     tl.int64)
    stride_x_n     = tl.cast(stride_x_n_in,     tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m_in, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n_in, tl.int64)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs   = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n

        if EVEN_M_N:
            x = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
        else:
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(tl.float32)

        # Production quant op — handles all nibble packing correctly via tl.split
        out_tensor, bs_e8m0 = _mxfp4_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        # Store packed FP4 — identical to original kernel
        out_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_offs_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
        out_offs   = out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n

        if EVEN_M_N:
            tl.store(x_fp4_ptr + out_offs, out_tensor)
        else:
            out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        # Store scales in shuffled layout
        # Copied verbatim from _fused_rms_mxfp4_quant_kernel — verified working
        bs_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)

        bs_offs_0 = bs_offs_m[:, None] // 32
        bs_offs_1 = bs_offs_m[:, None] % 32
        bs_offs_2 = bs_offs_1 % 16
        bs_offs_1 = bs_offs_1 // 16
        bs_offs_3 = bs_offs_n[None, :] // 8
        bs_offs_4 = bs_offs_n[None, :] % 8
        bs_offs_5 = bs_offs_4 % 4
        bs_offs_4 = bs_offs_4 // 4
        bs_offs = (
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * SCALE_N_PAD
        )

        # Pad out-of-bounds scales with 127 (E8M0 neutral value)
        num_bs_cols = (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
        bs_mask_127 = (bs_offs_m < M)[:, None] & (bs_offs_n < num_bs_cols)[None, :]
        bs_e8m0 = tl.where(bs_mask_127, bs_e8m0, 127)

        if EVEN_M_N:
            tl.store(bs_ptr + bs_offs, bs_e8m0)
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < num_bs_cols)[None, :]
            tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask)


def dynamic_mxfp4_quant_and_shuffle(
    x: torch.Tensor, scaling_mode: str = "even"
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.ndim == 2
    assert x.shape[1] % 32 == 0

    M, N = x.shape
    MXFP4_QUANT_BLOCK_SIZE = 32
    N_scale = N // MXFP4_QUANT_BLOCK_SIZE

    # SCALE_N_PAD must match e8m0_shuffle padding exactly
    SCALE_N_PAD = (N_scale + 7) // 8 * 8

    x_fp4           = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    # Allocate padded scale buffer — shuffle writes may land in padding region
    SM = (M + 255) // 256 * 256
    blockscale_e8m0 = torch.full((SM, SCALE_N_PAD), 127, dtype=torch.uint8, device=x.device)

    if M <= 32:
        NUM_ITER     = 1
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_N = 32
        NUM_WARPS    = 1
        NUM_STAGES   = 1
    else:
        NUM_ITER     = 4
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        NUM_WARPS    = 4
        NUM_STAGES   = 2
        if N <= 16384:
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 128

    if N <= 1024:
        NUM_ITER     = 1
        NUM_STAGES   = 1
        NUM_WARPS    = 4
        BLOCK_SIZE_N = max(32, min(256, triton.next_power_of_2(N)))
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(M))

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N * NUM_ITER),
    )

    _dynamic_mxfp4_quant_and_shuffle_kernel[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0.stride(),
        M=M, N=N,
        SCALE_N_PAD=SCALE_N_PAD,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_ITER=NUM_ITER,
        NUM_STAGES=NUM_STAGES,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SCALING_MODE=0,
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=NUM_STAGES,
    )

    return x_fp4, blockscale_e8m0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()

    A_q, A_scale_sh = dynamic_mxfp4_quant_and_shuffle(A)
    # No trimming — gemm_a4w4 accepts padded scale shape
    A_q        = A_q.view(dtypes.fp4x2)
    A_scale_sh = A_scale_sh.view(dtypes.fp8_e8m0)

    return aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale_sh,
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )


# ---------------------------------------------------------------------------
# Warmup — force JIT compilation at import time, before benchmark timer starts
# Covers all competition benchmark shapes
# ---------------------------------------------------------------------------
"""def _warmup():
    shapes = [
        (4,   512),
        (16,  7168),
        (32,  512),
        (32,  512),
        (64,  2048),
        (256, 1536),
    ]
    for M, N in shapes:
        x = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        dynamic_mxfp4_quant_and_shuffle(x)
    torch.cuda.synchronize()

_warmup()"""