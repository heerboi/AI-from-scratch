import torch
import math
import numpy as np

import triton
import triton.language as tl

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

@triton.jit
def fwd(Q, K, V, O, L,
        Q_stride_B, Q_stride_H,
        Q_stride_S, Q_stride_D,
        K_stride_B, K_stride_H,
        K_stride_S, K_stride_D,
        V_stride_B, V_stride_H,
        V_stride_S, V_stride_D,
        SEQ_LEN,
        SCALE,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM_Q: tl.constexpr,HEAD_DIM_KV: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr):


    block_idx = tl.program_id(0)

    head_batch_idx = tl.program_id(1)

    batch_idx = head_batch_idx // NUM_HEADS
    head_idx = head_batch_idx % NUM_HEADS

    q_offset = batch_idx.to(tl.int64) * Q_stride_B + head_idx.to(tl.int64) * Q_stride_H
    kv_offset = batch_idx.to(tl.int64) * K_stride_B + head_idx.to(tl.int64) * K_stride_H

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(SEQ_LEN, HEAD_DIM_Q),
        strides=(Q_stride_S, Q_stride_D),
        offsets=(block_idx * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM_Q),
        order=(1,0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + q_offset,
        shape=(SEQ_LEN, HEAD_DIM_Q),
        strides=(Q_stride_S, Q_stride_D),
        offsets=(block_idx * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM_Q),
        order=(1,0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(HEAD_DIM_KV, SEQ_LEN),
        strides=(K_stride_D, K_stride_S),
        offsets=(0, 0),
        block_shape=(HEAD_DIM_KV, BLOCK_SIZE_KV),
        order=(0,1),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(SEQ_LEN, HEAD_DIM_KV),
        strides=(V_stride_S, V_stride_D),
        offsets=(0,0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM_KV),
        order=(0,1),
    )

    q_offsets = block_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    kv_offsets = tl.arange(0, BLOCK_SIZE_KV)

    # adding 1 to prevent log(0) = undefined issues i think?
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype = tl.float32)

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")

    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM_Q], dtype=tl.float32)
    Q_block = tl.load(Q_block_ptr)

    for kv_idx in range(0, SEQ_LEN, BLOCK_SIZE_KV):
        kv_indices = kv_idx + kv_offsets

        mask = q_offsets[:, None] >= kv_indices[None, :]

        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block) * SCALE
        
        QK_block = tl.where(mask, QK_block, -1e9)

        m_ij = tl.maximum(m_i, tl.max(QK_block, 1))

        QK_block = QK_block - m_ij[:, None]

        P_block = tl.math.exp(QK_block)

        l_ij = tl.sum(P_block, 1)
        correction = tl.math.exp(m_i - m_ij)

        l_i = l_ij + l_i * correction

        V_block = tl.load(V_block_ptr)

        # P_block = P_block.to(tl.float16)

        O_block = O_block * correction[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    
    #logsumexp
    m_i += tl.math.log(l_i)

    O_block = O_block / l_i[:, None]
    L_ptrs = L + head_batch_idx * SEQ_LEN + q_offsets

    tl.store(O_block_ptr, O_block.to(O.type.element_ty))
    tl.store(L_ptrs, m_i)

@triton.jit
def _attn_bwd_computeD(O, dO, D, SEQ_LEN: tl.constexpr, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr):
    
    block_idx = tl.program_id(0)

    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    head_batch_idx = tl.program_id(1)


    head_idx = tl.arange(0, HEAD_DIM)

    O_block = tl.load(
        O 
        + head_batch_idx * HEAD_DIM * SEQ_LEN
        + offs[:, None] * HEAD_DIM
        + head_idx[None, :]
    ).to(tl.float32)

    dO_block = tl.load(
        dO 
        + head_batch_idx * HEAD_DIM * SEQ_LEN
        + offs[:, None] * HEAD_DIM
        + head_idx[None, :]
    ).to(tl.float32)

    # reduce cols of the pointwise multiply
    D_block = tl.sum(dO_block * O_block, axis = 1)

    D_block_ptrs = D + head_batch_idx * SEQ_LEN + offs

    tl.store(D_block_ptrs, D_block)

@triton.jit
def _attn_bwd_dk_dv(Q, K, V, softmax_scale, dO, dK, dV, D, L, stride_batch, stride_head, stride_seq, stride_dim,
                    NUM_HEADS, SEQ_LEN, BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, HEAD_DIM: tl.constexpr, STAGE: tl.constexpr):
    
    head_batch_idx = tl.program_id(2)
    batch_idx = head_batch_idx // NUM_HEADS
    head_idx = head_batch_idx % NUM_HEADS

    # to skip to the relevant (1, 1, SEQ_LEN, HEAD_DIM)
    # for tensors with head_dim
    offset_head_batch = (stride_batch * batch_idx + stride_head * head_idx).to(tl.int64)

    # this is for tensors with (B, N_H, SEQ_LEN) dim
    # since the stride is of the 4 dim tensors, we directly use the head_batch_idx and SEQ_LEN
    offset_head_batch_seq = (head_batch_idx * SEQ_LEN).to(tl.int64)


    Q += offset_head_batch
    K += offset_head_batch
    V += offset_head_batch
    dO += offset_head_batch
    dK += offset_head_batch
    dV += offset_head_batch

    L += offset_head_batch_seq
    D += offset_head_batch_seq

    head_dim_offsets = tl.arange(0, HEAD_DIM)

    # KV uses macro, and grid uses macro
    block_idx_kv = tl.program_id(0)
    start_idx_kv = block_idx_kv * BLOCK_KV

    # relevant 128 offsets
    kv_offsets = start_idx_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # (MACRO, HEAD_DIM)
    K_block = tl.load(
        K + kv_offsets[:, None] * stride_seq + head_dim_offsets[None, :] * stride_dim
    )

    V_block = tl.load(
        V + kv_offsets[:, None] * stride_seq + head_dim_offsets[None, :] * stride_dim
    )

    q_offsets = tl.arange(0, BLOCK_Q)

    # q_ptrs = Q + q_offsets[:, None] * stride_seq + head_dim_offsets[None, :] * stride_dim
    # qT_ptrs = tl.trans(q_ptrs)
    # better way of doing this is just by swapping the dimensions of the offsets

    # (HEAD_DIM, MICRO)
    qT_ptrs = Q + q_offsets[None, :] * stride_seq + head_dim_offsets[:, None] * stride_dim

    dO_ptrs = dO + q_offsets[:, None] * stride_seq + head_dim_offsets[None, :] * stride_dim

    q_ctr = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for step in range(num_steps):
        qT_block = tl.load(qT_ptrs)

        q_offsets = q_ctr + tl.arange(0, BLOCK_Q)

        # (MICRO, )
        l = tl.load(l + q_offsets)

        # (MACRO, MICRO)
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block) 

        # apply the logsumexp across the columns since QK^T is transposed
        # bit confusing fkjdsl
        P_T_block = tl.math.exp(QK_T_block - l[None, :])

class FlashAttn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal=False, softmax_scale=None):
        if softmax_scale == None:
            softmax_scale = 1.0/math.sqrt(Q.shape[-1])

        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]

        B, N_H, SEQ_LEN = Q.shape[:-1]

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.zeros_like(Q, device="cuda")
        L = torch.zeros((B, N_H, SEQ_LEN), device="cuda")
        
        kernel_warmup = fwd.warmup(Q, K, V, O, L, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                  K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                  V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                  SEQ_LEN=SEQ_LEN,
                  SCALE = softmax_scale,
                  NUM_HEADS = N_H, HEAD_DIM_Q=HEAD_DIM_Q, HEAD_DIM_KV=HEAD_DIM_K,
                  BLOCK_SIZE_Q=32,BLOCK_SIZE_KV=32, grid=(1,))

        grid = (
            SEQ_LEN // 32,
            N_H * B,
            1,
        )

        fwd[grid](Q, K, V, O, L, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                  K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                  V.stride(0), V.stride(1), V.stride(2), V.stride(3), SEQ_LEN=SEQ_LEN,
                  SCALE = softmax_scale,
                  NUM_HEADS = N_H, HEAD_DIM_Q=HEAD_DIM_Q, HEAD_DIM_KV=HEAD_DIM_K,
                  BLOCK_SIZE_Q=32,BLOCK_SIZE_KV=32)
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        B, N_H, SEQ_LEN = Q.shape[:-1]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        dQ = torch.empty_like(Q, device="cuda")
        dK = torch.empty_like(K, device="cuda")
        dV = torch.empty_like(V, device="cuda")

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO,
                N_H * B)
        D = torch.empty_like(L, device="cuda")

        _attn_bwd_computeD[preprocess_grid](
            O,
            dO,
            D,
            SEQ_LEN,
            BLOCK_SIZE_MACRO,
            ctx.HEAD_DIM,
        )

        grid = (
            SEQ_LEN // BLOCK_SIZE_MACRO,
            1,
            B * N_H
        )

        _attn_bwd_dk_dv[grid](
            Q, K, V,
            ctx.softmax_scale, dO, dK, dV, D, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            N_H, SEQ_LEN, BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO,
            ctx.HEAD_DIM, NUM_WARPS, NUM_STAGES,
        )


Q = torch.randn(1, 1, 512, 32, dtype=torch.float32, device="cuda")
K = torch.randn(1, 1, 512, 32, dtype=torch.float32, device="cuda")
V = torch.randn(1, 1, 512, 32, dtype=torch.float32, device="cuda")

scale = 1.0/math.sqrt(32)
mask = torch.tril(torch.ones(512, 512)).to(device="cuda")
P = torch.matmul(Q, K.transpose(2,3)) * scale
P.masked_fill_(torch.logical_not(mask), float("-inf"))
print(P)
P = torch.softmax(P.float(), dim=-1)

ref_O = torch.matmul(P, V)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
times = []

for _ in range(100):
    start.record()
    O = FlashAttn.forward(Q, K, V)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end)) 

median_ms = np.median(times)

print(times)
print(median_ms)


