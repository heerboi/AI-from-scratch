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
        BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr, STAGE: tl.constexpr):


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


        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block) * SCALE
        
        if STAGE == 3:
            mask = q_offsets[:, None] >= kv_indices[None, :]
            QK_block = tl.where(mask, QK_block, -1e9)

        m_ij = tl.maximum(m_i, tl.max(QK_block, 1))

        QK_block = QK_block - m_ij[:, None]

        P_block = tl.math.exp(QK_block)

        l_ij = tl.sum(P_block, 1)
        correction = tl.math.exp(m_i - m_ij)

        l_i = l_ij + l_i * correction

        V_block = tl.load(V_block_ptr)

        P_block = P_block.to(tl.float16)

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
def _attn_bwd_dq(Q, K, V, softmax_scale, dO, dQ, D, L, stride_batch, stride_head, stride_seq, stride_dim,
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
    dQ += offset_head_batch

    L += offset_head_batch_seq
    D += offset_head_batch_seq
    
    head_dim_offsets = tl.arange(0, HEAD_DIM)

    q_block_idx = tl.program_id(0)
    q_block_start = q_block_idx * BLOCK_Q

    q_offsets = q_block_start + tl.arange(0, BLOCK_Q)

    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype = tl.float32)

    Q_block = tl.load(
        Q + q_offsets[:, None] * stride_seq + head_dim_offsets[None, :] * stride_dim
    )

    kv_offsets = tl.arange(0, BLOCK_KV)
    dO_block = tl.load(dO + q_offsets[:, None] * stride_seq + head_dim_offsets[None, :] * stride_dim)

    L_block = tl.load(L + q_offsets)
    L_block = L_block[:, None]

    kT_ptrs = K + kv_offsets[None, :] * stride_seq + head_dim_offsets[:, None] * stride_dim
    vT_ptrs = V + kv_offsets[None, :] * stride_seq + head_dim_offsets[:, None] * stride_dim

    Di = tl.load(D + q_offsets)

    kv_ctr = 0
    num_steps = SEQ_LEN // BLOCK_KV

    for step in range(num_steps):

        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)

        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - L_block)

        if STAGE == 3:
            kv_offsets = kv_ctr + tl.arange(0, BLOCK_KV)
            mask_block = (q_offsets[:, None] >= kv_offsets[None, :])
            P_block = tl.where(mask_block, P_block, 0.0)

        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)

        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

        kv_ctr += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq
    
    dQ_block_ptrs = dQ + q_offsets[:, None] * stride_seq + head_dim_offsets[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)

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
        dO_block = tl.load(dO_ptrs)

        q_offsets = q_ctr + tl.arange(0, BLOCK_Q)

        # (MICRO, )
        l = tl.load(L + q_offsets)

        # (MACRO, MICRO)
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block) 

        # apply the logsumexp across the columns since QK^T is transposed
        # bit confusing fkjdsl
        P_T_block = tl.math.exp(QK_T_block - l[None, :])

        if STAGE == 3:

            mask_block = (
                q_offsets[None, :] >= kv_offsets[:, None]
            )
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        Di = tl.load(D + q_offsets)

        dPt_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        dS_T_block = P_T_block * (dPt_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))

        q_ctr += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    dV_block_ptrs = dV + kv_offsets[:, None] * stride_seq + head_dim_offsets[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    dK_block_ptrs = dK + kv_offsets[:, None] * stride_seq + head_dim_offsets[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)

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
        STAGE = 3 if causal else 1
        
        kernel_warmup = fwd.warmup(Q, K, V, O, L, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                  K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                  V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                  SEQ_LEN=SEQ_LEN,
                  SCALE = softmax_scale,
                  NUM_HEADS = N_H, HEAD_DIM_Q=HEAD_DIM_Q, HEAD_DIM_KV=HEAD_DIM_K,
                  BLOCK_SIZE_Q=32,BLOCK_SIZE_KV=32, grid=(1,), STAGE=STAGE)

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
                  BLOCK_SIZE_Q=32,BLOCK_SIZE_KV=32, STAGE=STAGE)
        
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
        stage = 3 if ctx.causal else 1

        _attn_bwd_dk_dv[grid](
            Q, K, V,
            ctx.softmax_scale, dO, dK, dV, D, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            N_H, SEQ_LEN, BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO,
            ctx.HEAD_DIM, STAGE=stage, num_stages=NUM_STAGES, num_warps=NUM_WARPS,
        )

        _attn_bwd_dq[grid](
            Q, K, V,
            ctx.softmax_scale, dO, dQ, D, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            N_H, SEQ_LEN, BLOCK_SIZE_MACRO, BLOCK_SIZE_MICRO,
            ctx.HEAD_DIM, STAGE=stage, num_stages=NUM_STAGES, num_warps=NUM_WARPS,
        )

        return dQ, dK, dV, None, None


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = FlashAttn.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_op(BATCH_SIZE=2, NUM_HEADS=2, SEQ_LEN=256, HEAD_DIM=32, causal=True)
    test_op(BATCH_SIZE=2, NUM_HEADS=2, SEQ_LEN=256, HEAD_DIM=32, causal=False)
    print("PASSED")