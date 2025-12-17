import torch
import math

import triton
import triton.language as tl

@triton.jit
def fwd(Q, K, V, O, L,
        Q_stride_B, Q_stride_H,
        Q_stride_S, Q_stride_D,
        K_stride_B, K_stride_H,
        K_stride_S, K_stride_D,
        V_stride_B, V_stride_H,
        V_stride_S, V_stride_D,
        SCALE: tl.constexpr,
        SEQ_LEN: tl.constexpr,
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
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        m_ij = tl.maximum(m_i, tl.max(QK_block, 1))

        QK_block = QK_block * SCALE - m_ij[:, None]

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


class FlashAttn(torch.autograd.Function):

    @staticmethod
    def forward( Q, K, V, causal=False, softmax_scale=None):
        if softmax_scale == None:
            softmax_scale = 1.0/math.sqrt(Q.shape[-1])

        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]

        B, N_H, SEQ_LEN = Q.shape[:-1]

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.zeros_like(Q, device="cuda")
        L = torch.zeros((B, N_H, SEQ_LEN), device="cuda")

        grid = lambda args: (
            SEQ_LEN // args["BLOCK_SIZE_Q"],
            N_H * B,
            1,
        )

        fwd[grid](Q, K, V, O, L, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                  K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                  V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                  SCALE = softmax_scale,
                  NUM_HEADS = N_H,
                  SEQ_LEN=SEQ_LEN, HEAD_DIM_Q=HEAD_DIM_Q, HEAD_DIM_KV=HEAD_DIM_K,
                  BLOCK_SIZE_Q=32,BLOCK_SIZE_KV=32)
        # print("here")
        print(O)
        # print(L)
        

Q = torch.randn(1, 1, 512, 32, dtype=torch.float16, device="cuda")
K = torch.randn(1, 1, 512, 32, dtype=torch.float16, device="cuda")
V = torch.randn(1, 1, 512, 32, dtype=torch.float16, device="cuda")

scale = 1.0/math.sqrt(32)
P = torch.matmul(Q, K.transpose(2,3)) * scale
P = torch.softmax(P.float(), dim=-1).half()

O = torch.matmul(P, V)
print(O)
FlashAttn.forward(Q, K, V)
