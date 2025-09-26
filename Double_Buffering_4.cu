#include <cuda_bf16.h>
#include "../kernels.cuh"
#include <stdio.h>
#define BLOCK_SIZE 16 

__global__ void V4(__nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V, __nv_bfloat16* out, 
                               int seq_len, int d_model) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int globRow = by*BLOCK_SIZE+ty;
    int globCol = bx*BLOCK_SIZE+tx;
    if (globRow >= seq_len || globCol >= d_model) return;

    float accumulator = 0.0f;
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float l_i_dummy = 0.0f;
    float O_accum = 0.0f;

    __shared__ __nv_bfloat16 sQ[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sK[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sV[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_S[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float m_ij[BLOCK_SIZE];
    __shared__ float l_ij[BLOCK_SIZE];

    //
    for (int j = 0; j < seq_len / BLOCK_SIZE; ++j) {
        sQ[0][ty][tx] = Q[globRow * d_model + tx];
        sK[0][tx][ty] = K[j * BLOCK_SIZE * d_model + ty * d_model + tx];
        // S_ij
        for (int p = 0; p < d_model / BLOCK_SIZE - 1; p++){
            __syncthreads();
            int current_buf_idx = p % 2;
            int next_buf_idx = 1 - current_buf_idx;
            sQ[next_buf_idx][ty][tx] = Q[globRow * d_model + (p + 1) * BLOCK_SIZE + tx];
            sK[next_buf_idx][tx][ty] = K[(p + 1) * BLOCK_SIZE + j * BLOCK_SIZE * d_model + ty * d_model + tx];
            __syncthreads();

            for(int l = 0; l < BLOCK_SIZE; l++){
                accumulator += (float)sQ[current_buf_idx][ty][l] * (float)sK[current_buf_idx][l][tx];
            }
            __syncthreads();
        }

        int last_buf = (d_model / BLOCK_SIZE - 1) % 2;
        for (int l = 0; l < BLOCK_SIZE; ++l) {
            accumulator += (float)sQ[last_buf][ty][l] * (float)sK[last_buf][l][tx];
        }  

        float scale = 1.0f / sqrtf((float)d_model);
        accumulator *= scale;
        tile_S[ty][tx] = accumulator;
        
        float warp_val = accumulator;
        // Cette boucle est maintenant une réduction sur 16 éléments
        for (int offset=8; offset>0; offset/=2) {
            warp_val = fmaxf(warp_val, __shfl_down_sync(0xFFFFFFFF, warp_val, offset));
        }

        // Le max est maintenant dans le thread tx=0 de chaque ligne.
        // On doit le diffuser aux autres.
        float m_ij = __shfl_sync(0xFFFFFFFF, warp_val, 0);

        // --- 2. RÉDUCTION DE LA SOMME (l_ij) ---
        float m_new = fmaxf(m_i, m_ij);
        float exp_val = expf(accumulator - m_ij);
        warp_val = exp_val;

        for (int offset=8; offset>0; offset/=2) {
            warp_val += __shfl_down_sync(0xFFFFFFFF, warp_val, offset);
        }
        float l_ij = __shfl_sync(0xFFFFFFFF, warp_val, 0);
        
        l_i = l_i * expf(m_i - fmaxf(m_i, m_ij)) + l_ij * expf(m_ij - fmaxf(m_i, m_ij));

        sV[ty][tx] = V[ (j * BLOCK_SIZE + ty) * d_model + (blockIdx.x * BLOCK_SIZE + tx) ];
        tile_S[ty][tx] = exp_val;
        accumulator = 0;
        __syncthreads();

        for(int l = 0; l < BLOCK_SIZE; l++){
            accumulator += tile_S[ty][l] * (float)sV[l][tx];
        }

        __syncthreads();

        O_accum = O_accum * expf(m_i - fmaxf(m_i, m_ij)) * (l_i_dummy / l_i) + accumulator * expf(m_ij - fmaxf(m_i, m_ij)) / l_i;

        m_i = fmax(m_i, m_ij);
        l_i_dummy = l_i;
        accumulator = 0;
    }
    out[globRow * d_model + globCol] = (__nv_bfloat16)O_accum;
}
