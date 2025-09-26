#include <stdio.h>
#include <cuda_bf16.h>
#include "../kernels.cuh"

#define BLOCK_SIZE 16 

__global__ void V2_coalesced(__nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V, __nv_bfloat16* out, 
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

    __shared__ __nv_bfloat16 sQ[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sK[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sV[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_S[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float m_ij[BLOCK_SIZE];
    __shared__ float l_ij[BLOCK_SIZE];
    
    // BOUCLE FLASH (externe) - Itère sur les blocs de K/V
    for (int j = 0; j < seq_len / BLOCK_SIZE; ++j) {
        // S_ij
        for (int p = 0; p < d_model / BLOCK_SIZE; p++){
            sQ[ty][tx] = Q[globRow * d_model + p * BLOCK_SIZE + tx];
            sK[tx][ty] = K[p * BLOCK_SIZE + j * BLOCK_SIZE * d_model + ty * d_model + tx];
            __syncthreads();

            for(int l = 0; l < BLOCK_SIZE; l++){
                accumulator += (float)sQ[ty][l] * (float)sK[l][tx];
            }
            __syncthreads();
        } 
        float scale = 1.0f / sqrtf((float)d_model);
        accumulator *= scale;
        tile_S[ty][tx] = accumulator;

        __syncthreads();
        
        // Réduction sur les lignes de S_ij
        for (int offset = BLOCK_SIZE/2; offset > 0; offset /= 2){
            if (tx < offset){
                tile_S[ty][tx] = fmax(tile_S[ty][tx], tile_S[ty][tx + offset]);
            }
            __syncthreads();
        }

        if (tx == 0) m_ij[ty] = tile_S[ty][0];
        __syncthreads(); //pour m_ij[ty] = tile_S[ty][0];
        accumulator = expf(accumulator - m_ij[ty]);
        tile_S[ty][tx] = accumulator;
        __syncthreads();

        for (int offset = BLOCK_SIZE/2; offset > 0; offset /= 2){
            if (tx < offset){
                tile_S[ty][tx] += tile_S[ty][tx + offset];
            }
            __syncthreads();
        }

        // m_i, l_i
        if (tx == 0) l_ij[ty] = tile_S[ty][0];
        __syncthreads();
        
        l_i = l_i * expf(m_i - fmaxf(m_i, m_ij[ty])) + l_ij[ty] * expf(m_ij[ty] - fmaxf(m_i, m_ij[ty]));

        sV[ty][tx] = V[ (j * BLOCK_SIZE + ty) * d_model + (blockIdx.x * BLOCK_SIZE + tx) ];
        tile_S[ty][tx] = accumulator;
        accumulator = 0;
        __syncthreads();

        for(int l = 0; l < BLOCK_SIZE; l++){
            accumulator += tile_S[ty][l] * (float)sV[l][tx];
        }

        __syncthreads();

        O_accum = O_accum * expf(m_i - fmaxf(m_i, m_ij[ty])) * (l_i_dummy / l_i) + accumulator * expf(m_ij[ty] - fmaxf(m_i, m_ij[ty])) / l_i;

        m_i = fmax(m_i, m_ij[ty]);
        l_i_dummy = l_i;
        accumulator = 0;
    }

    out[globRow * d_model + globCol] = (__nv_bfloat16)O_accum;
}