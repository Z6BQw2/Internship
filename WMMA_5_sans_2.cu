#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <math.h>
#include <mma.h>
#include "../kernels.cuh"


using namespace nvcuda;

#define BLOCK_SIZE 16
#define PADDED_D (512 + 8)

__global__ void V5_sans_2(__nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V, __nv_bfloat16* out, 
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

    __shared__ __nv_bfloat16 sQ[BLOCK_SIZE][PADDED_D];
    __shared__ __nv_bfloat16 sK[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sV[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_S[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float m_ij[BLOCK_SIZE];
    __shared__ float l_ij[BLOCK_SIZE];

    for (int i = tx; i < d_model; i += blockDim.x) {
        sQ[ty][i] = Q[globRow * d_model + i];
    }

    //
    for (int j = 0; j < seq_len / BLOCK_SIZE; ++j) {

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> k_frag;
        wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> work_frag;
        wmma::fill_fragment(work_frag, 0.0f);

        sK[0][ty][tx] = K[(j * BLOCK_SIZE + tx) * d_model + ty];

        // S_ij
        for (int p = 0; p < (d_model / BLOCK_SIZE) - 1; p++){
            int current_buf = p % 2;
            int next_buf = 1 - current_buf;
            __syncthreads();

            wmma::load_matrix_sync(q_frag, &sQ[0][p * BLOCK_SIZE], PADDED_D);
            wmma::load_matrix_sync(k_frag, &sK[current_buf][0][0], BLOCK_SIZE);
            
            sK[next_buf][ty][tx] = K[(j * BLOCK_SIZE + tx) * d_model + (p + 1) * BLOCK_SIZE + ty];

            wmma::mma_sync(work_frag, q_frag, k_frag, work_frag);
        }

        int last_buf_idx = (d_model / BLOCK_SIZE - 1) % 2;
        wmma::load_matrix_sync(q_frag, &sQ[0][(d_model / BLOCK_SIZE - 1) * BLOCK_SIZE], PADDED_D);
        wmma::load_matrix_sync(k_frag, &sK[last_buf_idx][0][0], BLOCK_SIZE);
        wmma::mma_sync(work_frag, q_frag, k_frag, work_frag);

        float scale = 1.0f / sqrtf((float)d_model);
        for(int i = 0; i < 8; i++) work_frag.x[i] *= scale;

        wmma::store_matrix_sync(&tile_S[0][0], work_frag, BLOCK_SIZE, wmma::mem_row_major);

        accumulator = tile_S[ty][tx];

        __syncthreads();
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

        sV[tx][ty] = V[ (j * BLOCK_SIZE + tx) * d_model + (bx * BLOCK_SIZE + ty) ];
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

