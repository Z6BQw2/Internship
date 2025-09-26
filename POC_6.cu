#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include "../kernels.cuh"

#include <math.h>
#include <mma.h>

using namespace nvcuda;

#define BLOCK_SIZE 16
#define PADDED_D (512 + 8)

__global__ void V6(__nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V, __nv_bfloat16* out, 
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

    float max_A;
    float max_B;
    float max_A_i = -INFINITY;
    float max_B_i = -INFINITY;
    float sum_A;
    float sum_B;
    float sum_A_i = 0.0f;
    float sum_B_i = 0.0f;

    __shared__ float s_O_accum[32][8];

    for(int i = 0; i < 8; i++) {
        s_O_accum[threadIdx.x][i] = 0.0f;
    }

    __shared__ __nv_bfloat16 sQ[BLOCK_SIZE][PADDED_D];
    __shared__ __nv_bfloat16 sK[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sV[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sP[BLOCK_SIZE][BLOCK_SIZE]; // Necessary because we tile_S is the wrong type, and wmma doesn't offer the possibility to convert it during loading
    __shared__ float tile_S[BLOCK_SIZE][BLOCK_SIZE];

    for (int i = tx; i < d_model; i += blockDim.x) {
        sQ[ty][i] = Q[globRow * d_model + i];
    }

    //
    for (int j = 0; j < seq_len / BLOCK_SIZE; ++j) {

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> k_frag;
        wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> work_frag;
        wmma::fill_fragment(work_frag, 0.0f);

        sK[0][tx][ty] = K[j * BLOCK_SIZE * d_model + ty * d_model + tx];

        // S_ij
        for (int p = 0; p < (d_model / BLOCK_SIZE) - 1; p++){
            int current_buf = p % 2;
            int next_buf = 1 - current_buf;
            __syncthreads();

            wmma::load_matrix_sync(q_frag, &sQ[0][p * BLOCK_SIZE], PADDED_D);
            wmma::load_matrix_sync(k_frag, &sK[current_buf][0][0], BLOCK_SIZE);
            
            sK[next_buf][tx][ty] = K[(p + 1) * BLOCK_SIZE + j * BLOCK_SIZE * d_model + ty * d_model + tx];

            wmma::mma_sync(work_frag, q_frag, k_frag, work_frag);
        }

        int last_buf_idx = (d_model / BLOCK_SIZE - 1) % 2;
        wmma::load_matrix_sync(q_frag, &sQ[0][(d_model / BLOCK_SIZE - 1) * BLOCK_SIZE], PADDED_D);
        wmma::load_matrix_sync(k_frag, &sK[last_buf_idx][0][0], BLOCK_SIZE);
        wmma::mma_sync(work_frag, q_frag, k_frag, work_frag);

        float scale = 1.0f / sqrtf((float)d_model);
        for(int i = 0; i < 8; i++) work_frag.x[i] *= scale;

        ///////////////////////////////////////////////////////////////////////////////////////

        float temp_f;

        temp_f = work_frag.x[2];
        work_frag.x[2] = work_frag.x[4];
        work_frag.x[4] = temp_f;

        temp_f = work_frag.x[3];
        work_frag.x[3] = work_frag.x[5];
        work_frag.x[5] = temp_f;
        
        max_A = fmaxf(fmaxf(work_frag.x[0], work_frag.x[1]), fmaxf(work_frag.x[2], work_frag.x[3]));
        max_B = fmaxf(fmaxf(work_frag.x[4], work_frag.x[5]), fmaxf(work_frag.x[6], work_frag.x[7]));

        float partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_A, 1);
        float partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_B, 1);

        max_A = fmaxf(max_A, partner_max_A);
        max_B = fmaxf(max_B, partner_max_B);

        partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_A, 2);
        partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_B, 2);

        max_A = fmaxf(max_A, partner_max_A);
        max_B = fmaxf(max_B, partner_max_B);

        // --- 2. RÉDUCTION DE LA SOMME (l_ij) ---
        float m_new_A = fmaxf(max_A_i, max_A);
        float m_new_B = fmaxf(max_B_i, max_B);

        sum_A = 0.0f;
        sum_B = 0.0f;

        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            work_frag.x[i] = expf(work_frag.x[i] - max_A);
            sum_A += work_frag.x[i];
        }

        #pragma unroll
        for(int i = 4; i < 8; ++i) {
            work_frag.x[i] = expf(work_frag.x[i] - max_B);
            sum_B += work_frag.x[i];
        }

        partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_A, 1);
        partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_B, 1);

        sum_A += partner_max_A;
        sum_B += partner_max_B;

        partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_A, 2);
        partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_B, 2);

        sum_A += partner_max_A;
        sum_B += partner_max_B; 

        float old_sum_A_i = sum_A_i; ///Remb (assignation potentiellement inutile)
        float old_sum_B_i = sum_B_i;

        sum_A_i = old_sum_A_i * expf(max_A_i - m_new_A) + sum_A * expf(max_A - m_new_A);
        sum_B_i = old_sum_B_i * expf(max_B_i - m_new_B) + sum_B * expf(max_B - m_new_B);
        
        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> p_frag;

        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            p_frag.x[i] = __float2bfloat16(work_frag.x[i] / sum_A); //Remb (/sum)
        }

        #pragma unroll
        for(int i = 4; i < 8; ++i) {
            p_frag.x[i] = __float2bfloat16(work_frag.x[i] / sum_B);
        }

        temp_f = p_frag.x[2];
        p_frag.x[2] = p_frag.x[4];
        p_frag.x[4] = temp_f;

        temp_f = p_frag.x[3];
        p_frag.x[3] = p_frag.x[5];
        p_frag.x[5] = temp_f;

        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> v_frag;
        sV[ty][tx] = V[ (j * BLOCK_SIZE + ty) * d_model + (blockIdx.x * BLOCK_SIZE + tx) ];
        __syncthreads();
        wmma::load_matrix_sync(v_frag, &sV[0][0], 16);

        wmma::fill_fragment(work_frag, 0.0f);

        wmma::mma_sync(work_frag, p_frag, v_frag, work_frag);

        // O_accum = O_accum * expf(m_i - fmaxf(m_i, m_ij)) * (l_i_dummy / l_i) + tile_S[ty][tx] * expf(m_ij - fmaxf(m_i, m_ij)) / l_i;

        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            s_O_accum[threadIdx.x][i] = s_O_accum[threadIdx.x][i] * (old_sum_A_i / sum_A_i) * expf(max_A_i - fmax(max_A_i, max_A)) + work_frag.x[i] * (sum_A / sum_A_i);
        }

        #pragma unroll
        for(int i = 4; i < 8; ++i) {
            s_O_accum[threadIdx.x][i] = s_O_accum[threadIdx.x][i] * (old_sum_B_i / sum_B_i) * expf(max_B_i - fmax(max_B_i, max_B)) + work_frag.x[i] * (sum_B / sum_B_i);
        }

        max_A_i = m_new_A;
        max_B_i = m_new_B;
    }
    float temp_f = s_O_accum[threadIdx.x][2];
    s_O_accum[threadIdx.x][2] = s_O_accum[threadIdx.x][4];
    s_O_accum[threadIdx.x][4] = temp_f;

    temp_f = s_O_accum[threadIdx.x][3];
    s_O_accum[threadIdx.x][3] = s_O_accum[threadIdx.x][5];
    s_O_accum[threadIdx.x][5] = temp_f;

    // Écriture avec les formules de mapping inverse
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        out[(blockIdx.y * 16 + (threadIdx.x / 4) + (((i >> 1) & 1) * 8)) * d_model + blockIdx.x * 16 + ((threadIdx.x % 4) * 2) + (i % 2) + (((i >> 2) & 1) * 8)] = __float2bfloat16(s_O_accum[threadIdx.x][i]);
    }
}