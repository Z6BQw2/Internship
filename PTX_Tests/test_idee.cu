#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <math.h>
#include <mma.h>

using namespace nvcuda;

#define BLOCK_SIZE 16

__global__ void flash_attention_kernel(__nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V, __nv_bfloat16* out, 
                               int seq_len, int d_model) {
    // int bx = blockIdx.x;
    // int by = blockIdx.y;
    // int tx = threadIdx.x;
    // int ty = threadIdx.y;
    // int globRow = by*BLOCK_SIZE+ty;
    // int globCol = bx*BLOCK_SIZE+tx;

    if (blockIdx.x*BLOCK_SIZE >= seq_len || blockIdx.y*BLOCK_SIZE >= seq_len) return;

    float O_accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    unsigned int* vals;
    float max_A;
    float max_B;
    float max_A_i = -INFINITY;
    float max_B_i = -INFINITY;
    float sum_A;
    float sum_B;
    float sum_A_i = 0.0f;
    float sum_B_i = 0.0f;
    float sum_A_dummy = 0.0f;
    float sum_B_dummy = 0.0f;

    __shared__ __nv_bfloat16 sQ[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sK[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sV[BLOCK_SIZE][BLOCK_SIZE];

    wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> o_frag;
    wmma::fill_fragment(o_frag, 0.0f);
    
    // BOUCLE FLASH (externe) - Itère sur les blocs de K/V
    for (int j = 0; j < seq_len / BLOCK_SIZE; ++j) {

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::col_major> k_frag;
        wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);

        for (int p = 0; p < d_model; p += BLOCK_SIZE) {

            sQ[threadIdx.y][threadIdx.x] = Q[blockIdx.y*BLOCK_SIZE+threadIdx.y * d_model + p + threadIdx.x];
            sK[threadIdx.y][threadIdx.x] = K[j * BLOCK_SIZE * d_model + p + threadIdx.x + threadIdx.y * d_model];

            __syncthreads();

            wmma::load_matrix_sync(q_frag, &sQ[0][0], BLOCK_SIZE);
            wmma::load_matrix_sync(k_frag, &sK[0][0], BLOCK_SIZE);
            wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);

        }

        float temp_f;

        // Échange le bloc {x[2], x[3]} avec le bloc {x[4], x[5]}
        temp_f = acc_frag.x[2];
        acc_frag.x[2] = acc_frag.x[4];
        acc_frag.x[4] = temp_f;

        temp_f = acc_frag.x[3];
        acc_frag.x[3] = acc_frag.x[5];
        acc_frag.x[5] = temp_f;
        
        max_A = fmaxf(fmaxf(acc_frag.x[0], acc_frag.x[1]), fmaxf(acc_frag.x[2], acc_frag.x[3]));
        max_B = fmaxf(fmaxf(acc_frag.x[4], acc_frag.x[5]), fmaxf(acc_frag.x[6], acc_frag.x[7]));

        float partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_A, 1);
        float partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_B, 1);

        max_A = fmaxf(max_A, partner_max_A);
        max_B = fmaxf(max_B, partner_max_B);

        partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_A, 2);
        partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_B, 2);

        max_A = fmaxf(max_A, partner_max_A);
        max_B = fmaxf(max_B, partner_max_B);

        sum_A = 0.0f;
        sum_B = 0.0f;
        for(int i = 0; i < 4; ++i) {
            acc_frag.x[i] = expf(acc_frag.x[i] - max_A);
            sum_A += acc_frag.x[i];
        }
        for(int i = 4; i < 8; ++i) {
            acc_frag.x[i] = expf(acc_frag.x[i] - max_B);
            sum_B += acc_frag.x[i];
        }

        partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_A, 1);
        partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_B, 1);

        sum_A += partner_max_A;
        sum_B += partner_max_B;

        partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_A, 2);
        partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_B, 2);

        sum_A += partner_max_A;
        sum_B += partner_max_B; 

        sum_A_i = sum_A_i * expf(max_A_i - fmax(max_A_i, max_A)) + sum_A * expf(max_A - max(max_A_i, max_A));
        sum_B_i = sum_B_i * expf(max_B_i - fmax(max_B_i, max_B)) + sum_B * expf(max_B - max(max_B_i, max_B));

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> p_frag;

        for(int i = 0; i < 4; ++i) {
            p_frag.x[i] = __float2bfloat16(acc_frag.x[i] / sum_A);
        }
        for(int i = 4; i < 8; ++i) {
            p_frag.x[i] = __float2bfloat16(acc_frag.x[i] / sum_B);
        }

        vals = reinterpret_cast<unsigned int*>(p_frag.x);
        unsigned int temp = vals[1];
        vals[1] = vals[2];
        vals[2] = temp;

        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> v_frag;

        for (int p = 0; p < d_model; p += BLOCK_SIZE) {

            sV[threadIdx.y][threadIdx.x] = V[(j * BLOCK_SIZE + threadIdx.y) * d_model + (p + threadIdx.x)];
            __syncthreads();
            
            wmma::load_matrix_sync(v_frag, &sV[0][0], BLOCK_SIZE);
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            
            __syncthreads();
        }

        __syncthreads();

        for(int i = 0; i < 4; ++i) {
            O_accum[i] = O_accum[i] * expf(max_A_i - fmax(max_A_i, max_A)) * (sum_A_dummy / sum_A_i) + o_frag.x[i] * sum_A / sum_A_i;
        }
        for(int i = 4; i < 8; ++i) {
            O_accum[i] = O_accum[i] * expf(max_B_i - fmax(max_B_i, max_B)) * (sum_B_dummy / sum_B_i) + o_frag.x[i] * sum_B / sum_B_i;
        }

        max_A_i = fmax(max_A_i, max_A);
        max_B_i = fmax(max_B_i, max_B);
    }

    // out[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * d_model + blockIdx.x * BLOCK_SIZE + threadIdx.x] = O_accum; à adapter pour les saleté de patternes de frag
}


int main() {
    const int seq_len = 1024;
    const int d_model = 512;
    const int size = seq_len * d_model * sizeof(float);
    const int warmup_rounds = 20;
    const int run = 10;

    float *h_Q, *h_K, *h_V, *h_out;
    h_Q = (float*)malloc(size);
    h_K = (float*)malloc(size);
    h_V = (float*)malloc(size);
    h_out = (float*)malloc(size);

    srand(42); // Use a fixed seed for reproducibility
    for (int i = 0; i < seq_len * d_model; i++) {
        h_Q[i] = ((float)rand()/RAND_MAX) - 0.5f;
        h_K[i] = ((float)rand()/RAND_MAX) - 0.5f;
        h_V[i] = ((float)rand()/RAND_MAX) - 0.5f;
    }

    float *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_out, size);

    cudaEvent_t start_total, end_total, start_kernel, end_kernel;
    cudaEventCreate(&start_total);
    cudaEventCreate(&end_total);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);

    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize(d_model/32, seq_len/32);

    // --- WARM-UP ROUNDS ---
    printf("Performing %d warm-up rounds...\n", warmup_rounds);
    for (int i = 0; i < warmup_rounds; ++i) {
        flash_attention_kernel<<<gridSize, blockSize>>>(d_Q, d_K, d_V, d_out, seq_len, d_model);
    }

    cudaDeviceSynchronize();
    printf("Warm-up complete. Performing timed execution...\n");

    cudaEventRecord(start_total);

    cudaEventRecord(start_kernel);
    for (int i = 0; i < run; ++i) {
        flash_attention_kernel<<<gridSize, blockSize>>>(d_Q, d_K, d_V, d_out, seq_len, d_model);
    }
    cudaEventRecord(end_kernel);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(end_total);
    cudaEventSynchronize(end_total);

    float total_ms, kernel_ms;
    cudaEventElapsedTime(&total_ms, start_total, end_total);
    cudaEventElapsedTime(&kernel_ms, start_kernel, end_kernel);

    // Print results
    printf("\n--- Results ---\n");
    printf("Attention computed for %dx%d\n", seq_len, d_model);
    printf("Total Time (one D2H transfer and 10 runs): %.3f ms\n", total_ms);
    printf("Kernel Execution Time (after warm-up):    %.3f ms\n", kernel_ms / run);

    free(h_Q); free(h_K); free(h_V); free(h_out);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    cudaEventDestroy(start_total);
    cudaEventDestroy(end_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(end_kernel);

    return 0;
}