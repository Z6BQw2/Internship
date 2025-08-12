#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <mma.h>

#define BLOCK_SIZE 32

__global__ void flash_attention_kernel(__nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V, __nv_bfloat16* out, 
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
        tile_S[ty][tx] = accumulator;
        __syncthreads();
        
        // Réduction sur les lignes de S_ij
        for (int offset = BLOCK_SIZE/2; offset > 0; offset /= 2){
            if (tx < offset){
                tile_S[ty][tx] = max(tile_S[ty][tx], tile_S[ty][tx + offset]);
            }
            __syncthreads();
        }

        __syncthreads();
        m_ij[ty] = tile_S[ty][0];
        __syncthreads(); //pour m_ij[ty] = tile_S[ty][0];
        accumulator = expf(accumulator - m_ij[ty]);
        tile_S[ty][tx] = accumulator;

        for (int offset = BLOCK_SIZE/2; offset > 0; offset /= 2){
            if (tx < offset){
                tile_S[ty][tx] += tile_S[ty][tx + offset];
            }
            __syncthreads();
        }

        // m_i, l_i
        l_ij[ty] = tile_S[ty][0];
        l_i = l_i * expf(m_i - max(m_i, m_ij[ty])) + l_ij[ty] * expf(m_ij[ty] - max(m_i, m_ij[ty]));

        sV[ty][tx] = V[ (j * BLOCK_SIZE + ty) * d_model + (blockIdx.x * BLOCK_SIZE + tx) ];
        tile_S[ty][tx] = accumulator;
        accumulator = 0;
        __syncthreads();

        for(int l = 0; l < BLOCK_SIZE; l++){
            accumulator += tile_S[ty][l]/l_ij[ty] * (float)sV[l][tx];
        }

        __syncthreads();

        O_accum = O_accum * expf(m_i - max(m_i, m_ij[ty])) * (l_i_dummy / l_i) + accumulator * l_ij[ty] / l_i;

        m_i = max(m_i, m_ij[ty]);
        l_i_dummy = l_i;
        accumulator = 0;
    }

    out[globRow * d_model + globCol] = O_accum;
}

__global__ void flash_attention_kernel(__nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V, __nv_bfloat16* out, 
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

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::col_major> k_frag;
        wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);

        for (int p = 0; p < d_model; p += BLOCK_SIZE) {

            sQ[ty][tx] = Q[globRow * d_model + p * BLOCK_SIZE + tx];
            sK[ty][tx] = K[j * BLOCK_SIZE * d_model + p * BLOCK_SIZE + tx + ty * d_model];

            wmma::load_matrix_sync(q_frag, &sQ[0][0], BLOCK_SIZE);
            wmma::load_matrix_sync(q_frag, &sK[0][0], BLOCK_SIZE);
            wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);

            __syncthreads();
        }

        // INCOMPLET
    }

    out[globRow * d_model + globCol] = O_accum;
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
