#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* out, 
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
    float m_i_dummy = -INFINITY;
    float l_i = 0.0f;
    float l_i_dummy = 0.0f;
    float O_accum = 0.0f;

    __shared__ float sQ[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sK[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sV[BLOCK_SIZE][BLOCK_SIZE];
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
                accumulator += sQ[ty][l] * sK[l][tx];
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
        m_i = max(m_i, m_ij[ty]);
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
        l_i = l_i * expf(m_i_dummy - m_i) + l_ij[ty] * expf(m_ij[ty] - m_i);

        sV[ty][tx] = V[ (j * BLOCK_SIZE + ty) * d_model + (blockIdx.x * BLOCK_SIZE + tx) ];
        tile_S[ty][tx] = accumulator;
        accumulator = 0;
        __syncthreads();

        for(int l = 0; l < BLOCK_SIZE; l++){
            accumulator += tile_S[ty][l]/l_ij[ty] * sV[l][tx];
        }

        __syncthreads();

        O_accum = O_accum * expf(m_i_dummy - m_i) * (l_i_dummy / l_i) + accumulator * l_ij[ty] / l_i;

        m_i_dummy = m_i;
        l_i_dummy = l_i;
        accumulator = 0;
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
