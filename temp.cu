#include <cuda_runtime.h>
#include <stdio.h>

__global__ void naive_attention(float* Q, float* K, float* V, float* out, 
                               int seq_len, int d_model) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= seq_len || j >= d_model) return;
    
    float sum = 0.0f;
    float output_accumulator = 0.0f;
    float max_score = -1e9f;

    for (int k = 0; k < seq_len; k++) {
        float score = 0.0f;
        for (int d = 0; d < d_model; d++) {
            score += Q[i * d_model + d] * K[k * d_model + d];
        }
        
        if (score > max_score) {
            float rescale_factor = expf(max_score - score);
            sum *= rescale_factor;
            output_accumulator *= rescale_factor;
            max_score = score;
        }
        
        float exp_score = expf(score - max_score);
        sum += exp_score;
        output_accumulator += exp_score * V[k * d_model + j];
    }

    out[i * d_model + j] = output_accumulator / sum;
}

int main() {
    const int seq_len = 1024;
    const int d_model = 512;
    const int size = seq_len * d_model * sizeof(float);
    
    float *h_Q, *h_K, *h_V, *h_out;
    h_Q = (float*)malloc(size);
    h_K = (float*)malloc(size);
    h_V = (float*)malloc(size);
    h_out = (float*)malloc(size);
    
    for (int i = 0; i < seq_len * d_model; i++) {
        h_Q[i] = ((float)rand()/RAND_MAX) - 0.5f;
        h_K[i] = ((float)rand()/RAND_MAX) - 0.5f;
        h_V[i] = ((float)rand()/RAND_MAX) - 0.5f;
    }   

    float *d_Q, *d_K, *d_V, *d_out;

    cudaEvent_t start_total, end_total, start_kernel, end_kernel;
    cudaEventCreate(&start_total); cudaEventCreate(&end_total);
    cudaEventCreate(&start_kernel); cudaEventCreate(&end_kernel);
    cudaEventRecord(start_total);
    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_out, size);
    
    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((seq_len + blockSize.x - 1) / blockSize.x,
                  (d_model + blockSize.y - 1) / blockSize.y);
    
    cudaEventRecord(start_kernel);
    naive_attention<<<gridSize, blockSize>>>(d_Q, d_K, d_V, d_out, seq_len, d_model); 
    cudaEventRecord(end_kernel);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    cudaEventRecord(end_total);
    float total_ms, kernel_ms;
    cudaEventSynchronize(end_total);
    cudaEventElapsedTime(&total_ms, start_total, end_total);
    cudaEventElapsedTime(&kernel_ms, start_kernel, end_kernel);
    printf("Attention computed for %dx%d\n", seq_len, d_model);
    printf("Total: %.2f ms, Kernel: %.2f ms, Overhead: %.2f ms\n", 
        total_ms, kernel_ms, total_ms - kernel_ms);
    free(h_Q); free(h_K); free(h_V); free(h_out);
    
    return 0;
}
