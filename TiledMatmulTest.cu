#define BLOCK_SIZE 32
#define epsilon 1e-5
#include <stdio.h>
#include <math.h>
__global__ void tiledMatmulKernel(float* S, const float* Q, const float* K, int N,  
                               int d_model) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int globRow = by*BLOCK_SIZE+ty;
    int globCol = bx*BLOCK_SIZE+tx;
    if (globRow >= N || globCol >= N) return;
    __shared__ float sQ[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sK[BLOCK_SIZE][BLOCK_SIZE];
    float accumulator = 0.0f;
    for (int p = 0; p < d_model / BLOCK_SIZE; p++){
        sQ[ty][tx] = Q[globRow * d_model + p * BLOCK_SIZE + tx];
        sK[tx][ty] = K[(bx * BLOCK_SIZE + ty) * d_model + p * BLOCK_SIZE + tx]; //K[globCol * d_model + p * BLOCK_SIZE + ty];
        __syncthreads();
        for(int l = 0; l < BLOCK_SIZE; l++){
            accumulator += sQ[ty][l] * sK[l][tx]; //* sK[tx][l]
        }
        __syncthreads();
    } 
    S[globRow * N + globCol] = accumulator;
}


int main(){
    srand(time(NULL));
    const int seq_len = 1024;
    const int d_model = 512;
    const int size_in = seq_len * d_model * sizeof(float);
    const int size_out = seq_len * seq_len * sizeof(float);

    float *h_Q, *h_K, *h_out, *h_out_test;
    h_Q = (float*)malloc(size_in);
    h_K = (float*)malloc(size_in);
    h_out = (float*)malloc(size_out);
    h_out_test = (float*)malloc(size_out);

    for (int i = 0; i < seq_len * d_model; i++) {
        h_Q[i] = ((float)rand()/RAND_MAX) - 0.5f;
        h_K[i] = ((float)rand()/RAND_MAX) - 0.5f;
    }   

    float *d_Q, *d_K, *d_out;

    cudaMalloc(&d_Q, size_in);
    cudaMalloc(&d_K, size_in);
    cudaMalloc(&d_out, size_out);
    
    cudaMemcpy(d_Q, h_Q, size_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size_in, cudaMemcpyHostToDevice);
    
    dim3 blockSize(32, 32);
    dim3 gridSize((seq_len + blockSize.x - 1) / blockSize.x,
                  (seq_len + blockSize.y - 1) / blockSize.y);
    
    tiledMatmulKernel<<<gridSize, blockSize>>>(d_out, d_Q, d_K, seq_len, d_model); 
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_model; ++k) {
                sum += h_Q[i * d_model + k] * h_K[j * d_model + k];
            }
            h_out_test[i * seq_len + j] = sum;
        }
    }

    cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < seq_len; j++){
            if (fabs(h_out[i * seq_len + j]-h_out_test[i * seq_len + j]) > epsilon) {
                printf("Error %d, %d, %f, %f", i, j, h_out[i * seq_len + j], h_out_test[i * seq_len + j]);
                exit(0);
            }
        }
    }
    printf("Success");
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_out);
    free(h_Q); free(h_K); free(h_out);
}