#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <math.h>
#include <mma.h>

#define BLOCK_SIZE 16
using namespace nvcuda;

__global__ void matmul_wmma(float* C){
    __shared__ __nv_bfloat16 A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 B[BLOCK_SIZE][BLOCK_SIZE];

    if (threadIdx.x < BLOCK_SIZE && threadIdx.y < BLOCK_SIZE){
        A[threadIdx.x][threadIdx.y] = (__nv_bfloat16)(threadIdx.x + threadIdx.y);
        B[threadIdx.x][threadIdx.y] = (__nv_bfloat16)(threadIdx.x * 10 + threadIdx.y * 10);
    }

    wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, &A[0][0], BLOCK_SIZE);
    wmma::load_matrix_sync(b_frag, &B[0][0], BLOCK_SIZE);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    wmma::store_matrix_sync(&C[0], acc_frag, BLOCK_SIZE, wmma::mem_row_major);
}



int main() {
    float *C;
    cudaMalloc(&C, BLOCK_SIZE*BLOCK_SIZE*sizeof(float));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 blockSize(16, 2);
    dim3 gridSize(1, 1);

    cudaEventRecord(start);
    matmul_wmma<<<gridSize, blockSize>>>(C);
    cudaEventRecord(end);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, end);
    printf("%f", total_ms);

    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}