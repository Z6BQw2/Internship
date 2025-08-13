//This code's purpose is identifying the memory paterns of wmma's fragments.

#include <cuda.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

#define BLOCK_SIZE 16
using namespace nvcuda;

__global__ void matmul_wmma(){
    __shared__ __nv_bfloat16 A[16][16];
    A[threadIdx.x][threadIdx.y] = threadIdx.x * 100 + threadIdx.y;
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::load_matrix_sync(a_frag, &A[0][0], 16);
    __syncthreads();
    if(threadIdx.x == 10 && threadIdx.y == 1){
        for(int i=0; i<8; i++){
            printf("%f", (float)a_frag.x[i]);
        }
    }
}



int main() {
    dim3 blockSize(16, 16);
    dim3 gridSize(1, 1);

    matmul_wmma<<<gridSize, blockSize>>>();
    return 0;
}