#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdexcept>

#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",                        \
                cudaGetErrorString(err), __FILE__, __LINE__);               \
        throw std::runtime_error(cudaGetErrorString(err));                  \
    }                                                                       \
} while (0)

#define TILE_DIM 16

__global__ void test_bit_shifting_kernel(const half* A, half* B, int lda, int ldb) 
{
    using namespace nvcuda;

    int warp_m = blockIdx.x;
    int warp_n = blockIdx.y;

    int start_row = warp_m * TILE_DIM;
    int start_col = warp_n * TILE_DIM;

    wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, half> test_frag;

    if (start_row < lda && start_col < ldb) {
        wmma::load_matrix_sync(test_frag, A + start_row * lda + start_col, lda, wmma::mem_row_major);
        
        unsigned int* vals = reinterpret_cast<unsigned int*>(test_frag.x);
        
        unsigned int temp = vals[1];
        vals[1] = vals[2];
        vals[2] = temp;
        
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            unsigned int int_representation = vals[1];

            unsigned short low_bits = int_representation & 0xFFFF;

            unsigned short high_bits = int_representation >> 16;

            half low_half  = *reinterpret_cast<half*>(&low_bits);
            half high_half = *reinterpret_cast<half*>(&high_bits);

            printf("--- Contre-Interrogatoire Thread 0 ---\n");
            printf("Valeur Entiere Observee: %u\n", int_representation);
            printf("Bits Hauts (shifte)   -> Valeur Flottante: %f\n", __half2float(high_half));
            printf("Bits Bas              -> Valeur Flottante: %f\n", __half2float(low_half));
            printf("---------------------------------------\n");
        }

        temp = vals[1];
        vals[1] = vals[2];
        vals[2] = temp;

        wmma::store_matrix_sync(B + start_row * ldb + start_col, test_frag, ldb, wmma::mem_row_major);
    }
}

int main() {
    const int M = 32;
    const int N = 32;

    std::cout << "Initialisation des matrices host..." << std::endl;
    std::vector<half> h_A(M * N);
    std::vector<half> h_B(M * N, __float2half(0.0f));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = __float2half(static_cast<float>(i * N + j + 1));
        }
    }

    half *d_A, *d_B;
    std::cout << "Allocation de la mémoire device..." << std::endl;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, M * N * sizeof(half)));

    std::cout << "Copie de la matrice A vers le device..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(half), cudaMemcpyHostToDevice));

    dim3 blockDim(32); // 1 warp par bloc
    dim3 gridDim(M / TILE_DIM, N / TILE_DIM);

    std::cout << "Lancement du kernel de contre-interrogatoire..." << std::endl;

    test_bit_shifting_kernel<<<gridDim, blockDim>>>(d_A, d_B, N, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "Copie de la matrice B vers l'host..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    std::cout << "Vérification des résultats..." << std::endl;
    bool match = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float val_a = __half2float(h_A[i * N + j]);
            float val_b = __half2float(h_B[i * N + j]);
            if (val_a != val_b) {
                std::cout << "!!! FAILURE !!!" << std::endl;
                std::cout << "Mismatch at (" << i << ", " << j << "): A=" << val_a << ", B=" << val_b << std::endl;
                match = false;
                goto verification_end; 
            }
        }
    }

verification_end:
    if (match) {
        std::cout << ">>> SUCCESS <<<" << std::endl;
        std::cout << "Les matrices A et B sont identiques. La réversibilité est confirmée." << std::endl;
    }

    std::cout << "Nettoyage..." << std::endl;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return 0;
}