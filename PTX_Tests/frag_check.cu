#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h> // On utilise __half
#include <mma.h>
#include <vector> // Pour stocker le mapping
#include <iomanip> // Pour std::setw

using namespace nvcuda;

#define M_SIZE 16

// Le kernel reste le même, il fait son travail de chargement et de stockage brut.
__global__ void matmul_wmma(__half* B_out) {
    __shared__ __half A_sh[M_SIZE][M_SIZE];
    __shared__ __half id_sh[M_SIZE][M_SIZE];

    unsigned int row = threadIdx.y;
    unsigned int col = threadIdx.x;
    A_sh[row][col] = __float2half((float)(row * 100 + col));
    id_sh[row][col] = __float2half((float)(row == col));

    __syncthreads();

    // On se limite au premier warp (threads 0-31)
    if (threadIdx.y < 2 && threadIdx.x < 16) {
        wmma::fragment<wmma::matrix_a, M_SIZE, M_SIZE, M_SIZE, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, M_SIZE, M_SIZE, M_SIZE, __half, wmma::col_major> id_frag;
        wmma::fragment<wmma::accumulator, M_SIZE, M_SIZE, M_SIZE, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);

        wmma::load_matrix_sync(a_frag, &A_sh[0][0], M_SIZE);
        wmma::load_matrix_sync(id_frag, &id_sh[0][0], M_SIZE);

        wmma::mma_sync(acc_frag, a_frag, id_frag, acc_frag);

        unsigned int lane_id = threadIdx.x + threadIdx.y * blockDim.x;

        for (int i = 0; i < 8; i++) {
            B_out[lane_id * 8 + i] = acc_frag.x[i];
        }
    }
}

struct MappingInfo {
    int lane_id = -1;  // Thread du warp (0-31) qui détient l'élément
    int frag_idx = -1; // Index dans le registre du fragment (0-7)
};

int main() {
    const int num_threads_in_warp = 32;
    const int values_per_thread = 8;
    const int total_values = num_threads_in_warp * values_per_thread; // 32 * 8 = 256

    __half *h_B, *d_B;
    h_B = (__half*)malloc(total_values * sizeof(__half));
    cudaMalloc(&d_B, total_values * sizeof(__half));

    dim3 blockSize(16, 16);
    dim3 gridSize(1, 1);

    matmul_wmma<<<gridSize, blockSize>>>(d_B);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, total_values * sizeof(__half), cudaMemcpyDeviceToHost);

    // --- ANALYSE DES DONNÉES SUR LE CPU ---

    // On va construire notre table de mapping inversé ici
    MappingInfo map[M_SIZE][M_SIZE];

    printf("--- PARTIE 1: Contenu brut des fragments avec coordonnées d'origine ---\n\n");
    for (int thread = 0; thread < num_threads_in_warp; thread++) {
        printf("Thread %2d (lane %2d): ", thread, thread);
        for (int i = 0; i < values_per_thread; i++) {
            float val_f = __half2float(h_B[thread * values_per_thread + i]);
            int row = (int)val_f / 100;
            int col = (int)val_f % 100;
            
            // On affiche la valeur et sa provenance
            printf("%4.0f (A[%02d][%02d])  ", val_f, row, col);

            // On remplit notre table de mapping pour plus tard
            if (row < M_SIZE && col < M_SIZE) {
                map[row][col].lane_id = thread;
                map[row][col].frag_idx = i;
            }
        }
        printf("\n");
    }

    printf("\n\n--- PARTIE 2: Tableau de mapping inversé ---\n");
    printf("Pour chaque élément de A, qui le détient et où ?\n\n");
    for (int row = 0; row < M_SIZE; row++) {
        for (int col = 0; col < M_SIZE; col++) {
            printf("A[%02d][%02d] -> %2d, %d\n", 
                   row, col, map[row][col].lane_id, map[row][col].frag_idx);
        }
        printf("\n");
    }

    printf("\n\n--- PARTIE 3: Graphe visuel du mapping des 'lane ID' sur la matrice A ---\n");
    printf("Chaque cellule (row, col) contient le lane_id (0-31) du thread qui la gère.\n\n");

    // En-tête des colonnes
    printf("      ");
    for (int col = 0; col < M_SIZE; col++) {
        printf(" C%02d", col);
    }
    printf("\n      ");
    for (int col = 0; col < M_SIZE; col++) {
        printf("----");
    }
    printf("\n");

    // Lignes de la matrice
    for (int row = 0; row < M_SIZE; row++) {
        printf("L%02d | ", row);
        for (int col = 0; col < M_SIZE; col++) {
            printf("%3d ", map[row][col].lane_id);
        }
        printf("\n");
    }

    free(h_B);
    cudaFree(d_B);

    return 0;
}