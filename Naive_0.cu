#include "../kernels.cuh"


__global__ void V0_naive_attention(float* Q, float* K, float* V, float* out, 
                               int seq_len, int d_model) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= seq_len || j >= d_model) return;
    
    float sum = 0.0f;
    float max_score = -1e9f;
    
    for (int k = 0; k < seq_len; k++) {
        float score = 0.0f;
        for (int d = 0; d < d_model; d++) {
            score += Q[i * d_model + d] * K[k * d_model + d];
        }
        max_score = fmaxf(max_score, score);
    }
    
    float softmax_sum = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        float score = 0.0f;
        for (int d = 0; d < d_model; d++) {
            score += Q[i * d_model + d] * K[k * d_model + d];
        }
        softmax_sum += expf(score - max_score);
    }
    
    for (int k = 0; k < seq_len; k++) {
        float score = 0.0f;
        for (int d = 0; d < d_model; d++) {
            score += Q[i * d_model + d] * K[k * d_model + d];
        }
        float attention_weight = expf(score - max_score) / softmax_sum;
        sum += attention_weight * V[k * d_model + j];
    }
    
    out[i * d_model + j] = sum;
}