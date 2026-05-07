// Naive matrix multiplication: C = A * B
// A is M x K, B is K x N, C is M x N
//
// Each thread computes one element of C using a 2D thread grid.
// "Naive" because we read directly from global memory without
// shared-memory tiling — straightforward, but not peak performance.
// (A tiled version with shared memory is the natural optimization
// target for the agent's CDNA-tuning stage.)

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul(int M, int N, int K, const float *A, const float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

int main(void) {
    const int M = 256;
    const int N = 256;
    const int K = 256;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *A = (float*)malloc(bytes_A);
    float *B = (float*)malloc(bytes_B);
    float *C = (float*)malloc(bytes_C);

    // Fill A and B with simple values
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 2.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul<<<grid, block>>>(M, N, K, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // Each C[i,j] = sum over k of (1.0 * 2.0) = 2.0 * K = 512
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        max_error = fmaxf(max_error, fabsf(C[i] - 2.0f * K));
    }
    printf("Matmul max error: %f (expected ~0)\n", max_error);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    return 0;
}