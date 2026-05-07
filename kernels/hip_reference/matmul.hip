// Naive matrix multiplication ported to ROCm/HIP — ground truth reference

#include <hip/hip_runtime.h>
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

    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 2.0f;

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, bytes_A);
    hipMalloc(&d_B, bytes_B);
    hipMalloc(&d_C, bytes_C);

    hipMemcpy(d_A, A, bytes_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, bytes_B, hipMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul<<<grid, block>>>(M, N, K, d_A, d_B, d_C);

    hipMemcpy(C, d_C, bytes_C, hipMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        max_error = fmaxf(max_error, fabsf(C[i] - 2.0f * K));
    }
    printf("Matmul max error: %f (expected ~0)\n", max_error);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(A);
    free(B);
    free(C);
    return 0;
}