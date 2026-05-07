// SAXPY: Single-precision A * X Plus Y
// Computes: y[i] = a * x[i] + y[i] for each element
//
// This is the "hello world" of GPU programming and serves as our
// first test kernel for the Rocmigrate agent. The agent's job is
// to port this CUDA code to AMD ROCm/HIP.

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main(void) {
    int N = 1 << 20;  // 1M elements
    float *x, *y, *d_x, *d_y;

    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify: every element should be 2.0 * 1.0 + 2.0 = 4.0
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        max_error = fmaxf(max_error, fabsf(y[i] - 4.0f));
    }
    printf("Max error: %f\n", max_error);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    return 0;
}