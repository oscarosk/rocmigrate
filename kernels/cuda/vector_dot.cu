// Vector dot product: computes sum of element-wise products of two vectors.
// dot = sum(x[i] * y[i]) for i in [0, N)
//
// Uses shared memory and a parallel reduction within each block,
// then atomicAdd to combine block results.

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void vector_dot(int n, float *x, float *y, float *result) {
    __shared__ float partial[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    partial[tid] = (i < n) ? x[i] * y[i] : 0.0f;
    __syncthreads();

    // Parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    // Block leader writes its partial sum to global result
    if (tid == 0) {
        atomicAdd(result, partial[0]);
    }
}

int main(void) {
    int N = 1 << 20;  // 1M elements
    float *x, *y, *d_x, *d_y, *d_result;
    float result = 0.0f;

    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_dot<<<blocks, BLOCK_SIZE>>>(N, d_x, d_y, d_result);

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: N * (1.0 * 2.0) = 2N = 2097152
    printf("Dot product: %f (expected: %f)\n", result, 2.0f * N);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    free(x);
    free(y);
    return 0;
}