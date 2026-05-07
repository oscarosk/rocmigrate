// Vector dot product ported to ROCm/HIP — ground truth reference

#include <hip/hip_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void vector_dot(int n, float *x, float *y, float *result) {
    __shared__ float partial[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    partial[tid] = (i < n) ? x[i] * y[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, partial[0]);
    }
}

int main(void) {
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y, *d_result;
    float result = 0.0f;

    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    hipMalloc(&d_x, N * sizeof(float));
    hipMalloc(&d_y, N * sizeof(float));
    hipMalloc(&d_result, sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    hipMemcpy(d_x, x, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_result, &result, sizeof(float), hipMemcpyHostToDevice);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_dot<<<blocks, BLOCK_SIZE>>>(N, d_x, d_y, d_result);

    hipMemcpy(&result, d_result, sizeof(float), hipMemcpyDeviceToHost);

    printf("Dot product: %f (expected: %f)\n", result, 2.0f * N);

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_result);
    free(x);
    free(y);
    return 0;
}