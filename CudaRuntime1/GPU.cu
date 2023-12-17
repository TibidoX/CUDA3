#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>

#define BLOCK_SIZE 16

__global__ void multGPU(float* A, float* B, float* res, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) { /////////////////////
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        res[row * n + col] = sum;
    }
}

__global__ void multOptimized(float* A, float* B, float* res, int n) {
    int aBegin = n * blockDim.y * blockIdx.y;
    int aEnd = aBegin + n - 1;
    int aStep = blockDim.x;
    int bBegin = blockDim.x * blockIdx.x;
    int bStep = blockDim.y * n;
    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];
    float sum = 0;

    for (int ia = aBegin, ib = bBegin; ia < aEnd; ia += aStep, ib += bStep) {
        as[threadIdx.y][threadIdx.x] = A[ia + n * threadIdx.y + threadIdx.x];
        bs[threadIdx.y][threadIdx.x] = B[ib + n * threadIdx.y + threadIdx.x];
        __syncthreads();
        for (int k = 0; k < blockDim.x; k++) {
            sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    int index = n * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    res[index] = sum;
}

void multOMP(float* A, float* B, float* res, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++) {
                res[i * n + j] += A[i * n + k] * B[k * n + j];
            }
}

void clear(float* A, int n) {
    for (int i = 0; i < n * n; i++) A[i] = 0;
}

void fill(float* A, float* B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = 2.0f;
            B[i * n + j] = 1.0f;
        }
    }
}

int main() {
    int n = 1024;
    dim3 block_size(16, 16);
    dim3 num_block((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

    float* resGPU;
    float* AGPU, *BGPU;
    cudaMalloc(&resGPU, n * n * sizeof(float));
    cudaMalloc(&AGPU, n * n * sizeof(float));
    cudaMalloc(&BGPU, n * n * sizeof(float));

    float* A = new float[n * n];
    float* B = new float[n * n];
    float* res = new float[n * n];
    fill(A, B, n);
    clear(res, n);

    float max_error = 0.0f;

    omp_set_num_threads(4);
    double start = omp_get_wtime();
    multOMP(A, B, res, n);
    double finish = omp_get_wtime();
    std::cout << "OMP: " << finish - start << std::endl;
    for (int i = 0; i < n * n; ++i) {
        max_error = std::max(max_error, std::abs(res[i] - 2048.0f));
    }
    std::cout << "Error: " << max_error << std::endl;
    clear(res, n);

    cudaMemcpy(AGPU, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(BGPU, B, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(resGPU, res, n * n * sizeof(float), cudaMemcpyHostToDevice);
    start = omp_get_wtime();
    multGPU << <num_block, block_size >> > (AGPU, BGPU, resGPU, n);
    cudaDeviceSynchronize();
    finish = omp_get_wtime();
    std::cout << "GPU: " << finish - start << std::endl;
    cudaMemcpy(res, resGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n * n; ++i) {
        max_error = std::max(max_error, std::abs(res[i] - 2048.0f));
    }
    std::cout << "Error: " << max_error << std::endl;
    clear(res, n);

    start = omp_get_wtime();
    multOptimized << <num_block, block_size >> > (AGPU, BGPU, resGPU, n);
    cudaDeviceSynchronize();
    finish = omp_get_wtime();
    std::cout << "Optimized: " << finish - start << std::endl;
    cudaMemcpy(res, resGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n * n; ++i) {
        max_error = std::max(max_error, std::abs(res[i] - 2048.0f));
    }
    std::cout << "Error: " << max_error << std::endl;

    cudaFree(resGPU);
    cudaFree(AGPU);
    cudaFree(BGPU);
    delete[] A;
    delete[] B;
    delete[] res;

    return 1;
}