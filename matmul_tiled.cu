#include <cuda_runtime.h>
#include <stdio.h>
#include <string>

#define TILE_SIZE 16

// Dummy kernel to flush L2 cache
__global__ void flush_cache(float* buffer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buffer[idx] += 1.0f; // dummy access
    }
}

// Naive kernel
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled kernel
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void initialize_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

bool compare_matrix(const float* A, const float* B, int size) {
    float tol = 1e-3f;
    for (int i = 0; i < size; ++i)
        if (fabs(A[i] - B[i]) > tol)
            return false;
    return true;
}

void measure_and_run(int M, int N, int K) {
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    size_t dummy_size = 32 * 1024 * 1024;
    int dummy_len = dummy_size / sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C_naive = (float*)malloc(bytes_C);
    float *h_C_tiled = (float*)malloc(bytes_C);

    initialize_matrix(h_A, M * K);
    initialize_matrix(h_B, K * N);

    float *d_A, *d_B, *d_C, *d_dummy;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    cudaMalloc(&d_dummy, dummy_size);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    dim3 dummy_threads(256);
    dim3 dummy_blocks((dummy_len + 255) / 256);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < 10; ++i)
        matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Naive Matmul
    cudaMemset(d_C, 0, bytes_C);

    flush_cache<<<dummy_blocks, dummy_threads>>>(d_dummy, dummy_len);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i)
        matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);

    cudaMemcpy(h_C_naive, d_C, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemset(d_C, 0, bytes_C);

    // Warm-up
    for (int i = 0; i < 10; ++i)
        matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemset(d_C, 0, bytes_C);

    flush_cache<<<dummy_blocks, dummy_threads>>>(d_dummy, dummy_len);
    cudaDeviceSynchronize();

    // Tiled Matmul
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i)
        matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tiled;
    cudaEventElapsedTime(&ms_tiled, start, stop);

    cudaMemcpy(h_C_tiled, d_C, bytes_C, cudaMemcpyDeviceToHost);

    bool correct = compare_matrix(h_C_naive, h_C_tiled, M * N);

    printf("Naive kernel time: %.6f ms\n", ms_naive / 100.0f);
    printf("Tiled kernel time: %.6f ms\n", ms_tiled / 100.0f);
    printf("Result correctness: %s\n", correct ? "PASS" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_dummy);
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    printf("Running matmul for M, N, K: %d\n", M);
    measure_and_run(M, N, K);
    return 0;
}
