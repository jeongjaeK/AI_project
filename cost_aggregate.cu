#include <cuda_runtime.h>
#include <stdio.h>

#define N 16
#define H 512
#define W 512
#define C 32

__global__ void guided_aggregation_kernel(const float* features, const float* weights, float* output) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= H || w >= W) return;

    __shared__ float s_weight[N];
    __shared__ float s_feature[N][C];

    if (threadIdx.y == 0 && threadIdx.x < N) {
        int n = threadIdx.x;

        s_weight[n] = weights[n * H * W + h * W + w];

        for (int c = 0; c < C; ++c) {
            s_feature[n][c] = features[n * H * W * C + h * W * C + w * C + c];
        }
    }

    __syncthreads();

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float num = 0.0f;
        float denom = 0.0f;

        for (int n = 0; n < N; ++n) {
            float w = s_weight[n];
            float f = s_feature[n][c];
            num += w * f;
            denom += w;
        }

        if (denom > 1e-6f)
            output[h * W * C + w * C + c] = num / denom;
        else
            output[h * W * C + w * C + c] = 0.0f;
    }
}

void launch_guided_aggregation(const float* features, const float* weights, float* output) {
    dim3 blockDim(32, 1);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y);

    guided_aggregation_kernel<<<gridDim, blockDim>>>(features, weights, output);
}

