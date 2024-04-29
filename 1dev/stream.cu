#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "\tError: %s:%d, ", __FILE__, __LINE__);               \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

void d2d_copy(float* d_from, float* d_to, u_int64_t N, cudaStream_t stream, int ite = 1);
void d2d_scale(float* d_from, float* d_to, u_int64_t N, cudaStream_t stream, int ite = 1);
void d2d_sum(float* d_from1, float* d_from2, float* d_to, u_int64_t N, cudaStream_t stream, int ite = 1);
void d2d_triad(float* d_from1, float* d_from2, float* d_to, u_int64_t N, cudaStream_t stream, int ite = 1);

__global__ void copy_gpu(float* d_from, float* d_to, u_int64_t N){
    u_int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)d_to[idx] = d_from[idx];
}

__global__ void scale_gpu(float* d_from, float* d_to, u_int64_t N){
    u_int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)d_to[idx] = 9 * d_from[idx];
}

__global__ void sum_gpu(float* d_from1, float* d_from2, float* d_to, u_int64_t N){
    u_int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)d_to[idx] = d_from1[idx] + d_from2[idx];
}

__global__ void triad_gpu(float* d_from1, float* d_from2, float* d_to, u_int64_t N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)d_to[idx] = d_from1[idx] + 9 * d_from2[idx];
}

void printtime(cudaEvent_t start[], cudaEvent_t end[], u_int64_t N, int ite);
void enableP2P();

int main(int argc, char const* argv[]){

    u_int64_t N = 1 << 30; //1G
    // N = N * 4;
    u_int64_t N2 = N; //1<<25;
    float* d_dataA, * d_dataB, * d_dataC, * d_data1D;       //4B
    int iteration = 5;
    CHECK(cudaMalloc(&d_dataA, N * sizeof(float)));
    CHECK(cudaMalloc(&d_dataB, N * sizeof(float)));
    CHECK(cudaMalloc(&d_dataC, N * sizeof(float)));


    float* h_data = new float[N];
    for (u_int64_t i = 0; i < N; ++i)h_data[i] = rand();
    std::vector<std::chrono::high_resolution_clock::time_point> cpytime(5);
    cpytime[0] = std::chrono::high_resolution_clock::now();
    CHECK(cudaMemcpy(d_dataA, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    cpytime[1] = std::chrono::high_resolution_clock::now();
    CHECK(cudaMemcpy(d_dataB, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    cpytime[2] = std::chrono::high_resolution_clock::now();
    CHECK(cudaMemcpy(d_dataC, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    cpytime[3] = std::chrono::high_resolution_clock::now();

    std::vector<float> cpyt1(4);
    for (u_int64_t i = 0; i < 4; i++){
        cpyt1[i] = std::chrono::duration<float, std::milli>(cpytime[i+1] - cpytime[i]).count();
        std::cout << "\t" << i << "th Time: " << cpyt1[i] << "ms" << std::endl;
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));


    std::cout << "copy" << std::endl;
    for (uint64_t i = N2; i < N + 1; i <<= 1){
        d2d_copy(d_dataC, d_dataA, i, stream, iteration);
        CHECK(cudaDeviceSynchronize());
    }

    std::cout << "scale" << std::endl;
    for (uint64_t i = N2; i < N + 1; i <<= 1){
        d2d_scale(d_dataC, d_dataA, i, stream, iteration);
        CHECK(cudaDeviceSynchronize());
    }

    std::cout << "sum" << std::endl;
    for (uint64_t i = N2; i < N + 1; i <<= 1){
        d2d_sum(d_dataC, d_data1D, d_dataA, i, stream, iteration);
        CHECK(cudaDeviceSynchronize());
    }

    std::cout << "triad" << std::endl;
    for (uint64_t i = N2; i < N + 1; i <<= 1){
        d2d_triad(d_dataC, d_data1D, d_dataA, i, stream, iteration);
        CHECK(cudaDeviceSynchronize());
    }

    delete[] h_data;
    CHECK(cudaFree(d_dataA));CHECK(cudaFree(d_dataC));CHECK(cudaFree(d_data1D));
    CHECK(cudaStreamDestroy(stream));

    return 0;
}

void d2d_copy(float* d_from, float* d_to, u_int64_t N, cudaStream_t stream, int ite){
    cudaEvent_t start[ite], end[ite];
    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventCreate(&start[i]));
        CHECK(cudaEventCreate(&end[i]));
    }

    dim3 block(1024, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, 1, 1);

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventRecord(start[i], stream));
        copy_gpu <<<grid, block, 0, stream>>> (d_from, d_to, N);
        CHECK(cudaEventRecord(end[i], stream));
    }
    CHECK(cudaDeviceSynchronize());

    printtime(start, end, N, ite);

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventDestroy(start[i]));
        CHECK(cudaEventDestroy(end[i]));
    }
}

void d2d_scale(float* d_from, float* d_to, u_int64_t N, cudaStream_t stream, int ite){
    cudaEvent_t start[ite], end[ite];
    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventCreate(&start[i]));
        CHECK(cudaEventCreate(&end[i]));
    }

    dim3 block(1024, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, 1, 1);

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventRecord(start[i], stream));
        scale_gpu <<<grid, block, 0, stream>>> (d_from, d_to, N);
        CHECK(cudaEventRecord(end[i], stream));
    }
    CHECK(cudaDeviceSynchronize());

    printtime(start, end, N, ite);

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventDestroy(start[i]));
        CHECK(cudaEventDestroy(end[i]));
    }
}

void d2d_sum(float* d_from1,float* d_from2, float* d_to, u_int64_t N, cudaStream_t stream, int ite){
    cudaEvent_t start[ite], end[ite];
    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventCreate(&start[i]));
        CHECK(cudaEventCreate(&end[i]));
    }

    dim3 block(1024, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, 1, 1);

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventRecord(start[i], stream));
        sum_gpu <<<grid, block, 0, stream>>> (d_from1, d_from2, d_to, N);
        CHECK(cudaEventRecord(end[i], stream));
    }
    CHECK(cudaDeviceSynchronize());

    printtime(start, end, N, ite);

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventDestroy(start[i]));
        CHECK(cudaEventDestroy(end[i]));
    }
}

void d2d_triad(float* d_from1, float* d_from2, float* d_to, u_int64_t N, cudaStream_t stream, int ite){
    cudaEvent_t start[ite], end[ite];
    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventCreate(&start[i]));
        CHECK(cudaEventCreate(&end[i]));
    }

    dim3 block(1024, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, 1, 1);

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventRecord(start[i], stream));
        triad_gpu <<<grid, block, 0, stream>>> (d_from1, d_from2, d_to, N);
        CHECK(cudaEventRecord(end[i], stream));
    }
    CHECK(cudaDeviceSynchronize());

    printtime(start, end, N, ite);

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventDestroy(start[i]));
        CHECK(cudaEventDestroy(end[i]));
    }
}

void printtime(cudaEvent_t start[], cudaEvent_t end[], u_int64_t N, int ite){
    std::vector<float> t1(ite);
    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventElapsedTime(&t1[i], start[i], end[i]));
    }
    float sum = std::accumulate(t1.begin() + 1, t1.end(), 0.0f);
    float mean_time = sum / (ite - 1);
    std::cout << " N: " << (N>>20) << "M iter: " << ite << std::endl;
    std::cout << "\t1st Time: " << t1[0] << "ms" << std::endl;
    std::cout << "\tMean Time: " << mean_time << "ms" << std::endl;
    std::cout << "\tBandwidth: " << (N * sizeof(float) >> 20) / (mean_time / 1000) << "MB/s (" << ((N * sizeof(float) >> 20) / (mean_time / 1000)) / (1 << 10) << "GB/s)" << std::endl;
}
