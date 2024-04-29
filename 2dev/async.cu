#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>

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

void h2d_mcpyA(float* d_data, float* h_data, u_int64_t N, cudaStream_t stream, int ite = 1);
void d2h_mcpyA(float* d_data, float* h_data, u_int64_t N, cudaStream_t stream, int ite = 1);
void d2d_mcpyA(float* d_from, float* d_to, u_int64_t N, cudaStream_t stream, int ite = 1);

void d2d_copy(float* d_from, float* d_to, u_int64_t N, cudaStream_t stream, int ite = 1);
__global__ void copy_gpu(float* d_from, float* d_to, u_int64_t N);

void printtime(cudaEvent_t start[], cudaEvent_t end[], u_int64_t N, int ite);
void enableP2P();

int main(int argc, char const* argv[]){

    u_int64_t N = 1 << 30; //1G
    // N = N * 4;
    u_int64_t N2 =  N; //1<<25;
    float* d_data;       //4B
    int iteration = 5;
    CHECK(cudaMalloc(&d_data, N * sizeof(float)));

    float* h_data = new float[N];
    for (u_int64_t i = 0; i < N; ++i)h_data[i] = rand();

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    std::cout << "H2D" << std::endl;
    for (uint64_t i = N2; i < N + 1; i <<= 1){
        h2d_mcpyA(d_data, h_data, i, stream, iteration);
        CHECK(cudaDeviceSynchronize());
    }

    std::cout << "D2H" << std::endl;
    for (uint64_t i = N2; i < N + 1; i <<= 1){
        h2d_mcpyA(d_data, h_data, i, stream, iteration);
        CHECK(cudaDeviceSynchronize());
    }

    CHECK(cudaSetDevice(1));
    float* d_data2;
    CHECK(cudaMalloc(&d_data2, N * sizeof(float)));

    CHECK(cudaSetDevice(0));
    std::cout << "D2D" << std::endl;
    for (uint64_t i = N2; i < N + 1; i <<= 1){
        d2d_mcpyA(d_data, d_data2, i, stream, iteration);
        CHECK(cudaDeviceSynchronize());
    }

    enableP2P();
    CHECK(cudaSetDevice(0));
    std::cout << "D2D P2P" << std::endl;
    for (uint64_t i = N2; i < N + 1; i <<= 1){
        d2d_mcpyA(d_data, d_data2, i, stream, iteration);
        CHECK(cudaDeviceSynchronize());
    }

    CHECK(cudaSetDevice(0));
    std::cout << "copy P2P" << std::endl;
    for (uint64_t i = N2; i < N + 1; i <<= 1){
        d2d_copy(d_data, d_data2, i, stream, iteration);
        CHECK(cudaDeviceSynchronize());
    }



    delete[] h_data;
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_data2));
    CHECK(cudaStreamDestroy(stream));

    return 0;
}

void h2d_mcpyA(float* d_data, float* h_data, u_int64_t N, cudaStream_t stream, int ite){
    cudaEvent_t start[ite], end[ite];
    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventCreate(&start[i]));
        CHECK(cudaEventCreate(&end[i]));
    }

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventRecord(start[i], stream));
        CHECK(cudaMemcpyAsync(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK(cudaEventRecord(end[i], stream));
    }
    CHECK(cudaDeviceSynchronize());

    printtime(start, end, N, ite);

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventDestroy(start[i]));
        CHECK(cudaEventDestroy(end[i]));
    }
}

void d2h_mcpyA(float* d_data, float* h_data, u_int64_t N, cudaStream_t stream, int ite){
    cudaEvent_t start[ite], end[ite];
    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventCreate(&start[i]));
        CHECK(cudaEventCreate(&end[i]));
    }

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventRecord(start[i], stream));
        CHECK(cudaMemcpyAsync(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaEventRecord(end[i], stream));
    }
    CHECK(cudaDeviceSynchronize());

    printtime(start, end, N, ite);
    
    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventDestroy(start[i]));
        CHECK(cudaEventDestroy(end[i]));
    }
}

void d2d_mcpyA(float* d_from, float* d_to, u_int64_t N, cudaStream_t stream, int ite){
    cudaEvent_t start[ite], end[ite];
    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventCreate(&start[i]));
        CHECK(cudaEventCreate(&end[i]));
    }

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventRecord(start[i], stream));
        CHECK(cudaMemcpyAsync(d_to, d_from, N * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CHECK(cudaEventRecord(end[i], stream));
    }
    CHECK(cudaDeviceSynchronize());

    printtime(start, end, N, ite);

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventDestroy(start[i]));
        CHECK(cudaEventDestroy(end[i]));
    }
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
        copy_gpu <<<grid, block, 0, stream >>> (d_from, d_to, N);
        CHECK(cudaEventRecord(end[i], stream));
    }
    CHECK(cudaDeviceSynchronize());

    printtime(start, end, N, ite);

    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventDestroy(start[i]));
        CHECK(cudaEventDestroy(end[i]));
    }
}

__global__ void copy_gpu(float* d_from, float* d_to, u_int64_t N){
    u_int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)d_to[idx] = d_from[idx];
}

void printtime(cudaEvent_t start[], cudaEvent_t end[], u_int64_t N, int ite){
    std::vector<float> t1(ite);
    for (u_int64_t i = 0; i < ite; i++){
        CHECK(cudaEventElapsedTime(&t1[i], start[i], end[i]));
    }
    float sum = std::accumulate(t1.begin() + 1, t1.end(), 0.0f);
    float mean_time = sum / (ite - 1);
    std::cout << " N: " << (N>>20) << " iter: " << ite << std::endl;
    std::cout << "\t1st Time: " << t1[0] << "ms" << std::endl;
    std::cout << "\tMean Time: " << mean_time << "ms" << std::endl;
    std::cout << "\tBandwidth: " << (N * sizeof(float) >> 20) / (mean_time / 1000) << "MB/s (" << ((N * sizeof(float) >> 20) / (mean_time / 1000)) / (1 << 10) << "GB/s)" << std::endl;
}

void enableP2P(){
    int ndev;
    CHECK(cudaGetDeviceCount(&ndev));
    for (int dev_id = 0; dev_id < ndev; dev_id++){//P2Pのデバイス間での有効化
        CHECK(cudaSetDevice(dev_id));
        for (int j = 0; j < ndev; j++){
            if (dev_id == j) continue;
            int peer_access_available = 0;
            CHECK(cudaDeviceCanAccessPeer(&peer_access_available, dev_id, j));

            if (peer_access_available){
                CHECK(cudaDeviceEnablePeerAccess(j, 0));
                printf("> GPU%d enabled direct access to GPU%d\n", dev_id, j);
            } else{
                printf("(%d, %d)\n", dev_id, j);
            }
        }
    }
}
