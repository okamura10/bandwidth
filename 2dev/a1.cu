#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

void h2d_memcpy(float* d_data, float* h_data, u_int64_t N, cudaStream_t stream,int ite = 1);
void enableP2P();

int main(){

    u_int64_t N = 1 << 30; //1G
    float* d_data;       //4B
    CHECK(cudaMalloc(&d_data, N * sizeof(float)));

    float* h_data = new float[N];
    for (int i = 0; i < N; ++i)h_data[i] = rand();

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    h2d_memcpy(d_data, h_data, N, stream,2);

    CHECK(cudaDeviceSynchronize());



    
    delete[] h_data;
    CHECK(cudaFree(d_data));


    return 0;
}

void h2d_memcpy(float* d_data, float* h_data, u_int64_t N, cudaStream_t stream,int ite){
    int tNum = 10;
    cudaEvent_t time[ite*tNum];
    for (int i = 0; i < ite*tNum; i++) CHECK(cudaEventCreate(&time[i]));
    float* h_result = new float[N];

    for (size_t k = 0; k < ite; k++){
        //h2d
        CHECK(cudaEventRecord(time[k*ite+0], stream));
        CHECK(cudaMemcpyAsync(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK(cudaEventRecord(time[k*ite+1], stream));
        CHECK(cudaEventSynchronize(time[k*ite+1]));

        // ちゃんと転送されていることを確認するための操作
        thrust::device_ptr<float> d_vec(d_data);
        thrust::sort(thrust::cuda::par.on(stream), d_vec, d_vec + N);
        CHECK(cudaEventRecord(time[k*ite+2], stream));
        CHECK(cudaEventSynchronize(time[k*ite+2]));

        //d2h
        CHECK(cudaEventRecord(time[k*ite+3], stream));
        CHECK(cudaMemcpyAsync(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaEventRecord(time[k*ite+4], stream));
        CHECK(cudaEventSynchronize(time[k*ite+4]));

        std::cout << "Sorted :";
        for (u_int64_t i = 0; i < 10; i++){
            std::cout << h_result[i] << " ";
        }
        std::cout << std::endl;

    }
    
    for (size_t k = 0; k < ite; k++){
        float mstime = 0;
        CHECK(cudaEventElapsedTime(&mstime, time[k*ite+0], time[k*ite+1]));
        std::cout << "time: " << mstime << " ms" << std::endl;
        CHECK(cudaEventElapsedTime(&mstime, time[k*ite+3], time[k*ite+4]));
        std::cout << "time: " << mstime << " ms" << std::endl;
    }
    
    
    CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < tNum; i++) CHECK(cudaEventDestroy(time[i]));
    CHECK(cudaDeviceSynchronize());
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