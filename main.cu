#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void initSieve(bool* is_prime, unsigned long long limit) {
    // Calculate thread ID and offset between IDs in thread block
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long offset = blockDim.x * gridDim.x;

    // Mark 2 and all odd numbers as prime, everything else is initialized as not prime
    for (unsigned long long i = idx; i < limit; i += offset) {
        if (i < 2) {
            is_prime[i] = false;
        } else if (i == 2) {
            is_prime[i] = true;
        } else if (i%2 == 0) {
            is_prime[i] = false;
        } else {
            is_prime[i] = true;
        }
    }
}

__global__ void sieveKernel(bool* is_prime, unsigned long long limit, unsigned long long prime) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long offset = blockDim.x * gridDim.x;

    unsigned long long start = prime*prime;

    for (unsigned long long i = idx; i < limit; i += offset) {
        is_prime[i] = false;
    }    
}

int main() {
    unsigned long long limit = 100;
    unsigned long long limit_sqrt = sqrt(limit);
    bool* h_is_prime = (bool*)malloc((limit + 1)*sizeof(bool));
    bool *d_is_prime;
    CUDA_CHECK(cudaMalloc(&d_is_prime, (limit + 1) * sizeof(bool)));
    
    initSieve<<<10, 10>>>(d_is_prime, limit);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check initialized prime values (0 or 1)
    CUDA_CHECK(cudaMemcpy(h_is_prime, d_is_prime, (limit + 1) * sizeof(bool), cudaMemcpyDeviceToHost));
    for (int i = 0; i < limit + 1; i++) {
        printf("%d: %d\n", i, h_is_prime[i]);
    }
}