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

    for (unsigned long long i = start + idx*prime; i < limit; i += offset*prime) {
        is_prime[i] = false;
    }    
}

int main() {
    unsigned long long limit = 1000000000;
    unsigned long long limit_sqrt = sqrt(limit);
    bool* h_is_prime = (bool*)malloc((limit + 1)*sizeof(bool));
    bool *d_is_prime;
    CUDA_CHECK(cudaMalloc(&d_is_prime, (limit + 1) * sizeof(bool)));

    initSieve<<<10, 10>>>(d_is_prime, limit);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    for (int i = 2; i < limit_sqrt; i++) {
        bool i_is_prime;
        CUDA_CHECK(cudaMemcpy(&i_is_prime, &d_is_prime[i], sizeof(bool), cudaMemcpyDeviceToHost));
        if (i_is_prime) {
            sieveKernel<<<10, 10>>>(d_is_prime, limit, i);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Check initialized prime values (0 or 1) and return total count
    CUDA_CHECK(cudaMemcpy(h_is_prime, d_is_prime, (limit + 1) * sizeof(bool), cudaMemcpyDeviceToHost));
    int count = 0;
    for (int i = 0; i < limit + 1; i++) {
        if (h_is_prime[i] == 1) count++;
    }
    printf("Number of primes: %d\n", count);
}