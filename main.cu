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

__global__ void getPrimeCount(bool* is_prime, unsigned long long limit, int* count) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long offset = blockDim.x * gridDim.x;

    for (unsigned long long i = idx; i < limit; i += offset) {
        if (is_prime[i]) atomicAdd(count, 1);
    }
}

__global__ void calculateMaxPrimeDifference(bool* is_prime, unsigned long long limit, unsigned long long* max_diff_global) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long offset = blockDim.x * gridDim.x;
    unsigned long long local_max_diff = 0;
    unsigned long long local_prev_prime = 0;
    bool local_first_prime_found = false;

    for (unsigned long long i = idx; i < limit; i += offset) {
        if (is_prime[i]) {
            if (local_first_prime_found) {
                unsigned long long check_diff = i - local_prev_prime;
                if (check_diff > local_max_diff) local_max_diff = check_diff;
            } else {
                local_first_prime_found = true;
            }
            local_prev_prime = i;
        }
    }

    if (local_max_diff > 0) atomicMax(max_diff_global, local_max_diff);
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
    int* h_count = (int*)malloc(sizeof(int));
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    getPrimeCount<<<10, 10>>>(d_is_prime, limit, d_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Number of primes: %d\n", *h_count);

    unsigned long long* h_max_diff = (unsigned long long*)malloc(sizeof(unsigned long long));
    unsigned long long* d_max_diff;
    CUDA_CHECK(cudaMalloc(&d_max_diff, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_max_diff, 0, sizeof(unsigned long long)));
    calculateMaxPrimeDifference<<<1, 1>>>(d_is_prime, limit, d_max_diff);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_max_diff, d_max_diff, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Maximum gap between primes: %d\n", *h_max_diff);
}