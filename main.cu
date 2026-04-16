#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

__global__ void sieveKernel(uint8_t* bit_array, uint64_t limit, uint64_t max_num, bool* prime) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < max) {
        uint64_t p = primes[tid];
        // Assume all numbers are prime initially
        for (uint64_t i = p * p; i <= limit; i += p) {
            prime[i] = true; 
        }
    }
}