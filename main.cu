#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

__global__ void sieveKernel(bool* is_prime, unsigned long long limit, unsigned long long prime) {
    // Calculate thread ID and offset between IDs in thread block
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long offset = blockDim.x * gridDim.x;

    unsigned long long start = prime*prime;

    for (unsigned long long i = idx*; i < limit; i += offset) {
        is_prime[i] = false;
    }    
}