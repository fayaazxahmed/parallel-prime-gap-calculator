Parallel programming assignment in MPI C rewritten in CUDA.

## MPI C Implementation
+ Prime number search is parallelized across multiple CPU cores
+ The search range [0, n] is distributed equally across processes
+ Each range is searched sequentially to find prime numbers and calculate gaps
+ Gaps between numbers at the edges of the parallelized ranges are handled using MPI message passing between processes
+ Don't try to run this on your pc

## CUDA Implementation
+ Sieve of Eratosthenes algorithm is used to locate prime numbers from 2 to √*n* by marking their multiples as non-prime using a hybrid CPU-GPU architecture
+ Normal CPU loop iterates through primes, with multiple GPU threads used to mark the multiples of each prime
+ This version can now be run on your pc or on cloud instances using Google Colab or Jupyter  Notebooks

## Checklist (CUDA implementation)
✅ Marking primes in parallel

☐ Gap calculation still needs to be parallelized, currently runs on single thread
