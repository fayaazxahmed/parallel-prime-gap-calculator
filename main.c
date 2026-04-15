#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "gmp.h"

int main(int argc, char* argv[]) {
    int my_rank;           // ID of current process (0, 1, 2, ...)
    int p;                 // number of processes
    int source;            // rank of sender (used by rank 0)
    int dest = 0;          // target for messages (worker -> Rank 0)
    int tag = 0;           // tag for messages

    unsigned long long global_limit; // 1 billion or higher
    unsigned long long range_size;   // how much processing power per process
    unsigned long long start_val;    // start searching
    unsigned long long end_val;      // stop searching

    // execution time
    double start_time, end_time;

    // GMP variables
    mpz_t current_prime, next_prime, gap;
    mpz_t local_max_gap, limit;
    mpz_t max_gap;
    mpz_t p1_res, p2_res;          
    mpz_t prev_prime;              // used for cross-boundary gap checking?

    // MPI transfer buffers
    unsigned long long local_gap;
    unsigned long long remote_gap;

    char local_p1[512];
    char local_p2[512];
    char remote_p1[512];
    char remote_p2[512];

    // Start MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc < 2) {
        if (my_rank == 0)
            printf("Usage: mpirun -np <p> ./program <limit>\n");

        MPI_Finalize();
        return 0;
    }

    global_limit = strtoull(argv[1], NULL, 10);
    start_time = MPI_Wtime();

    // Define ranges for each process
    // someone make better distributon ideas? 
    // (more processing closer to 1billion? randomized computing?)
    range_size = global_limit / p;
    start_val = my_rank * range_size;
    end_val = (my_rank == p - 1) ? global_limit : (my_rank + 1) * range_size;

    // Initialize GMP variables
    mpz_inits(current_prime, next_prime, gap, local_max_gap, limit,
              p1_res, p2_res, prev_prime, max_gap, NULL);

    mpz_set_ui(local_max_gap, 0);
    mpz_set_ui(limit, end_val);

    // Start at the first prime >= start_val
    mpz_set_ui(current_prime, (start_val < 2) ? 2 : start_val);
    
    // Boundary gap check (previous partition -> this one) 
    // Prevents missing prime gaps BETWEEN two ranges
    if (my_rank > 0) {
        mpz_set_ui(prev_prime, start_val);
        // Find the prime strictly less than start_val
        unsigned long long check = start_val - 1;
        while (check > 1) {
            mpz_set_ui(prev_prime, check); // Store it in prev_prime!
            if (mpz_probab_prime_p(prev_prime, 25)) break;
            check--;
        }
        // Now find the first prime >= start_val
        mpz_set_ui(current_prime, start_val - 1);
        mpz_nextprime(current_prime, current_prime);

        // Check the gap between the last prime of the previous range and the first of this one
        mpz_sub(gap, current_prime, prev_prime); 

        if(mpz_cmp(gap, local_max_gap) > 0){ 
            mpz_set(local_max_gap, gap); 
            mpz_set(p1_res, prev_prime); 
            mpz_set(p2_res, current_prime); 
        }
    } else {
        mpz_nextprime(current_prime, current_prime);
    }

    // Main prime search loop
    mpz_set(prev_prime, current_prime);
    while (1) {
        // find next consecutive prime after current_prime
        mpz_nextprime(next_prime, current_prime);
        
        // stop if next prime leaves this rank's range
        if (mpz_cmp_ui(next_prime, end_val) > 0) {
            break;
        }
        mpz_sub(gap, next_prime, current_prime); // gap=next-current

        // update local max gap + store the prime pair
        if (mpz_cmp(gap, local_max_gap) > 0) {
            mpz_set(local_max_gap, gap);
            mpz_set(p1_res, current_prime);
            mpz_set(p2_res, next_prime);
        }

        // advance
        mpz_set(current_prime, next_prime);
        mpz_set(prev_prime, current_prime);
    }

    gmp_snprintf(local_p1, sizeof(local_p1), "%Zd", p1_res);
    gmp_snprintf(local_p2, sizeof(local_p2), "%Zd", p2_res);
    local_gap = mpz_get_ui(local_max_gap);

    if (my_rank != 0) {
        // Workers send: max_gap, p1, and p2
        char buf_gap[512], buf_p1[512], buf_p2[512];
        gmp_snprintf(buf_gap, 512, "%Zd", local_max_gap);
        gmp_snprintf(buf_p1, 512, "%Zd", p1_res);
        gmp_snprintf(buf_p2, 512, "%Zd", p2_res);

        MPI_Send(buf_gap, 512, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Send(buf_p1, 512, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        MPI_Send(buf_p2, 512, MPI_CHAR, 0, 2, MPI_COMM_WORLD);
    } else {
        // Rank 0 starts with its own local maximum
        mpz_set(max_gap, local_max_gap);
        // p1_res and p2_res (already in local_p1/p2) hold Rank 0's best pair

        for (source = 1; source < p; source++) {
            char r_gap[512], r_p1[512], r_p2[512];
            MPI_Recv(r_gap, 512, MPI_CHAR, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(r_p1, 512, MPI_CHAR, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(r_p2, 512, MPI_CHAR, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            mpz_t remote_g;
            mpz_init_set_str(remote_g, r_gap, 10);

            if (mpz_cmp(remote_g, max_gap) > 0) {
                mpz_set(max_gap, remote_g);
                // Update buffers for final print
                strcpy(local_p1, r_p1);
                strcpy(local_p2, r_p2);
            }
            mpz_clear(remote_g);
        }

        end_time = MPI_Wtime();
        printf("\n--- Global Search Results ---\n");
        printf("Time Elapsed: %.4f seconds\n", end_time - start_time);
        gmp_printf("Search Limit: %llu\n", global_limit);
        gmp_printf("Largest Gap Found: %Zd\n", max_gap);
        printf("Occurred between: %s and %s\n", local_p1, local_p2);
    }

    // Cleanup
    mpz_clears(current_prime, next_prime, gap,
               local_max_gap, limit,
               p1_res, p2_res,
               prev_prime, max_gap, NULL);

    MPI_Finalize();

    return 0;
}