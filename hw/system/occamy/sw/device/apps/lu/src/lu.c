
#include "snrt.h"
#include "lu.h"

#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))
#define DATA_TYPE double

__thread volatile comm_buffer_t* comm_buffer_lu;

void lu(uint32_t n, uint32_t core_idx, uint32_t core_num, DATA_TYPE *A)
{
    uint32_t i, j, k;
    DATA_TYPE tmp;
    /*gaussian reduction*/

    // for (i = 0; i < n-1; i++)
    // {
    //     for (j = i+1; j < n; j++)
    //     {
    //         tmp = A[j][i]/A[i][i];
    //         for (k = i; k < n; k++)
    //             A[j][k] -= A[i][k] * tmp;

    //         A[j][i] = tmp;
     //     }
    // }
 
    for (i = 0; i < n-1; i++)
    {
        for(j = i + 1 + core_idx; j < n; j += core_num)
        {
            tmp = A[j*n + i]/A[i*n + i];
            for (k = i; k < n; k++)
                A[j*n + k] -= A[i*n + k] * tmp;

            A[j*n + i] = tmp;           
        }
    }


    snrt_fpu_fence();
}


void lu_job_dm_core(job_t* job) {

    // Get local job pointer as next free slot in l1 alloc
    lu_local_job_t* lu_job = (lu_local_job_t*)snrt_l1_next();

    mcycle(); //2|3

    // Copy job info
    snrt_dma_start_1d(lu_job, job, sizeof(lu_job_t));
    snrt_dma_wait_all();
    uint32_t n = lu_job->args.n;
    size_t matrix_size = n * n * 8;

    // Copy operand A
    double* A = (double*)((uint32_t)lu_job + sizeof(lu_local_job_t));
    snrt_dma_start_1d(A, (void*)(uint32_t)lu_job->args.A_l3_ptr, matrix_size);

    // Set pointers to local job operands
    lu_job->args.A = A;
    // Wait for DMA transfers to complete
    snrt_dma_wait_all();

    mcycle(); //3|4

    snrt_cluster_hw_barrier();

    // Now we can update the L1 alloc pointer
    void* next = (void*)((uint32_t)(lu_job->args.A) + matrix_size);
    snrt_l1_update_next(next);

    mcycle(); //4|5
    // Synchronize cores to make sure results are available before
    // DMA starts transfer to L3
    snrt_cluster_hw_barrier();

    mcycle(); //5|6

    // Transfer data out
    snrt_dma_start_1d((void*)(uint32_t)lu_job->args.A_l3_ptr, lu_job->args.A, matrix_size);
    snrt_dma_wait_all();

    mcycle(); //6|7


}




void lu_job_compute_core(job_t* job) {

    // Synchronize with DM core to wait for operands
    // to be fully transferred in L1
    //snrt_cluster_hw_barrier();

    // Cast local job
    lu_local_job_t* lu_job = (lu_local_job_t*)job;

    // Get args
    lu_local_args_t args = lu_job->args;
    uint32_t n = args.n;
    double* A = args.A;

    // Run kernel
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t core_num = snrt_cluster_compute_core_num();
    mcycle();//4|5
    lu(n, core_idx, core_num, A);
    mcycle();//5|6


    // Synchronize with DM core to make sure results are available
    // before DMA starts transfer to L3
    snrt_cluster_hw_barrier();

    mcycle();//6|7


}



__attribute__((weak)) static inline void run_job() {
    // Force compiler to assign fallthrough path of the branch to
    // the DM core. This way the cache miss latency due to the branch
    // is incurred by the compute cores, and overlaps with the data
    // movement performed by the DM core.
    asm goto("bnez %0, %l[run_job_compute_core]"
             :
             : "r"(snrt_is_compute_core())
             :
             : run_job_compute_core);

    // Retrieve remote job data pointer
    job_t* job_remote = ((job_t*)comm_buffer_lu->usr_data_ptr);

    // Invoke job
    lu_job_dm_core(job_remote);

    // Synchronize clusters
    uint32_t cnt = snrt_sw_barrier_arrival();
    if (cnt == snrt_cluster_num()) {
        snrt_reset_barrier();
        snrt_int_sw_set(0);
    }
    mcycle(); //7|8
    goto run_job_end;

run_job_compute_core:;

    // Get pointer to local copy of job
    job_t* job_local = (job_t*)(snrt_l1_next());

    mcycle();//2|3

    // Synchronize with DM core such that it knows
    // it can update the l1 alloc pointer, and we know
    // job data is locally available
    snrt_cluster_hw_barrier();

   mcycle();//3|4

    // Invoke job
    lu_job_compute_core(job_local);

run_job_end:;
}


__attribute__((weak)) int main() {

    comm_buffer_lu = (volatile comm_buffer_t*)get_communication_buffer();

    // Notify CVA6 when snRuntime initialization is done
    post_wakeup_cl();
    return_to_cva6(SYNC_ALL);
    snrt_wfi();

    // Reset state after wakeup
    mcycle(); //0|1
    post_wakeup_cl();

    // Execute job
    mcycle(); //1|2
    run_job();

    return_to_cva6(SYNC_ALL);
}

