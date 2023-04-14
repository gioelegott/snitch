#include "snrt.h"
#include "gesummv.h"

#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))
#define DATA_TYPE double
// Other variables
__thread volatile comm_buffer_t* comm_buffer_gesummv;

double* glob_ptr;

static inline void gesummv(uint32_t n, uint32_t core_idx, uint32_t core_num, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE *x, DATA_TYPE *y)
{
    uint32_t i, j;
    DATA_TYPE tmp1, tmp2;
    uint32_t lb;
    uint32_t ub;
    uint32_t c;

    //STRATEGY 1

    // for (i = core_idx; i < n; i+=core_num)
    // {
    //     tmp1 = tmp2 = 0;
    //     for (j = 0; j < n; j++)
    //     {
    //         tmp1 += alpha * A[i*n + j] * x[j];
    //         tmp2 += beta * B[i*n + j] * x[j];
    //     }
    //     y[i] = tmp1 + tmp2;
    // }

    //STRATEGY 2

    c = CEIL(n, core_num);
    lb = c * core_idx;
    ub = MIN((c * (core_idx + 1)), n);

    for (i = lb; i < ub; i++)
    {
        tmp1 = tmp2 = 0;
        for (j = 0; j < n; j++)
        {
            tmp1 += alpha * A[i*n + j] * x[j];
            tmp2 += beta * B[i*n + j] * x[j];
        }
        y[i] = tmp1 + tmp2;
    }

    snrt_fpu_fence();

}



void gesummv_job_dm_core(job_t* job) {

    // Get local job pointer as next free slot in l1 alloc
    gesummv_local_job_t* gesummv_job = (gesummv_local_job_t*)snrt_l1_next();

    mcycle();

    // Copy job info
    snrt_dma_start_1d(gesummv_job, job, sizeof(gesummv_job_t));

    // Wait for job info transfer to complete
    snrt_dma_wait_all();

    mcycle();

    // Synchronize with compute cores before updating the l1 alloc pointer
    // such that they can retrieve the local job pointer.
    // Also ensures compute cores see the transferred job information.
    snrt_cluster_hw_barrier();

    uint32_t n = gesummv_job->args.n;
    size_t matrix_size = n * n * 8;
    size_t vector_size = n * 8;

    // Copy operand A
    double* A = (double*)((uint32_t)gesummv_job + sizeof(gesummv_local_job_t));
    snrt_dma_start_1d(A, (void*)(uint32_t)gesummv_job->args.A_l3_ptr, matrix_size);
    glob_ptr = A;

    // Copy operand B
    double* B = (double*)((uint32_t)A + matrix_size);
    snrt_dma_start_1d(B, (void*)(uint32_t)gesummv_job->args.B_l3_ptr, matrix_size);

    // Copy operand x
    double* x = (double*)((uint32_t)B + matrix_size);
    snrt_dma_start_1d(x, (void*)(uint32_t)gesummv_job->args.x_l3_ptr, vector_size);


    // Set pointers to local job operands
    gesummv_job->args.A = A;
    gesummv_job->args.B = B;
    gesummv_job->args.x = x;
    gesummv_job->args.y = (double*)((uint32_t)x + vector_size);

    // Now we can update the L1 alloc pointer
    void* next = (void*)((uint32_t)(gesummv_job->args.y) + vector_size);
    snrt_l1_update_next(next);

    // Wait for DMA transfers to complete
    snrt_dma_wait_all();

    mcycle();

    // Synchronize with compute cores to make sure the data
    // is available before they can start computing on it
    snrt_cluster_hw_barrier();

    mcycle();

    // Synchronize cores to make sure results are available before
    // DMA starts transfer to L3
    snrt_cluster_hw_barrier();

    mcycle();

    // Transfer data out
    snrt_dma_start_1d((void*)(uint32_t)gesummv_job->args.y_l3_ptr, gesummv_job->args.y, vector_size);
    snrt_dma_wait_all();

    mcycle();



}




void gesummv_job_compute_core(job_t* job) {

    // Synchronize with DM core to wait for operands
    // to be fully transferred in L1
    snrt_cluster_hw_barrier();

    // Cast local job
    gesummv_local_job_t* gesummv_job = (gesummv_local_job_t*)job;

    // Get args
    gesummv_local_args_t args = gesummv_job->args;
    uint32_t n = args.n;
    double alpha = args.alpha;
    double beta = args.beta;
    double* A = args.A;
    double* B = args.B;
    double* x = args.x;
    double* y = args.y;

    mcycle();

    mcycle();

    // Run kernel
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t core_num = snrt_cluster_compute_core_num();
    gesummv(n, core_idx, core_num, alpha, beta, A, B, x, y);


    mcycle();

    // Synchronize with DM core to make sure results are available
    // before DMA starts transfer to L3
    snrt_cluster_hw_barrier();

    mcycle();
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
    job_t* job_remote = ((job_t*)comm_buffer_gesummv->usr_data_ptr);

    // Invoke job
    gesummv_job_dm_core(job_remote);

    // Synchronize clusters
    uint32_t cnt = snrt_sw_barrier_arrival();
    if (cnt == snrt_cluster_num()) {
        snrt_reset_barrier();
        snrt_int_sw_set(0);
    }

    goto run_job_end;

run_job_compute_core:;

    // Get pointer to local copy of job
    job_t* job_local = (job_t*)(snrt_l1_next());

    mcycle();

    // Synchronize with DM core such that it knows
    // it can update the l1 alloc pointer, and we know
    // job information is locally available
    snrt_cluster_hw_barrier();

   mcycle();

    // Invoke job
    gesummv_job_compute_core(job_local);

run_job_end:;
}





__attribute__((weak)) int main() {

    comm_buffer_gesummv = (volatile comm_buffer_t*)get_communication_buffer();

    // Notify CVA6 when snRuntime initialization is done
    post_wakeup_cl();
    return_to_cva6(SYNC_ALL);
    snrt_wfi();

    // Reset state after wakeup
    mcycle();
    post_wakeup_cl();

    // Execute job
    mcycle();
    run_job();

    // Go to sleep until next job
    mcycle();
    return_to_cva6(SYNC_ALL);
}

