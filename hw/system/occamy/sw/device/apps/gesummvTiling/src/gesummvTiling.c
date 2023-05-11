#include "snrt.h"
#include "gesummvTiling.h"

#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))
#define double double
#define MEM 67584 
// Other variables
__thread volatile comm_buffer_t* comm_buffer_gesummv;


static inline __attribute__((always_inline)) void gesummvTiling(int32_t n_rows, int32_t n_columns, double alpha, double beta, double* A, double* B, double *x, double *y)
{
    int32_t i, j;
    double tmp1, tmp2;

    for (i = 0; i < n_rows; i++)
    {
        tmp1 = tmp2 = 0;
        for (j = 0; j < n_columns; j++)
        {
            tmp1 += alpha * A[i*n_columns + j] * x[j];
            tmp2 += beta * B[i*n_columns + j] * x[j];
        }
        y[i] = tmp1 + tmp2;//n_rows;

    }

    snrt_fpu_fence();
}



void gesummvTiling_job_dm_core(job_t* job) {

    // Get local job pointer as next free slot in l1 alloc
    gesummv_local_job_t* gesummv_job = (gesummv_local_job_t*)snrt_l1_next();

    mcycle(); //2|3

    // Copy job info
    snrt_dma_start_1d(gesummv_job, job, sizeof(gesummv_job_t));
    snrt_dma_wait_all();
    uint32_t n = gesummv_job->args.n;


    uint32_t tcdm_ptr = (uint32_t)gesummv_job + sizeof(gesummv_local_job_t);

    uint32_t rows_per_batch = (MEM/(16*n) - 1);
    rows_per_batch = (rows_per_batch > n) ? n/2 : rows_per_batch/2;

    size_t mt_size = n * rows_per_batch * 8;

    uint32_t A_l3_ptr = (uint32_t)gesummv_job->args.A_l3_ptr;
    double* A = (double*)((uint32_t)gesummv_job + sizeof(gesummv_local_job_t));
    snrt_dma_start_1d(A, (void*)A_l3_ptr, mt_size);

    uint32_t B_l3_ptr = (uint32_t)gesummv_job->args.B_l3_ptr;
    double* B = (double*)((uint32_t)A + mt_size*2);
    snrt_dma_start_1d(B, (void*)B_l3_ptr, mt_size);

    size_t vt_size = n * 8;

    // Copy operand x
    double* x = (double*)((uint32_t)B + mt_size*2);
    snrt_dma_start_1d(x, (void*)(uint32_t)gesummv_job->args.x_l3_ptr, mt_size);


    // Set pointers to local job operands
    gesummv_job->args.A = A;
    gesummv_job->args.B = B;
    gesummv_job->args.x = x;
    gesummv_job->args.y = (double*)((uint32_t)x + vt_size);

    // Wait for DMA transfers to complete
    snrt_dma_wait_all();

    mcycle(); //3|4

    snrt_cluster_hw_barrier();



    double* A_ptr[2] = {A, A + rows_per_batch * n};
    double* B_ptr[2] = {B, B + rows_per_batch * n};

    uint32_t i;
    for(i = 1; i < n / rows_per_batch; i++) 
    {
        mcycle(); //3|4
        snrt_dma_start_1d(A_ptr[i%2], (void*)A_l3_ptr + i * mt_size, mt_size);
        snrt_dma_start_1d(B_ptr[i%2], (void*)B_l3_ptr + i * mt_size, mt_size);

        snrt_dma_wait_all();

        mcycle(); //3|4

        snrt_cluster_hw_barrier(); 
        
    }
    uint32_t rem = n%rows_per_batch;
    if (rem)
    {
        mcycle(); //3|4
        snrt_dma_start_1d(A_ptr[i%2], (void*)A_l3_ptr + i * mt_size, rem * n * 8);
        snrt_dma_start_1d(B_ptr[i%2], (void*)B_l3_ptr + i * mt_size, rem * n * 8);
        snrt_dma_wait_all();

        mcycle(); //3|4

        snrt_cluster_hw_barrier(); 

    }

    // Now we can update the L1 alloc pointer
    void* next = (void*)((uint32_t)(gesummv_job->args.y) + vt_size);
    snrt_l1_update_next(next);

    mcycle(); //4|5
    // Synchronize cores to make sure results are available before
    // DMA starts transfer to L3
    snrt_cluster_hw_barrier();

    mcycle(); //5|6

    // Transfer data out
    snrt_dma_start_1d((void*)(uint32_t)gesummv_job->args.y_l3_ptr, gesummv_job->args.y, vt_size);
    snrt_dma_wait_all();

    mcycle(); //6|7


}




void gesummvTiling_job_compute_core(job_t* job) {

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

    // Run kernel
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t core_num = snrt_cluster_compute_core_num();

    uint32_t rows_per_batch = (MEM/(16*n) - 1);
    rows_per_batch = (rows_per_batch > n) ? n/2 : rows_per_batch/2;

    uint32_t c = CEIL(rows_per_batch, core_num);
    int32_t lb = c * core_idx;
    int32_t ub = MIN((c * (core_idx + 1)), rows_per_batch);

    double* A_ptr[2] = {A + lb * n, A + lb * n + rows_per_batch * n};
    double* B_ptr[2] = {B + lb * n, B + lb * n + rows_per_batch * n};
    y = y + lb;


    mcycle();//4|5
    gesummvTiling(ub - lb, n, alpha, beta, A_ptr[0], B_ptr[0], x, y);
    mcycle();//5|6
    snrt_cluster_hw_barrier();


    uint32_t i;
    for(i = 1; i < n / rows_per_batch; i++)
    {

        mcycle();//4|5
        gesummvTiling(ub - lb, n, alpha, beta, A_ptr[i%2], B_ptr[i%2], x, y + i* rows_per_batch);
        mcycle();//5|6
        snrt_cluster_hw_barrier();
    }
    
    uint32_t rem = n%rows_per_batch;
    if (rem)
    {
        c = CEIL(rem, core_num);
        lb = c * core_idx;
        ub = MIN((c * (core_idx + 1)), rem);

        mcycle();//4|5
        gesummvTiling(ub - lb, n, alpha, beta, A_ptr[i%2], B_ptr[i%2], x, y + i * rows_per_batch);
        mcycle();//5|6
        snrt_cluster_hw_barrier();
    }
    mcycle();//4|5


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
    gesummvTiling_job_dm_core(job_remote);

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
    gesummvTiling_job_compute_core(job_local);

run_job_end:;
}





__attribute__((weak)) int main() {

    comm_buffer_gesummv = (volatile comm_buffer_t*)get_communication_buffer();

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

