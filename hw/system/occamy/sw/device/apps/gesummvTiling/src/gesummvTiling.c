#include "snrt.h"
#include "gesummvTiling.h"

#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))
#define double double
#define MEM (1024 * 64) 
// Other variables
__thread volatile comm_buffer_t* comm_buffer_gesummv;


static inline void gesummvTiling(uint32_t n_rows, uint32_t n_columns, uint32_t core_idx, uint32_t core_num, double alpha, double beta, double* A, double* B, double *x, double *y)
{
    uint32_t i, j;
    double tmp1, tmp2;
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

    c = CEIL(n_rows, core_num);
    lb = c * core_idx;
    ub = MIN((c * (core_idx + 1)), n_rows);

    for (i = lb; i < ub; i++)
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
    size_t available_mem = MEM;//snrt_l1_end_addr() - tcdm_ptr;

    if (n > available_mem/4)
        return;


    uint32_t n_rows = ((available_mem/sizeof(double) - 2*n)/2)/n;
    n_rows = (n_rows > n) ? n/2 : n_rows/2;

    size_t mt_size = n * n_rows * sizeof(double);
    size_t vt_size = n * sizeof(double);
    size_t total = 2 * mt_size + 2 * vt_size;

    double* A = (double*)((uint32_t)gesummv_job + sizeof(gesummv_local_job_t));
    snrt_dma_start_1d(A, (void*)(uint32_t)gesummv_job->args.A_l3_ptr, mt_size);

    // Copy operand B
    double* B = (double*)((uint32_t)A + mt_size*2);
    snrt_dma_start_1d(B, (void*)(uint32_t)gesummv_job->args.B_l3_ptr, mt_size);

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

    for(int i = 1; i < n / n_rows; i++) 
    {
        mcycle(); //3|4
        snrt_dma_start_1d(A  + n_rows * n * (i%2), (void*)((uint32_t)gesummv_job->args.A_l3_ptr + i * mt_size), mt_size);
        snrt_dma_start_1d(B  + n_rows * n * (i%2), (void*)((uint32_t)gesummv_job->args.B_l3_ptr + i * mt_size), mt_size);

        snrt_dma_wait_all();

        mcycle(); //3|4

        snrt_cluster_hw_barrier(); 
        
    }
    if (n%n_rows)
    {
        mcycle(); //3|4
        snrt_dma_start_1d(A  + n_rows * n * ((n / n_rows)%2), (void*)((uint32_t)gesummv_job->args.A_l3_ptr + n / n_rows * mt_size), (n%n_rows) * n * sizeof(double));
        snrt_dma_start_1d(B  + n_rows * n * ((n / n_rows)%2), (void*)((uint32_t)gesummv_job->args.B_l3_ptr + n / n_rows * mt_size), (n%n_rows) * n * sizeof(double));

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

    // Synchronize with DM core to wait for operands
    // to be fully transferred in L1
    //snrt_cluster_hw_barrier();

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
    size_t available_mem = MEM;//(uint32_t)snrt_l1_end_addr() - (uint32_t)x;


    uint32_t n_rows = ((available_mem/sizeof(double) - 2*n)/2)/n;
    n_rows = (n_rows > n) ? n/2 : n_rows/2;


    uint32_t n_columns = n;

    mcycle();//4|5
    gesummvTiling(n_rows, n_columns, core_idx, core_num, alpha, beta, A, B, x, y);
    mcycle();//5|6
    snrt_cluster_hw_barrier();


    for(int i = 1; i < n / n_rows; i++)
    {

        mcycle();//4|5
        gesummvTiling(n_rows, n_columns, core_idx, core_num, alpha, beta, A + n_rows * n_columns * (i%2), B + n_rows * n_columns * (i%2), x, y + i* n_rows);
        mcycle();//5|6
        snrt_cluster_hw_barrier();


    }

    if (n%n_rows)
    {

        mcycle();//4|5
        gesummvTiling(n%n_rows, n_columns, core_idx, core_num, alpha, beta, A + n_rows * n_columns * ((n / n_rows)%2), B + n_rows * n_columns * ((n / n_rows)%2), x, y + n / n_rows * n_rows);
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
