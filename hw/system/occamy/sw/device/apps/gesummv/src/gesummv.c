#include "snrt.h"
#include "gesummv.h"

#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))
#define double double
// Other variables
__thread volatile comm_buffer_t* comm_buffer_gesummv;


static inline void gesummv(int32_t n_rows, int32_t n_columns, double alpha, double beta, double* A, double* B, double *x, double *y)
{
    int32_t i;
    double tmp, tmp1, tmp2;


    // for (i = 0; i < n_rows; i++)
    // {
    //     tmp = 0;
    //     snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, n_columns, 8);
    //     snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, A + i*n_columns);
    //     snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, B + i*n_columns);
    //     snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D, x);

    //     snrt_ssr_enable();

    //     asm volatile
    //     ("frep.o %[n_frep], 3, 0, 0        \n"
    //      "fmul.d fa1, ft0, %[alpha]        \n"
    //      "fmadd.d fa1, ft1, %[beta], fa1   \n"
    //      "fmadd.d %[tmp], fa1,  ft2, %[tmp]\n"
    //      : [tmp] "+f"(tmp)
    //      : [n_frep] "r"(n_columns-1), [alpha] "f"(alpha), [beta] "f"(beta)
    //      : "ft0", "ft1", "ft2", "fa1"
    //     );

    //     snrt_fpu_fence();
    //     snrt_ssr_disable();
    //     y[i] = tmp; //maybe put y inside asm
    // }


    for (i = 0; i < n_rows; i++)
    {
        tmp1 = tmp2 = 0;
        snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, n_columns, 8);
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, A + i*n_columns);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, B + i*n_columns);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D, x);

        snrt_ssr_enable();

        asm volatile
        ("frep.o %[n_frep], 3, 0, 0          \n"
         "fmv.d fa1, ft2                     \n"
         "fmadd.d %[tmp1], ft0,  fa1, %[tmp1]\n"
         "fmadd.d %[tmp2], ft1,  fa1, %[tmp2]\n"
         : [tmp1] "+f"(tmp1), [tmp2] "+f"(tmp2)
         : [n_frep] "r"(n_columns-1)
         : "ft0", "ft1", "ft2", "fa1"
        );

        snrt_fpu_fence();
        snrt_ssr_disable();
        y[i] = alpha * tmp1 + beta * tmp2; //maybe put y inside asm
    }




    // for (i = 0; i < n_rows; i++)
    // {
    //     double tmp1 = 0;
    //     double tmp2 = 0;
    //     for (int j = 0; j < n_columns; j++)
    //     {
    //         tmp1 += A[i*n_columns + j] * x[j];
    //         tmp2 += B[i*n_columns + j] * x[j];
    //     }
    //     y[i] = alpha * tmp1+ beta* tmp2;//n_rows;

    // }
     snrt_fpu_fence();
}



void gesummv_job_dm_core(job_t* job) {

    // Get local job pointer as next free slot in l1 alloc
    gesummv_local_job_t* gesummv_job = (gesummv_local_job_t*)snrt_l1_next();

    mcycle(); //2|3

    // Copy job info
    snrt_dma_start_1d(gesummv_job, job, sizeof(gesummv_job_t));
    snrt_dma_wait_all();
    uint32_t n = gesummv_job->args.n;
    size_t mt_size = n * n * 8;
    size_t vt_size = n * 8;

    // Copy operand A
    double* A = (double*)((uint32_t)gesummv_job + sizeof(gesummv_local_job_t));
    snrt_dma_start_1d(A, (void*)(uint32_t)gesummv_job->args.A_l3_ptr, mt_size);

    // Copy operand B
    double* B = (double*)((uint32_t)A + mt_size);
    snrt_dma_start_1d(B, (void*)(uint32_t)gesummv_job->args.B_l3_ptr, mt_size);

    // Copy operand x
    double* x = (double*)((uint32_t)B + mt_size);
    snrt_dma_start_1d(x, (void*)(uint32_t)gesummv_job->args.x_l3_ptr, vt_size);


    // Set pointers to local job operands
    gesummv_job->args.A = A;
    gesummv_job->args.B = B;
    gesummv_job->args.x = x;
    gesummv_job->args.y = (double*)((uint32_t)x + vt_size);

    // Wait for DMA transfers to complete
    snrt_dma_wait_all();

    mcycle(); //3|4

    snrt_cluster_hw_barrier();

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




void gesummv_job_compute_core(job_t* job) {

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

    uint32_t c = CEIL(n, core_num);
    int32_t lb = c * core_idx;
    int32_t ub = MIN((c * (core_idx + 1)), n);

    
    mcycle();//4|5
    gesummv(ub - lb, n, alpha, beta, A + lb * n, B + lb * n, x, y + lb);
    mcycle();//5|6


    // Synchronize with DM core to make sure results are available
    // before DMA starts transfer to L3
    snrt_cluster_hw_barrier();

    mcycle();//6|7


}


static inline void run_job() {
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
    while(1)
    {
        mcycle(); //0|1
        post_wakeup_cl();

        // Execute job
        mcycle(); //1|2
        run_job();

        snrt_wfi();
    }
}


