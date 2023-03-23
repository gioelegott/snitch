
#include "snrt.h"
#include "data.h"

#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))

void lu(uint32_t n, uint32_t core_idx, uint32_t core_num, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE *x, DATA_TYPE *y)
{


    snrt_fpu_fence();


}


int main() {

    post_wakeup_cl();

    if(snrt_is_compute_core()) 
    {
        uint32_t core_idx = snrt_cluster_core_idx();
        uint32_t core_num = snrt_cluster_compute_core_num();

        uint32_t start_time_snitch = mcycle();

        lu(N, core_idx, core_num, alpha, beta, A, B, x, y);

        uint32_t end_time_snitch = mcycle();
    }

    return_to_cva6(SYNC_ALL);

    return 0;
}

