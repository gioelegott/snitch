
#include "snrt.h"
#include "data.h"

#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))

void gesummv(uint32_t n, uint32_t core_idx, uint32_t core_num, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE *x, DATA_TYPE *y)
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
    //         tmp1 += alpha * A[i][j] * x[j];
    //         tmp2 += beta * B[i][j] * x[j];
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
            tmp1 += alpha * A[i][j] * x[j];
            tmp2 += beta * B[i][j] * x[j];
        }
        y[i] = tmp1 + tmp2;
    }



    snrt_fpu_fence();


}


int main() {

    post_wakeup_cl();
    volatile int v [100] = {1, 2, 3};
    v[0] = 3;
    if(snrt_is_compute_core()) 
    {
        uint32_t core_idx = snrt_cluster_core_idx();
        uint32_t core_num = snrt_cluster_compute_core_num();

        uint32_t start_time_snitch = mcycle();

        gesummv(N, core_idx, core_num, alpha, beta, A, B, x, y);

        uint32_t end_time_snitch = mcycle();
    }

    return_to_cva6(SYNC_ALL);

    return 0;
}

