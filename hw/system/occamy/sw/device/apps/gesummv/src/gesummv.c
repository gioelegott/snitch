
#include "snrt.h"
#include "data.h"


void gesummv(uint32_t n, uint32_t core_idx, uint32_t core_num, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE *x, DATA_TYPE *y)
{
    uint32_t i, j;
    DATA_TYPE tmp1, tmp2;
    DATA_TYPE y1[N], y2[N];
    DATA_TYPE correct, wrong1, wrong2;
    correct = wrong1 = wrong2 = 0;
    uint32_t portion;
    uint32_t start;

    //STRATEGY 1
    if (core_idx <= n%core_num)
    {
        portion = n/core_num + 1;
        start = core_idx * portion;
    }
    else
    {
        portion = n/core_num;
        start = n%core_num * (portion + 1) + (core_idx - n%core_num)*portion;
    }

    for (i = start; i < portion; i++)
    {
        tmp1 = tmp2 = 0;
        for (j = 0; j < n; j++)
        {
            tmp1 += alpha * A[i][j] * x[j];
            tmp2 += beta * B[i][j] * x[j];
        }
        y1[i] = tmp1 + tmp2;
    }



    //STRATEGY 2
    // if (!n%core_num)
    // {
    //     portion = n/core_num;
    //     start = core_idx;
    // }
    // else
    // {
    //     portion = n/core_num + 1;
    //     if (core_idx * portion < n)
    //     {
    //         start = core_idx * portion;
    //         if((core_idx + 1) * portion >= n)
    //             portion = n - core_idx*portion;
    //     }
    //     else
    //     {
    //         portion = 0;
    //         start = 0;
    //     }
    // }

    // for (i = start; i < portion; i++)
    // {
    //     tmp1 = tmp2 = 0;
    //     for (j = 0; j < n; j++)
    //     {
    //         tmp1 += alpha * A[i][j] * x[j];
    //         tmp2 += beta * B[i][j] * x[j];
    //     }
    //     y2[i] = tmp1 + tmp2;
    // }


    //STRATEGY 3

    for (i = core_idx; i < n; i+=core_num)
    {
        tmp1 = tmp2 = 0;
        for (j = 0; j < n; j++)
        {
            tmp1 += alpha * A[i][j] * x[j];
            tmp2 += beta * B[i][j] * x[j];
        }
        y2[i] = tmp1 + tmp2;
    }

    //barrier
    snrt_fpu_fence();

    if (core_idx == 1)
    {
        for (i = 0; i < n; i++)
            y[i] = (y1[i] == y2[i]) ? (DATA_TYPE)12345 : (DATA_TYPE)54321;
    }
        snrt_cluster_hw_barrier();


}


int main() {

    post_wakeup_cl();

    if(snrt_is_dm_core())
    {
        snrt_fpu_fence();
        snrt_cluster_hw_barrier();
        return 0;
    }

    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t core_num = snrt_cluster_compute_core_num();

    uint32_t start_time_snitch = mcycle();

    gesummv(N, core_idx, core_num, alpha, beta, A, B, x, y);

    uint32_t end_time_snitch = mcycle();

    return_to_cva6(SYNC_ALL);

    return 0;
}

