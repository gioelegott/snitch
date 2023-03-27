
#include "snrt.h"
#include "data.h"

#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))

void lu(uint32_t n, uint32_t core_idx, uint32_t core_num, DATA_TYPE A[N][N])
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
        for(j = i + 1 + core_idx; j < n; j=+ core_num)
        {
            tmp = A[j][i]/A[i][i];
            for (k = i; k < n; k++)
                A[j][k] -= A[i][k] * tmp;

            A[j][i] = tmp;           
        }
    }


    snrt_fpu_fence();
}


int main() {

    post_wakeup_cl();

    if(snrt_is_compute_core()) 
    {
        uint32_t core_idx = snrt_cluster_core_idx();
        uint32_t core_num = snrt_cluster_compute_core_num();

        //barrier
        uint32_t start_time_snitch = mcycle();


        lu(N, core_idx, core_num, A);

        uint32_t end_time_snitch = mcycle();

        //barrier
    }
    else
    {
        //allocate space on memeory snrtL1alloc

        //dma transfer

        //barrier
        //barrier

        //dma transfer
    }
    return_to_cva6(SYNC_ALL);

}

