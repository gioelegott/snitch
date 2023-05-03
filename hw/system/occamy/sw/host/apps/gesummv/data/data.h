#ifndef GESUMMV_DATA_H_
#define GESUMMV_DATA_H_

// Statically define the data which will be used for the computation
// (this will be loaded into DRAM together with the binary)

#define N 1
#define double double

const double alpha = 9.51310901950086;
const double beta = 2.401644033326267;

double A[N*N] = {
     0.20091};


double B[N*N] = {
    -8.08693};


double x[N] = {
     0.13635};

double y[N];

uint32_t finished = 0;

#endif // GESUMMV_DATA_H_
