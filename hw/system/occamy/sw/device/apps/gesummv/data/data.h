#ifndef GESUMMV_DATA_H_
#define GESUMMV_DATA_H_

// Statically define the data which will be used for the computation
// (this will be loaded into DRAM together with the binary)

#define N 2
#define DATA_TYPE double

DATA_TYPE alpha = -9.551078094476653;
DATA_TYPE beta = 9.355568001741752;

DATA_TYPE A[N][N] = {
    {-6.41170, -6.09353},
    { 1.96539,  8.50262}};


DATA_TYPE B[N][N] = {
    { 5.24297, -4.13041},
    { 4.06349,  8.94448}};


DATA_TYPE x[N] = { 5.88952,  7.19583};

DATA_TYPE y[N];

uint32_t finished = 0;

#endif // GESUMMV_DATA_H_
