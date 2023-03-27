



typedef struct {
    uint32_t l;
    double a;
    uint64_t x_ptr;
    uint64_t y_ptr;
    uint64_t z_ptr;
} axpy_args_t;

typedef struct {
    uint32_t l;
    double a;
    uint64_t x_l3_ptr;
    uint64_t y_l3_ptr;
    uint64_t z_l3_ptr;
    double* x;
    double* y;
    double* z;
} axpy_local_args_t;

typedef struct {
    job_id_t id;
    axpy_args_t args;
} axpy_job_t;

typedef struct {
    job_id_t id;
    axpy_local_args_t args;
} axpy_local_job_t;



// Job function declarations
void axpy_job_dm_core(job_t* job);
void axpy_job_compute_core(job_t* job);
