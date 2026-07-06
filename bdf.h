//Zachary G. Nicolaou 2/4/2024
//Dormand Prince 4/5 stepper on the GPU
#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cudss.h"
#include "cusparse.h"

int bdf_step (double *t, double *h, double hmin, double hmax, int *order, void* pars);
double* bdf_init(int *order, int n, int nnzmax,double atl, double rtl, int fixedstep, double *yloc, cublasHandle_t h, void (*dydt)(double, double*, double*, void*), void (*jac_func)(double, double*, int*, int*, int*, double*, void*));
void bdf_run(double *t, double *h, double hmin, double hmax, int *order, double t1, void *pars, void (*step_eval)(double, double, double*, void*));
void bdf_destroy();
double* bdf_eval(const double t, const double h, const double order, const double t_eval);
