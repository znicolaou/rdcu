//Zachary G. Nicolaou 6/24/26
//BDF stepper on the gpu
//requires sparse jacobian input (as in finite-difference discretizations)
//implicit linear system solved with sparse lu cudss
//iterative solver w/ preconditioning not yet implemented
#include "bdf.h"

//bdf coefficients
const double kappa[6] = {0.0000000000000000,-0.1850000000000000,-0.1111111111111111,-0.0823000000000000,-0.0415000000000000,0.0000000000000000};
const double gamma[6] = {0.0000000000000000,1.0000000000000000,1.5000000000000000,1.8333333333333333,2.0833333333333330,2.2833333333333332};
const double alpha[6] = {0.0000000000000000,1.1850000000000001,1.6666666666666667,1.9842166666666667,2.1697916666666663,2.2833333333333332};
const double beta[6] = {1.0000000000000000,0.3150000000000000,0.1666666666666667,0.0991166666666667,0.1135416666666667,0.1666666666666667};

static double *y, *d, *dd, *D, *ytemp, *yerr, *y_eval, *Avals, ntol;

static unsigned long int N;
static void (*dydt)(double, double*, double*, void*) = NULL;
static void (*jac)(double, double*, double*, int*, int*, void*) = NULL;
static double atl, rtl, t_last;
static int fixed, nwit=4, current_jac=0;
static cublasHandle_t handle;
static cudssHandle_t dsshandle;
static cudssMatrix_t A, b, x;
static cudssConfig_t config;
static cudssData_t data;

void makeR(double *R, int order, double factor){
  for (int i=1; i<=order, i++){
    for (int j=1; j<=order; j++){
      R[(order+1)*i+j]=(i-1-factor*j)/i;
    }
  }

  for (int k=0; k<=order; k++){
    for (int j=0; j<=order; j++){
    double prod=R[(order+1)*j+k];
      for (int i=0; i<=j; i++){
        prod*=R[(order+1)*i+j]
      }
    }
    R[(order+1)*j+k]=prod;
  }
}

// update backward differences when stepsize changes
__global__ void updateD (double *D, const unsigned long int N, int order, double *RU){
  if(i<N){
    for(int k=0; k<order; k++){
      double val=0;
      for(int j=0; j<order; j++){
          val+=RU[j]*D[N*j+i];
      }
      D[N*k+i]=val;
    }
  }
}

// predictor for the BDF stepper
__global__ void step0 (double *y, double *D, const unsigned long int N, int order){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    y[i]=D[i];
    for (int j=1; j<=order; j++){
      y[i]+=D[N*j+i];
    }
  }
}

// RHS for Newton solver step
__global__ void rhs (double *b, double *f, double *d, double *D, const unsigned long int N, int order, double h){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    b[i]=h*f[i];
    for (int j=1; j<=order; j++){
      b[i]+=D[N*j+i]*gamma[j];
    }
    b[i]/=alpha[order];
    b[i]-=d[i];
  }
}

__global__ void interpolate (double *y, double *D, const unsigned long int N, int order, double h, double t, double t_last) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    double factor=1.0;
    y[i]=D[i];
    for (int j=1; j<=order; j++){
      factor*=(t-t_last+j*h)/(j*h);
      y[i]+=D[N*j+i]*factor;
    }
  }
}

//Error estimate for the BDF stepper
__global__ void error (double *err, double *D, const unsigned long int N, int order) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    double factor=1.0;
    err[i]=D[N*(order+1)+i]*beta[order+1];
  }
}

__global__ void scale_err (double *d, double *errscaled, const unsigned long int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    double factor=1.0;
    errscaled[i]=d[i]/errscaled[i];
  }
}

__global__ void makeAvals (double *vals, double *Avals, int *rows, int *cols, const unsigned long int N, const unsigned long int nnz, double factor) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<nnz){
    Avals[i]*=-factor;
    if(rows[i]==cols[i]){
      Avals[i]+=1;
    }
  }
}


//Attempt a DP step
int bdf_step (double *t, double *h, void* pars){
  double norm=0;

  //Calculate the predictor and store in ytemp
  step0<<<(N+255)/256, 256>>>(y, ytemp, N, order);

  //find the scale for the newton iteration stopping 
  cublasDaxpy(handle, N, &(pars->rtol), ytemp, 1, &(pars->atol), yerr, 1);
  
  //Newton solver for implicit equation
  int converged=0;
  double dnorm=0, dnormold=0,rate=0,one=1.0,zero=0.0;
  for (int i=0; i<nwit; i++){
    //calculate f
    (*dydt)((*t)+(*h),ytemp,f,pars);
    //refactor to find LU if needed
    if(refactor){
      makeAvals<<(nnz+266)/256,256>>(vals,Avals,nnz,(*h)/((1-kappa[order])*gamma[order]));
      cudssExecute(dsshandle, CUDSS_PHASE_REFACTORIZATION, config, data, A, dd, b);
      refactor=0;
    }
    //calculate rhs and solve to find dd
    rhs<<<(N+255)/256, 256>>>(b, f, d, D, N, order, h);
    cudssExecute(dsshandle, CUDSS_PHASE_SOLVE, config, data, A, dd, b);
    //scale d and calculate norm for convergence test
    scale_err<<<(N+255)/256, 256>>>(dd, yerr, N);
    dnorm=cublasDnrm2(handle, N, d, 1, &yerr);
    if (dnormold>0){
      rate=dnorm/dnormold;
    }
    //not converging; stop
    if (rate>=1){
      break;
    }
    //extrapolated rate will not converge in nwit; stop early
    if (pow(rate,nwit-i)/(1-rate)*dnorm > ntol)){
      break;
    }
    //converged
    if (rate/(1-rate)*dnorm < ntol){
      converged=1;
      break;
    }
    //else adjust state and continue
    cublasDaxpy(handle, N, &one, dd, 1, &one, ytemp, 1);
    cublasDaxpy(handle, N, &one, dd, 1, &one, d, 1);
    dnormold=dnorm;
  }

  //no newton convergence
  if(!converged){
    //recalculate the jacobian if needed
    if(rejac){
      //ensure jac gives no repeated (row,col) pairs and the first N are the diagonal entries
      (*jac)((*t)+(*h),y,vals,rows,cols,nnz,pars);
      makeAvals<<(nnz+266)/256,256>>(vals,Avals,nnz,(*h)/((1-kappa[order])*gamma[order]));
      refactor=1;
      rejac=0;
    }

    //reject and half the stepsize  
    else{
      (*h)*=0.5;
      updateD (D, N, order, RU);
      refactor=1;
      return 0;
    }
  }

  if(fixed){
    t_last=*t;
    cublasDcopy(handle, N, ytemp, 1, y, 1);
    (*t)=(*t)+(*h);
    return 1;
  }
  
  //test for step acceptance
  safety = (2*nwit+1)*0.9/(2*nwit+i);
  error<<<(N+255)/256, 256>>>(yerr, D, N, order);
  scale_err<<<(N+255)/256, 256>>>(ytemp, yerr, N);
  errnorm=cublasDnrm2(handle, N, d, 1, &yerr);

  //reject and increase h
  if (errnorm>1){
      factor=safety * pow(error_norm, (-1 / (order + 1)));
      (*h)=(*h)*factor;
      updateD (D, N, order, RU);
      return 0;
  }
  
  //accept; update t, y, and D
  (*t)=(*t)+(*h);
  rejac=1;
  cublasDcopy(handle, N, y, 1, ytemp, 1);
  cublasDcopy(handle, N, d, 1, &(D[N*(order+2)]), 1);
  cublasDaxpy(handle, N, &(-1*one), &(D[N*(order+1)], &one, &(D[N*(order+2)]), 1);
  cublasDcopy(handle, N, &(D[N*(order+1)]), 1, d, 1);
  for (int i=order; i>=0; i--){
    cublasDaxpy(handle, N, &one, &(D[N*(order+1)]), &one, &(D[N*(i)]), 1);
  }

  //update order and h
  //predicted rate for order-1
  if (order>1){
    error<<<(N+255)/256, 256>>>(yerr, D, N, order-1);
    scale_err<<<(N+255)/256, 256>>>(ytemp, yerr, N);
    errnormminus=cublasDnrm2(handle, N, d, 1, &yerr);
    factorminus=safety * pow(error_norm, (-1 / (order)));
  }
  //predicted rate for order+1
  if (order<5){
    error<<<(N+255)/256, 256>>>(yerr, D, N, order-1);
    scale_err<<<(N+255)/256, 256>>>(ytemp, yerr, N);
    errnormplus=cublasDnrm2(handle, N, d, 1, &yerr);
    factorplus=safety * pow(error_norm, (-1 / (order + 2)));
  }
  if (factorminus < factor && factorminus < factorplus){
    factor=factorminus;
    order=order-1;
  }
  else if (factorplus < factor){
    factor=factorplus;
    order=order+1;
  }
  (*h)=(*h)*factor;
  updateD (D, N, order, RU);

  return 1;
}

double *bdf_eval(const double t,const double t_eval){
  (double *y_eval, double *D, const unsigned long int N, int order, double h, double t, double t_last)
  interpolate<<<(N+255)/256, 256>>>(y_eval, D, N, order, h, t_eval, t);
  return y_eval;
}

double* bdf_run(double *t, double *h, double t1, void *pars, void (*step_eval)(double, double, double*, void*)){

  cudaMalloc ((void**)&y_eval, N*sizeof(double));
  (*dydt)(*t,y,f,pars);
  (*jac)((*t),y,vals,rows,cols,pars);


  while(*t<t1){
    // let us integrate beyond t1 so if we restart there is no artifact
    // if(*t+*h>t1)
    //   *h=t1-*t;

    int success=bdf_step (t, h, pars);
    if(success){
      (*step_eval)(*t,*h,y,pars);
    }
  }
  return y;
}

double* bdf_init(int n, double atol, double rtol, int fixedstep, double *yloc, cublasHandle_t h, void (*func)(double, double*, double*, void*)){
  N=n;
  rtl=rtol;
  atl=atol;
  fixed=fixedstep;
  dydt=func;
  handle=h;

  (*jac)((*t),y,vals,rows,cols,nnz,pars);
  makeAvals<<(nnz+266)/256,256>>(vals,Avals,nnz,(*h)/((1-kappa[order])*gamma[order]));


  cudaMalloc ((void**)&y, N*sizeof(double));
  cudaMalloc ((void**)&yerr, N*sizeof(double));
  cudaMalloc ((void**)&ytemp, N*sizeof(double));
  cudaMemcpy (y, yloc, N*sizeof(double), cudaMemcpyHostToDevice);

  return y;
}

void bdf_destroy(){
  cudaFree(y);
  cudaFree(yerr);
  cudaFree(ytemp);
  cudaFree(y_eval);
}
