//Zachary G. Nicolaou 6/24/26
//BDF stepper on the gpu
//requires sparse jacobian input (as in finite-difference discretizations)
//implicit linear system solved with sparse lu cudss
//iterative solver w/ preconditioning not yet implemented
#include "bdf.h"

//bdf coefficients
const double kappaloc[6] = {0.0000000000000000,-0.1850000000000000,-0.1111111111111111,-0.0823000000000000,-0.0415000000000000,0.0000000000000000};
const double gammabdfloc[6] = {0.0000000000000000,1.0000000000000000,1.5000000000000000,1.8333333333333333,2.0833333333333330,2.2833333333333332};
const double alphaloc[6] = {0.0000000000000000,1.1850000000000001,1.6666666666666667,1.9842166666666667,2.1697916666666663,2.2833333333333332};
const double betaloc[6] = {1.0000000000000000,0.3150000000000000,0.1666666666666667,0.0991166666666667,0.1135416666666667,0.1666666666666667};

static double *kappa, *gammabdf, *alpha, *beta;
static double one=1.0, nconststeps=0;
static double *y, *yscale, *f, *d, *dd, *b, *D, *Dnew, *ytemp, *yerr, *y_eval, *vals, *Ivals, *Avals, ntol;
static double R[36], U[36], RUloc[36];
static double *RU;
static int *rows, *cols, *Irows, *Icols, *Arows, *Acols;

static unsigned long int N;
static void (*dydt)(double, double*, double*, void*) = NULL;
static void (*jac)(double, double*, int*, int*, int*, double*, void*) = NULL;
static double atl, rtl;
static int fixed, nwit=4, refactor=1, rejac=1, order=1, nnz=0, nnzA=0;
static cublasHandle_t handle;

static cudssHandle_t dsshandle;
static cudssMatrix_t A, bdss, dddss;
static cudssConfig_t config;
static cudssData_t data;

static cusparseHandle_t sphandle;
static cusparseMatDescr_t Jdesc, Idesc, Adesc;


void makeRU(double *RU, int order, double factor){
  //make R
  double prod;
  for (int i=0; i<=order; i++){
    for (int j=0; j<=order; j++){
      prod=1.0/max(i,1);
      for (int k=0; k<i; k++){
        prod*=(k-factor*j)/max(k,1);
      }
      R[(order+1)*i+j]=prod;
    }
  }

  //make U
  for (int i=0; i<=order; i++){
    for (int j=0; j<=order; j++){
      prod=1.0/max(i,1);
      for (int k=0; k<i; k++){
        prod*=(k-j)*1.0/max(k,1);
      }
      U[(order+1)*i+j]=prod;
    }
  }

  //make RU
  for (int i=0; i<=order; i++){
    for (int j=0; j<=order; j++){
      RUloc[(order+1)*i+j]=0;
      for (int k=0; k<=order; k++){
        RUloc[(order+1)*i+j]+=R[(order+1)*i+k]*U[(order+1)*k+j];
      }
    }
  }
}

// update backward differences when stepsize changes
__global__ void updateD (double *D, double *Dnew, const unsigned long int N, int order, double *RU){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    for(int k=0; k<=order; k++){
      double val=0;
      for(int j=0; j<=order; j++){
          val+=RU[(order+1)*j+k]*D[N*j+i];
      }
      Dnew[N*k+i]=val;
    }
  }
}

__global__ void errscale (double *y, double atl, double rtl, const unsigned long int N){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    y[i]=rtl*abs(y[i])+atl;
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
__global__ void rhs (double *b, double *f, double *d, double *D, const unsigned long int N, int order, double h, double *gammabdf, double *alpha){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    b[i]=h*f[i];
    for (int j=1; j<=order; j++){
      b[i]-=D[N*j+i]*gammabdf[j];
    }
    b[i]/=alpha[order];
    b[i]-=d[i];
  }
}

__global__ void interpolate (double *y, double *D, const unsigned long int N, int order, double h, double t_eval, double t) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    y[i]=D[i];
    double factor=1.0;
    for (int j=1; j<=order; j++){
      factor*=(t_eval-t+j*h)/(j*h);
      y[i]+=D[N*j+i]*factor;
    }
  }
}

__global__ void scale_err (double *d, double *yscale, double *yerr, const unsigned long int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    yerr[i]=d[i]/yscale[i];
  }
}

void makeA(int *nnzA, double factor){
  size_t bufferSize = 0;
  void* buffer;
  cusparseDcsrgeam2_bufferSizeExt(sphandle,N,N,&one,Idesc,N,Ivals,Irows,Icols,&factor,Jdesc,nnz,vals,rows,cols,Adesc,Avals,Arows,Acols,&bufferSize);
  cudaMalloc(&buffer, bufferSize);
  cusparseXcsrgeam2Nnz(sphandle,N,N,Idesc,N,Irows,Icols,Jdesc,nnz,rows,cols,Adesc,Arows,nnzA,buffer);
  cusparseDcsrgeam2(sphandle,N,N,&one,Idesc,N,Ivals,Irows,Icols,&factor,Jdesc,nnz,vals,rows,cols,Adesc,Avals,Arows,Acols,buffer); 
  cudaFree(buffer);
}


//Attempt a BDF step
int bdf_step (double *t, double *h, void* pars){
  int converged=0, it=0;
  double dnorm=0, dnormold=0,rate=0,zero=0.0,one=1.0,mone=-1.0,factor=1.0;
  //Calculate the predictor and store in ytemp
  step0<<<(N+255)/256, 256>>>(ytemp, D, N, order);
  cudaMemcpy(yscale, ytemp, N*sizeof(double), cudaMemcpyDeviceToDevice);
  errscale<<<(N+255)/256, 256>>>(yscale, atl, rtl, N);

  //find the scale for the newton iteration stopping 
  cublasDscal(handle, N, &zero, d, 1);
  
  //Newton solver for implicit equation
  for (it=0; it<nwit; it++){
    //calculate f
    (*dydt)((*t)+(*h),ytemp,f,pars);
    //refactor to find LU if needed
    if(refactor){
      makeA(&nnzA, -(*h)/(alphaloc[order]));
      cudssExecute(dsshandle, CUDSS_PHASE_REFACTORIZATION, config, data, A, dddss, bdss);
      refactor=0;
    }
    //calculate rhs and solve to find dd
    rhs<<<(N+255)/256, 256>>>(b, f, d, D, N, order, *h, gammabdf, alpha);
    cudssExecute(dsshandle, CUDSS_PHASE_SOLVE, config, data, A, dddss, bdss);

    //scale d and calculate norm for convergence test
    scale_err<<<(N+255)/256, 256>>>(dd, yscale, yerr, N);
    cublasDnrm2(handle, N, yerr, 1, &dnorm);
    dnorm /= pow(N,0.5);
    if (dnormold>0){
      rate=dnorm/dnormold;
    }
    //not converging; stop
    if (rate>=1){
      break;
    }
    //extrapolated rate will not converge in nwit; stop early
    if (pow(rate,nwit-it)/(1-rate)*dnorm > ntol){
      break;
    }
    //converged
    if (dnormold>0 && rate/(1-rate)*dnorm < ntol){
      converged=1;
      break;
    }
    //else adjust state and continue
    cublasDaxpy(handle, N, &one, dd, 1, ytemp, 1);
    cublasDaxpy(handle, N, &one, dd, 1, d, 1);
    dnormold=dnorm;
  }

  //no newton convergence
  if(!converged){

    //recalculate the jacobian if needed
    if(rejac){
      step0<<<(N+255)/256, 256>>>(ytemp, D, N, order);
      (*jac)((*t)+(*h),ytemp,&nnz,rows,cols,vals,pars);
      refactor=1;
      rejac=0;
      return 0;
    }

    //reject and half the stepsize  
    else{
      factor=0.5;
      (*h)*=factor;

      makeRU(RU, order, factor);
      cudaMemcpy(RU, RUloc, (order+1)*(order+1)*sizeof(double), cudaMemcpyHostToDevice);
      updateD<<<(N+255)/256, 256>>>(D, Dnew, N, order, RU);
      cudaMemcpy(D, Dnew, (order+1)*N*sizeof(double), cudaMemcpyDeviceToDevice);

      refactor=1;
      rejac=1;
      nconststeps=0;
      return 0;
    }
  }

  if(fixed){
    cublasDcopy(handle, N, ytemp, 1, y, 1);
    (*t)=(*t)+(*h);
    return 1;
  }
  
  //test for step acceptance
  double safety = (2*nwit+1)*0.9/(2*nwit+it+1);
  double errnorm;

  cudaMemcpy(yscale, ytemp, N*sizeof(double), cudaMemcpyDeviceToDevice);
  errscale<<<(N+255)/256, 256>>>(yscale, atl, rtl, N);
  cudaMemcpy(yerr, d, N*sizeof(double), cudaMemcpyDeviceToDevice);
  scale_err<<<(N+255)/256, 256>>>(yerr, yscale, yerr, N);
  cublasDnrm2(handle, N, yerr, 1, &errnorm);
  errnorm *= betaloc[order]/pow(N,0.5);
  factor=safety * pow(errnorm, (-1.0 / (order + 1)));

  //reject and decrease h
  if (errnorm>1){
      if(factor < 0.2){
        factor=0.2;
      }
      (*h)=(*h)*factor;

      makeRU(RU, order, factor);
      cudaMemcpy(RU, RUloc, (order+1)*(order+1)*sizeof(double), cudaMemcpyHostToDevice);
      updateD<<<(N+255)/256, 256>>>(D, Dnew, N, order, RU);
      cudaMemcpy(D, Dnew, (order+1)*N*sizeof(double), cudaMemcpyDeviceToDevice);
      //refactor=1;

      nconststeps=0;
      return 0;
  }
  
  //accept; update t, y, and D
  (*t)=(*t)+(*h);
  cudaMemcpy(y, ytemp,  N*sizeof(double), cudaMemcpyDeviceToDevice);
  rejac=1;

  cudaMemcpy(&(D[N*(order+2)]), d,  N*sizeof(double), cudaMemcpyDeviceToDevice);
  cublasDaxpy(handle, N, &(mone), &(D[N*(order+1)]), 1, &(D[N*(order+2)]), 1);
  cudaMemcpy(&(D[N*(order+1)]), d,  N*sizeof(double), cudaMemcpyDeviceToDevice);
  for (int i=order; i>=0; i--){
    cublasDaxpy(handle, N, &one, &(D[N*(i+1)]), 1, &(D[N*(i)]), 1);
  }

  nconststeps++;
  if(nconststeps>order) {
    //update order and h
    double errnormminus=INFINITY, errnormplus=INFINITY, factorminus=0.0, factorplus=0.0;
    //predicted rate for order-1
    if (order>1){
      cudaMemcpy(yerr, &(D[N*order]), N*sizeof(double), cudaMemcpyDeviceToDevice);
      scale_err<<<(N+255)/256, 256>>>(yerr, yscale, yerr, N);
      cublasDnrm2(handle, N, yerr, 1, &errnormminus);
      errnormminus *= betaloc[order-1]/pow(N,0.5);
      factorminus=safety * pow(errnormminus, (-1.0 / (order)));

    }
    //predicted rate for order+1
    if (order<5){
      cudaMemcpy(yerr, &(D[N*(order+2)]), N*sizeof(double), cudaMemcpyDeviceToDevice);
      scale_err<<<(N+255)/256, 256>>>(yerr, yscale, yerr, N);
      cublasDnrm2(handle, N, yerr, 1, &errnormplus);
      errnormplus *= betaloc[order+1]/pow(N,0.5);
      factorplus=safety * pow(errnormplus, (-1.0 / (order + 2)));
    }  

    int dorder=0;
    if (order>1 && factorminus > factor && factorminus > factorplus){
      factor=factorminus;
      dorder=-1;
    }
    else if (order<5 && factorplus > factor){
      factor=factorplus;
      dorder=1;
    }
    if (factor > 10){
      factor=10;
    }

    (*h)=(*h)*factor;
    order+=dorder;

    makeRU(RU, order, factor);
    cudaMemcpy(RU, RUloc, (order+1)*(order+1)*sizeof(double), cudaMemcpyHostToDevice);
    updateD<<<(N+255)/256, 256>>>(D, Dnew, N, order, RU);
    cudaMemcpy(D, Dnew, (order+1)*N*sizeof(double), cudaMemcpyDeviceToDevice);
    
    nconststeps=0;
    refactor=1;
  }
  return 1;
}

double *bdf_eval(const double t, const double h, const double t_eval){
  interpolate<<<(N+255)/256, 256>>>(y_eval, D, N, order, h, t_eval, t);
  return y_eval;
}

double* bdf_run(double *t, double *h, double t1, void *pars, void (*step_eval)(double, double, double*, void*)){
  cudaMalloc ((void**)&y_eval, N*sizeof(double));
  double zero=0.0;
  (*dydt)(*t,y,f,pars);
  (*jac)((*t),y,&nnz,rows,cols,vals,pars);

  makeA(&nnzA, -(*h)/(alphaloc[order]));

  cudssMatrixCreateCsr(&A, N, N, nnzA, Arows, NULL, Acols, Avals, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);
  cudssMatrixCreateDn(&dddss, N, 1, N, dd, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&bdss, N, 1, N, b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);

  cublasDscal(handle, 8*N, &zero, D, 1);
  cudaMemcpy (&(D[0]), y, N*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpy (&(D[N]), f, N*sizeof(double), cudaMemcpyDeviceToDevice);
  cublasDscal(handle, N, h, &(D[N]), 1);
  cublasDscal(handle, N, &zero, dd, 1);

  step0<<<(N+255)/256, 256>>>(ytemp, D, N, order);
  rhs<<<(N+255)/256, 256>>>(b, f, d, D, N, order, *h, gammabdf, alpha);

  cudssExecute(dsshandle, CUDSS_PHASE_ANALYSIS, config, data, A, dddss, bdss);
  cudssExecute(dsshandle, CUDSS_PHASE_FACTORIZATION, config, data, A, dddss, bdss);
  

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

double* bdf_init(int n, int nnzmax, double atol, double rtol, int fixedstep, double *yloc, cublasHandle_t h, void (*func)(double, double*, double*, void*), void (*jac_func)(double, double*, int*, int*, int*, double*, void*)){
  N=n;
  rtl=rtol;
  atl=atol;
  ntol=0.03;
  if (pow(rtol,0.5)<0.03) {
    ntol=pow(rtol,0.5);
  }
  if (10*pow(2,-52)/rtol > ntol) {
    ntol=10*pow(2,-52)/(rtol+pow(2,-52));
  }

  fixed=fixedstep;
  dydt=func;
  jac=jac_func;

  handle=h;

  cudaMalloc ((void**)&y, N*sizeof(double));
  cudaMalloc ((void**)&yscale, N*sizeof(double));
  cudaMalloc ((void**)&yerr, N*sizeof(double));
  cudaMalloc ((void**)&ytemp, N*sizeof(double));
  cudaMalloc ((void**)&f, N*sizeof(double));
  cudaMalloc ((void**)&d, N*sizeof(double));
  cudaMalloc ((void**)&dd, N*sizeof(double));
  cudaMalloc ((void**)&b, N*sizeof(double));
  cudaMalloc ((void**)&D, 8*N*sizeof(double));
  cudaMalloc ((void**)&Dnew, 8*N*sizeof(double));
  

  cudaMalloc ((void**)&kappa, 6*sizeof(double));
  cudaMalloc ((void**)&gammabdf, 6*sizeof(double));
  cudaMalloc ((void**)&alpha, 6*sizeof(double));
  cudaMalloc ((void**)&beta, 6*sizeof(double));
  cudaMalloc ((void**)&RU, 36*sizeof(double));

  cudaMalloc ((void**)&rows, (N+1)*sizeof(int));
  cudaMalloc ((void**)&cols, nnzmax*sizeof(int));
  cudaMalloc ((void**)&vals, nnzmax*sizeof(double));
  cudaMalloc ((void**)&Arows, (N+1)*sizeof(int));
  cudaMalloc ((void**)&Acols, (nnzmax+N)*sizeof(int));
  cudaMalloc ((void**)&Avals, (nnzmax+N)*sizeof(double));
  cudaMalloc ((void**)&Irows, (N+1)*sizeof(int));
  cudaMalloc ((void**)&Icols, N*sizeof(int));
  cudaMalloc ((void**)&Ivals, N*sizeof(double));

  double *Ivalsloc=(double *) malloc(N*sizeof(double));
  int *Irowsloc=(int *) malloc((N+1)*sizeof(int));
  int *Icolsloc=(int *) malloc(N*sizeof(int));
  for (int i = 0; i < N; i++) {
    Irowsloc[i] = i;
    Icolsloc[i] = i;
    Ivalsloc[i] = 1.0;
  }
  Irowsloc[N] = N; 

  cudaMemcpy(Irows, Irowsloc, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(Icols, Icolsloc, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(Ivals, Ivalsloc, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy (y, yloc, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (D, yloc, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (gammabdf, gammabdfloc, 6*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (kappa, kappaloc, 6*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (alpha, alphaloc, 6*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (beta, betaloc, 6*sizeof(double), cudaMemcpyHostToDevice);
  cudssCreate(&dsshandle);
  cudssConfigCreate(&config);
  cudssDataCreate(dsshandle,&data);

  cusparseCreate(&sphandle);

  cusparseCreateMatDescr(&Jdesc);
  cusparseCreateMatDescr(&Idesc);
  cusparseCreateMatDescr(&Adesc);

  

  return y;
}

void bdf_destroy(){
  cudaFree(y);
  cudaFree(yerr);
  cudaFree(ytemp);
  cudaFree(y_eval);
}
