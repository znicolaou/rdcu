//Zachary G. Nicolaou 6/24/26
//BDF stepper on the gpu
//requires sparse jacobian input (as in finite-difference discretizations)
//implicit linear system solved with sparse lu cudss
#include "bdf.h"

//bdf coefficients
const double kappaloc[6] = {0.0,-0.1850,-1.0/9,-0.0823,-0.0415,0.0};
const double gammabdfloc[6] = {0.0,1.0,1.0+1.0/2,1.0+1.0/2+1.0/3,1.0+1.0/2+1.0/3+1.0/4,1.0+1.0/2+1.0/3+1.0/4+1.0/5};

static double *gammabdf;
static double zero=0.0, one=1.0, mone=-1.0, nconststeps=0;
static double *yscale, *f, *d, *dd, *b, *D, *Dnew, *ytemp, *yerr, *y_eval, *vals, *Ivals, *Avals, ntol;
static double R[36], U[36], *RUloc;
static double *RU;
static int *rows, *cols, *Irows, *Icols, *Arows, *Acols;
static void *buffer;

static unsigned long int N;
static void (*dydt)(double, double*, double*, void*) = NULL;
static void (*jac)(double, double*, int*, int*, int*, double*, void*) = NULL;
static double atl, rtl;
static int fixed, nwit=4, refactor=1, rejac=1, nnz=0, nnzA=0;
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
  cudaMemcpy(RU, RUloc, (order+1)*(order+1)*sizeof(double), cudaMemcpyHostToDevice);
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
    y[i]=atl+rtl*fabs(y[i]);
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
__global__ void rhs (double *b, double *f, double *d, double *D, const unsigned long int N, int order, double h, double *gammabdf, double alpha){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    b[i]=h/alpha*f[i];
    for (int j=1; j<=order; j++){
      b[i]-=D[N*j+i]*gammabdf[j]/alpha;
    }
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

//Attempt a BDF step
int bdf_step (double *t, double *h, double hmin, double hmax, int *order, void* pars){
  int converged=0, it=0;
  double dnorm=0, dnormold=0,rate=0,factor=1.0;
  //Calculate the predictor and store in ytemp
  step0<<<(N+255)/256, 256>>>(ytemp, D, N, *order);
  //find the scale for the newton iteration stopping 
  cudaMemcpy(yscale, ytemp, N*sizeof(double), cudaMemcpyDeviceToDevice);
  errscale<<<(N+255)/256, 256>>>(yscale, atl, rtl, N);
  cublasDscal(handle, N, &zero, d, 1);
  
  //Newton solver for implicit equation
  for (it=0; it<nwit; it++){
    //calculate f
    (*dydt)((*t)+(*h),ytemp,f,pars);
    //refactor to find LU if needed
    if(refactor){
      double alpha=-(*h)/((1-kappaloc[*order])*gammabdfloc[*order]);
      cusparseDcsrgeam2(sphandle,N,N,&one,Idesc,N,Ivals,Irows,Icols,&alpha,Jdesc,nnz,vals,rows,cols,Adesc,Avals,Arows,Acols,buffer); 
      cudssExecute(dsshandle, CUDSS_PHASE_REFACTORIZATION, config, data, A, dddss, bdss);
      refactor=0;
    }
    //calculate rhs and solve to find dd
    rhs<<<(N+255)/256, 256>>>(b, f, d, D, N, *order, *h, gammabdf, (1-kappaloc[*order])*gammabdfloc[*order]);
    cudssExecute(dsshandle, CUDSS_PHASE_SOLVE, config, data, A, dddss, bdss);

    //scale d and calculate norm for convergence test
    scale_err<<<(N+255)/256, 256>>>(dd, yscale, yerr, N);
    cublasDnrm2(handle, N, yerr, 1, &dnorm);
    dnorm /= pow(N,0.5);
    // printf("ddnorm %i %e\n", it, dnorm);
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
      step0<<<(N+255)/256, 256>>>(ytemp, D, N, *order);
      (*jac)((*t)+(*h),ytemp,&nnz,rows,cols,vals,pars);
      refactor=1;
      rejac=0;
      return 0;
    }

    //reject and half the stepsize  
    else{
      factor=0.5;
      (*h)*=factor;
      if (*h<hmin){
        return 0;
      }


      makeRU(RU, *order, factor);
      updateD<<<(N+255)/256, 256>>>(D, Dnew, N, *order, RU);
      cudaMemcpy(D, Dnew, (*order+1)*N*sizeof(double), cudaMemcpyDeviceToDevice);

      refactor=1; //optional?
      nconststeps=0;
      return 0;
    }
  }

  if(fixed){
    cublasDcopy(handle, N, ytemp, 1, D, 1);
    (*t)=(*t)+(*h);
    return 1;
  }
  
  //test for step acceptance
  double safety = (2*nwit+1)*0.9/(2*nwit+it+1);
  double errnorm;

  cudaMemcpy(yscale, ytemp, N*sizeof(double), cudaMemcpyDeviceToDevice);
  errscale<<<(N+255)/256, 256>>>(yscale, atl, rtl, N);
  cudaMemcpy(yerr, d, N*sizeof(double), cudaMemcpyDeviceToDevice);
  scale_err<<<(N+255)/256, 256>>>(d, yscale, yerr, N);
  cublasDnrm2(handle, N, yerr, 1, &errnorm);
  errnorm *= (kappaloc[*order]*gammabdfloc[*order]+1.0/(*order+1))/pow(N,0.5);
  factor=safety * pow(errnorm, (-1.0 / (*order + 1)));

  //reject and decrease h
  if (errnorm>1){
      if(factor < 0.2){
        factor=0.2;
      }
      (*h)=(*h)*factor;
      if (*h<hmin){
        return 0;
      }

      makeRU(RU, *order, factor);
      updateD<<<(N+255)/256, 256>>>(D, Dnew, N, *order, RU);
      cudaMemcpy(D, Dnew, (*order+1)*N*sizeof(double), cudaMemcpyDeviceToDevice);
      refactor=1; //optional?

      nconststeps=0;
      return 0;
  }
  
  //accept; update t, y, and D
  (*t)=(*t)+(*h);
  cudaMemcpy(D, ytemp,  N*sizeof(double), cudaMemcpyDeviceToDevice);
  rejac=1;

  cudaMemcpy(&(D[N*(*order+2)]), d,  N*sizeof(double), cudaMemcpyDeviceToDevice);
  cublasDaxpy(handle, N, &(mone), &(D[N*(*order+1)]), 1, &(D[N*(*order+2)]), 1);
  cudaMemcpy(&(D[N*(*order+1)]), d,  N*sizeof(double), cudaMemcpyDeviceToDevice);
  for (int i=*order; i>=1; i--){
    cublasDaxpy(handle, N, &one, &(D[N*(i+1)]), 1, &(D[N*(i)]), 1);
  }

  nconststeps++;
  if(nconststeps>*order) {
    //update order and h
    double errnormminus=INFINITY, errnormplus=INFINITY, factorminus=0.0, factorplus=0.0;
    //predicted rate for order-1
    if (*order>1){
      scale_err<<<(N+255)/256, 256>>>(&(D[N*(*order)]), yscale, yerr, N);
      cublasDnrm2(handle, N, yerr, 1, &errnormminus);
      errnormminus *= (kappaloc[*order-1]*gammabdfloc[*order-1]+1.0/(*order))/pow(N,0.5);
      factorminus=safety * pow(errnormminus, (-1.0 / (*order)));

    }
    //predicted rate for order+1
    if ( *order<5){
      scale_err<<<(N+255)/256, 256>>>(&(D[N*(*order+2)]), yscale, yerr, N);
      cublasDnrm2(handle, N, yerr, 1, &errnormplus);
      errnormplus *= (kappaloc[*order+1]*gammabdfloc[*order+1]+1.0/(*order+2))/pow(N,0.5);
      factorplus=safety * pow(errnormplus, (-1.0 / (*order + 2)));
    }

    int dorder=0;
    if ( *order>1 && factorminus > factor && factorminus > factorplus){
      factor=factorminus;
      dorder=-1;
    }
    else if ( *order<5 && factorplus > factor){
      factor=factorplus;
      dorder=1;
    }
    if (factor > 10){
      factor=10;
    }

    if ((*h)*factor>hmax){
      factor=hmax/(*h);
      *h=hmax;
    }
    else{
      (*h)=(*h)*factor;
    }
    *order+=dorder;

    makeRU(RU, *order, factor);    
    updateD<<<(N+255)/256, 256>>>(D, Dnew, N, *order, RU);
    cudaMemcpy(D, Dnew, (*order+1)*N*sizeof(double), cudaMemcpyDeviceToDevice);

    nconststeps=0;
    refactor=1;
  }
  return 1;
}

double* bdf_eval(const double t, const double h, const double order, const double t_eval){
  interpolate<<<(N+255)/256, 256>>>(y_eval, D, N, order, h, t_eval, t);
  return y_eval;
}

void bdf_run(double *t, double *h, double hmin, double hmax,int *order, double t1, void *pars, void (*step_eval)(double, double, double*, void*)){

  size_t bufferSize;
  double alpha=-(*h)/((1-kappaloc[*order])*gammabdfloc[*order]);
  (*dydt)(*t,D,f,pars);
  (*jac)((*t),D,&nnz,rows,cols,vals,pars);
  cusparseDcsrgeam2_bufferSizeExt(sphandle,N,N,&one,Idesc,N,Ivals,Irows,Icols,&alpha,Jdesc,nnz,vals,rows,cols,Adesc,Avals,Arows,Acols,&bufferSize);
  cudaMalloc(&buffer, bufferSize);
  cusparseXcsrgeam2Nnz(sphandle,N,N,Idesc,N,Irows,Icols,Jdesc,nnz,rows,cols,Adesc,Arows,&nnzA,buffer);
  cusparseDcsrgeam2(sphandle,N,N,&one,Idesc,N,Ivals,Irows,Icols,&alpha,Jdesc,nnz,vals,rows,cols,Adesc,Avals,Arows,Acols,buffer);

  cudssMatrixCreateCsr(&A, N, N, nnzA, Arows, NULL, Acols, Avals, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);
  cudssMatrixCreateDn(&dddss, N, 1, N, dd, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&bdss, N, 1, N, b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);

  cudssExecute(dsshandle, CUDSS_PHASE_ANALYSIS, config, data, A, dddss, bdss);
  cudssExecute(dsshandle, CUDSS_PHASE_FACTORIZATION, config, data, A, dddss, bdss);

  if(*order==0){
    cudaMemcpy (&(D[N]), f, N*sizeof(double), cudaMemcpyDeviceToDevice);
    cublasDscal(handle, N, h, &(D[N]), 1);
    *order=1;
  }

  while(*t<t1 && *h>hmin){
    // let us integrate beyond t1 so if we restart there is no artifact
    // if(*t+*h>t1)
    //   *h=t1-*t;

    int success=bdf_step (t, h, hmin, hmax, order, pars);
    if(success){
      (*step_eval)(*t,*h,D,pars);
    }
  }

  cudaFree(buffer);
  cudssMatrixDestroy(A);
  cudssMatrixDestroy(dddss);
  cudssMatrixDestroy(bdss);
}

double* bdf_init(int *order, int n, int nnzmax, double atol, double rtol, int fixedstep, double *yloc, cublasHandle_t h, void (*func)(double, double*, double*, void*), void (*jac_func)(double, double*, int*, int*, int*, double*, void*)){
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

  cudaMalloc ((void**)&yscale, N*sizeof(double));
  cudaMalloc ((void**)&yerr, N*sizeof(double));
  cudaMalloc ((void**)&ytemp, N*sizeof(double));
  cudaMalloc ((void**)&f, N*sizeof(double));
  cudaMalloc ((void**)&d, N*sizeof(double));
  cudaMalloc ((void**)&dd, N*sizeof(double));
  cudaMalloc ((void**)&b, N*sizeof(double));
  cudaMalloc ((void**)&D, 8*N*sizeof(double));
  cudaMalloc ((void**)&Dnew, 8*N*sizeof(double));
  cudaMalloc ((void**)&gammabdf, 6*sizeof(double));
  cudaHostAlloc((void**)&RUloc, 36*sizeof(double), cudaHostAllocDefault);
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
  cudaMalloc ((void**)&y_eval, N*sizeof(double));

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
  free(Ivalsloc);
  free(Irowsloc);
  free(Icolsloc);

  cudaMemcpy (gammabdf, gammabdfloc, 6*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (D, yloc, N*(*(order)+1)*sizeof(double), cudaMemcpyHostToDevice);
  cudssCreate(&dsshandle);
  cudssConfigCreate(&config);
  cudssDataCreate(dsshandle,&data);

  cusparseCreate(&sphandle);

  cusparseCreateMatDescr(&Jdesc);
  cusparseCreateMatDescr(&Idesc);
  cusparseCreateMatDescr(&Adesc);

  return D;
}

void bdf_destroy(){
  cudaFree(yscale);
  cudaFree(yerr);
  cudaFree(ytemp);
  cudaFree(f);
  cudaFree(d);
  cudaFree(dd);
  cudaFree(b);
  cudaFree(D);
  cudaFree(Dnew);
  cudaFree(gammabdf);
  cudaFree(RUloc);
  cudaFreeHost(RU);
  cudaFree(rows);
  cudaFree(cols);
  cudaFree(vals);
  cudaFree(Arows);
  cudaFree(Acols);
  cudaFree(Avals);
  cudaFree(Irows);
  cudaFree(Icols);
  cudaFree(Ivals);
  cudaFree(y_eval);

  cudssDataDestroy(dsshandle,data);
  cudssDestroy(dsshandle);
  cudssConfigDestroy(config);

  cusparseDestroy(sphandle);

  cusparseDestroyMatDescr(Jdesc);
  cusparseDestroyMatDescr(Idesc);
  cusparseDestroyMatDescr(Adesc);
}
