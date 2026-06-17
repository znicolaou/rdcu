//Zachary G. Nicolaou 6/16/2026
//nvcc -lcufft -lcublas -O3 -o rdcu dp45_64.cu rdcu.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "dp45_64.h"
#include "cublas_v2.h"
#include <cufft.h>

typedef struct parameters
{
  cublasHandle_t handle;
  cufftHandle *plans;
  int N;
  int n;
  int ndim;
  int *Ns;
  cufftDoubleComplex *Y;
  cufftDoubleComplex *yfft;
  int *mu;
  int *nu;
  double *C;
  int *eta;
  int n_terms;
  double t0;
  double t1;
  int steps;
  int verbose;
  int dense;
  double *t_eval;
  int n_eval;
  int eval_i;
  double *yloc;
  char *filebase;
  struct timeval start;
}parameters;


__global__ void add_term (double* f, double* Y, const int N, double C, int eta) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N){
      f[i]+=C*pow(Y[i],eta);
  }
}

__global__ void d1 (cufftDoubleComplex* Yin, cufftDoubleComplex* Yout, const int N, const int n, int *Ns, double *Ls, const int ndim, const int axis) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N*n){
    //find the index for the specified axis
    int j=i%N;
    for (int d=ndim-1; d>axis; d--){
      j/=Ns[d];
    }
    int na=j%Ns[axis];

    //multiply by frequency in the specified axis
    double freq=na*Ls[axis]/Ns[axis];
    if (freq>0.5*Ls[axis]{
      freq=freq-Ls[axis];
    }
    Yout[i].x=-freq*Yin[i].y;
    Yout[i].y=freq*Yin[i].x;
  }
}


void makeY (double *y, void *pars){
  parameters *p = (parameters *)pars;
  cublasDcopy(p->handle, p->N*p->n, y, 1, (double *)(p->Y), 2);
  // cudaMemcpy(p->Y,y, p->N*p->n*sizeof(double), cudaMemcpyDeviceToDevice);

  cufftExecZ2Z(p->plans[0], p->Y, p->yfft, CUFFT_FORWARD);
  for (int i=0; i<p->ndim; i++){
    d1<<<(p->N*p->n+255)/256, 256>>>(p->yfft, &(p->yfft[(i+1)*p->N*p->n]), p->N, p->n, p->Ns, p->ndim, i);
    for (int j=0; j<p->ndim; j++){
      d1<<<(p->N*p->n+255)/256, 256>>>(&(p->yfft[(i+1)*p->N*p->n]), &(p->yfft[(j+i*p->ndim+p->ndim+1)*p->N*p->n]), p->N, p->n, p->Ns, p->ndim, j);
    }
  }
  cufftExecZ2Z(p->plans[1], p->yfft, p->Y, CUFFT_INVERSE);
  double scale=1.0/p->N;
  cublasDscal(p->handle, 2*p->N*p->n*(1+p->ndim+p->ndim*p->ndim), &scale, (double *)p->Y, 1);
  //Add the length scales for first and second derivatives too
}
void dydt (double t, double *y, double *f, void *pars){
  parameters *p = (parameters *)pars;

  makeY(y, pars);
  double scale=0.0;
  cublasDscal(p->handle, 2*p->N*p->n, &scale, f, 1);
  for (int i=0; i<p->n_terms; i++){
    add_term<<<(p->N+255)/256, 256>>>(&(f[N*mu[i]]), &(p->Y[N*nu[i]]), p->N, p->C[i], p->eta[i]);
  }

}



void step_eval(double t, double h, double* y, void *pars){
  parameters *p = (parameters *)pars;
  static char file[256];
  struct timeval end;

  p->steps++;
  if(p->verbose) {
    gettimeofday(&end,NULL);
    printf("%.3f\t%1.3e\t%1.3e\t%f\t%i\t\r",(t-p->t0)/(p->t1-p->t0), end.tv_sec-p->start.tv_sec + 1e-6*(end.tv_usec-p->start.tv_usec), (end.tv_sec-p->start.tv_sec + 1e-6*(end.tv_usec-p->start.tv_usec))/((t-p->t0+h)/(p->t1-p->t0))*(1-(t-p->t0)/(p->t1-p->t0)), h, p->steps);
    fflush(stdout);
    strcpy(file,p->filebase);
    strcat(file,".out");
    FILE *out = fopen(file,"ab");
    fprintf(out,"%.3f\t%1.3e\t%1.3e\t%f\t%i\t\n",(t-p->t0)/(p->t1-p->t0), end.tv_sec-p->start.tv_sec + 1e-6*(end.tv_usec-p->start.tv_usec), (end.tv_sec-p->start.tv_sec + 1e-6*(end.tv_usec-p->start.tv_usec))/((t-p->t0+h)/(p->t1-p->t0))*(1-(t-p->t0)/(p->t1-p->t0)), h, p->steps);
    fflush(out);
    fclose(out);
  }
  if(p->dense>=1){
    strcpy(file,p->filebase);
    strcat(file, "times.dat");
    FILE *outtimes = fopen(file,"ab");
    fwrite(&t,sizeof(double),1,outtimes);
    fflush(outtimes);
    fclose(outtimes);
  }

  FILE *outanimation;
  if(p->dense>=1){
    strcpy(file,p->filebase);
    strcat(file, "eval.dat");
    outanimation = fopen(file,"ab");
  }

  while (t >= p->t_eval[p->eval_i] && p->eval_i<p->n_eval){
    double *y_eval;
    y_eval=dp45_eval(t,p->t_eval[p->eval_i]);
    if(p->dense>=1){
      // cublasGetVector(p->N, sizeof(double), y_eval, 1, p->yloc, 1);
      cudaMemcpy(p->yloc, y_eval, p->N*sizeof(double), cudaMemcpyDeviceToHost);
      fwrite(p->yloc,sizeof(double),p->N,outanimation);
      fflush(outanimation);
    }

    p->eval_i++;
  }
  if(p->dense>=1){
    fclose(outanimation);
  }

  // cublasGetVector(p->N, sizeof(double), y, 1, p->yloc, 1);
  cudaMemcpy(p->yloc, y, p->N*sizeof(double), cudaMemcpyDeviceToHost);

  strcpy(file,p->filebase);
  strcat(file,"fs.dat");
  FILE *outlast=fopen(file,"wb");

  fwrite(p->yloc,sizeof(double),p->N,outlast);
  fwrite(&t,sizeof(double),1,outlast);
  fwrite(&h,sizeof(double),1,outlast);
  fflush(outlast);
  fclose(outlast);
}

void parse_list(const char *optarg, const char* delim, int *lst, int *len, int max_len){
  ndim=0;
  *len=0;
  if (optarg != NULL) {
    char *optarg_copy = strdup(optarg);
    char *token = strtok(optarg_copy, delim);
    while (token != NULL) {
      lst[*len++]=(int)atoi(token);
      token = strtok(NULL, delim);
      if (*len>max_len){
        printf("List is too long!");
        exit(0);
      }
    }
  }
}

int main (int argc, char* argv[]) {
    struct timeval start,end;
    gettimeofday(&start,NULL);

    double t1=1e2, dt=1e0;
    double atl=1e-6, rtl=0, I=1.0;
    int gpu=0, seed=1, fixed=0;
    int n=1, ndim=1;
    char* filebase;
    int verbose=0;
    char c;
    int help=1;
    int dense=3;
    int reload=0;
    int Nsloc[3]={128,128,128};
    const char delim[] = ",";
  
    while (optind < argc) {
      if ((c = getopt(argc, argv, "N:n:I:D:g:t:d:s:r:a:hvFR")) != -1) {
        switch (c) {
          case 'N': {
            if (optarg != NULL) {
              parse_list(optarg, delim, Nsloc, &ndim, 3);
              // char *optarg_copy = strdup(optarg);
              // char *token = strtok(optarg_copy, delim);
              // while (token != NULL) {
              //   Nsloc[ndim++]=(int)atoi(token);
              //   token = strtok(NULL, delim);
              //   if (ndim>3){
              //     printf("Too many dimension!");
              //     return 0;
              //   }
              }
            }
            break;
          }
          case 'n': {
            if (optarg != NULL) {
              n = (int)atoi(optarg);
            }
            break;
          }
          case 'I': {
            if (optarg != NULL) {
              I = (double)atof(optarg);
            }
            break;
          }
          case 'D':{
            if (optarg != NULL) {
              dense = (int)atoi(optarg);
            }
            break;
          }
          case 'g': {
            if (optarg != NULL) {
              gpu = (int)atoi(optarg);
            }
            break;
          }
          case 't': {
            if (optarg != NULL) {
              t1 = (double)atof(optarg);
            }
            break;
          }
          case 'd': {
            if (optarg != NULL) {
              dt = (double)atof(optarg);
            }
            break;
          }
          case 's': {
            if (optarg != NULL) {
              seed = (int)atoi(optarg);
            }
            break;
          }
          case 'r': {
            if (optarg != NULL) {
              rtl = (double)atof(optarg);
            }
            break;
          }
          case 'a': {
            if (optarg != NULL) {
              atl = (double)atof(optarg);
            }
            break;
          }
          case 'R': {
            reload = 1;
            break;
          }
          case 'F': {
            fixed = 1;
            break;
          }
          case 'h': {
            help=1;
            break;
          }
          case 'v': {
            verbose=1;
            break;
          }
        }
      }
      else {
        filebase=argv[optind];
        optind++;
        help=0;
      }
    }
    if (help) {
      printf("usage:\trdcu [-hvnRFA] [-N N] [-n n] [-D D]\n");
      printf("\t[-c c] [-e e] [-i i] [-t t] [-d dt] [-s seed] \n");
      printf("\t[-I I] [-r rtol] [-a atol] [-g gpu] filebase \n\n");
      printf("-h for help \n");
      printf("-v for verbose \n");
      printf("-R to reload initial conditions from files if possible\n");
      printf("-F for fixed timestep \n");
      printf("D is the output density\n");
      printf("N is number of grid points in each dimension, separated by commas. \n");
      printf("n is number of fields. \n");
      printf("c is a list of coupling constants, separated by commas \n");
      printf("e is a list of exponents, separated by commas \n");
      printf("i is a list of indices, separated by commas \n");
      printf("I is uniform random initial condition scale. Default 1. \n");
      printf("t is total integration time. Default 1e2. \n");
      printf("dt is the time between outputs. Default 1e0. \n");
      printf("seed is random seed. Default 1. \n");
      printf("rtol is relative error tolerance. Default 0.\n");
      printf("atol is absolute error tolerance. Default 1e-6.\n");
      printf("gpu is index of the gpu. Default 0.\n");
      printf("filebase is base file name for output. \n");


      exit(0);
    }

    double t=0,h=1;
    int i=0,j=0,k=0;
    FILE *out, *in;

    char file[256];
    strcpy(file,filebase);
    strcat(file,".out");
    out = fopen(file,"ab");

    int *Ns;
    double *yloc, *y;
    cufftDoubleComplex *Yloc, *Y, *yfft, *yfftloc;
    int N=Nsloc[0];
    for (i=1; i<ndim; i++){
      N*=Nsloc[i];
    }
    yloc = (double*)calloc(N*n,sizeof(double));
    Yloc = (cufftDoubleComplex*)calloc(N*n*(1+ndim+ndim*ndim),sizeof(cufftDoubleComplex));
    yfftloc = (cufftDoubleComplex*)calloc(N*n*(1+ndim+ndim*ndim),sizeof(cufftDoubleComplex));

    cublasStatus_t stat;
    cublasHandle_t handle;
    cufftHandle *plans = (cufftHandle*) malloc(2 * sizeof(cufftHandle));

    cudaSetDevice(gpu);
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        fprintf (out,"CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    srand(seed);

    for (int  i=0; i<argc; i++){
      fprintf(out, "%s ", argv[i]);
    }
    fprintf(out, "\n");

    cudaMalloc ((void**)&Ns, ndim*sizeof(int));
    cudaMalloc ((void**)&y, N*n*sizeof(double));
    cudaMalloc ((void**)&yfft, N*n*(1+ndim+ndim*ndim)*sizeof(cufftDoubleComplex));
    cudaMalloc ((void**)&Y, N*n*(1+ndim+ndim*ndim)*sizeof(cufftDoubleComplex));
    if (fixed){
      h = dt;
    }
    else{
      h = dt/100;
    }

    strcpy(file,filebase);
    strcat(file, "fs.dat");

    int reloaded=0;
    if (reload && (in = fopen(file,"r"))){
      reloaded=1;
      printf("Using initial conditions from file\n");
      fprintf(out, "Using initial conditions from file\n");
      size_t read=fread(yloc,sizeof(double),n*N,in);
      if (read!=N){
        printf("initial conditions file not compatible with N!\n");
        fprintf(out,"initial conditions file not compatible with N!\n");
        reloaded=0;
      }
      if(reloaded){
        read=fread(&t,sizeof(double),1,in);
        read=fread(&h,sizeof(double),1,in);
        if (read!=1){
          printf("Couldn't read start time and step!\n");
          fprintf(out,"Couldn't read start time and step!\n");
          // reloaded=0;
        }
      }
      fclose(in);

      printf("Restarting at t=%f with h=%f\n",t,h);
      fprintf(out,"Restarting at t=%f with h=%f\n",t,h);
    }
    if (!reloaded) {
      printf("Using random initial conditions\n");
      fprintf(out, "Using random initial conditions\n");
      for(i=0; i<n; i++){
        for(j=0; j<N; j++) {
          yloc[N*i+j] = I/2*(2.0/RAND_MAX*rand()-1);
          for(k=0; k<(1+ndim+ndim*ndim); k++) {
            yfftloc[N*i*(1+ndim+ndim*ndim)+j*(1+ndim+ndim*ndim)+k] = {0.0,0.0};
          }
        }
      }

      in=fopen(file,"wb");
      fwrite(yloc,sizeof(double),N*n,in);
      fwrite(&t,sizeof(double),1,in);
      fwrite(&h,sizeof(double),1,in);
      fclose(in);

    }

    int n0=int(t/dt)+1;
    int n1=int(t1/dt)+1;
    int n_eval=n1-n0;
    double *t_eval=(double *)calloc(n_eval,sizeof(double));
    int ind=0;
    for(int n=n0; n<n1; n++){
      t_eval[ind++]=dt*n;
    }

    
    // cufftPlanMany(&(plans[0]), ndim, Nsloc, Nsloc, 1, N, Nsloc, 1, N, CUFFT_D2Z, n);
    // cufftPlanMany(&(plans[1]), ndim, Nsloc, Nsloc, 1, N, Nsloc, 1, N, CUFFT_Z2D, n*(1+ndim+ndim*ndim));
    cufftPlanMany(&(plans[0]), ndim, Nsloc, Nsloc, 1, N, Nsloc, 1, N, CUFFT_Z2Z, n);
    cufftPlanMany(&(plans[1]), ndim, Nsloc, Nsloc, 1, N, Nsloc, 1, N, CUFFT_Z2Z, n*(1+ndim+ndim*ndim));
    parameters pars={.handle=handle, .plans=plans, .N=N, .n=n, .ndim=ndim, .Ns=Ns, .Y=Y, .yfft=yfft, .t0=t, .t1=t1, .steps=0, .verbose=verbose, .dense=dense, .t_eval=t_eval, .n_eval=n_eval, .eval_i=0, .yloc=yloc, .filebase=filebase,.start=start};

    y=dp45_init(n*N, atl, rtl, fixed, yloc, handle, &dydt);
    cudaMemcpy(yfft, yfftloc, N*n*(1+ndim+ndim*ndim)*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice); 
    cudaMemcpy(Ns, Nsloc, ndim*sizeof(int), cudaMemcpyHostToDevice);

    //////////////////////////////////////////////////////////////////////////
    makeY(y, &pars);
    cudaMemcpy(yloc, y, N*n*sizeof(double), cudaMemcpyDeviceToHost);
    strcpy(file,filebase);
    strcat(file,"y.dat");
    FILE *outtmp=fopen(file,"wb");

    fwrite(yloc,sizeof(double),N*n,outtmp);

    fflush(outtmp);
    fclose(outtmp);

    cudaMemcpy(Yloc, Y, N*n*(1+ndim+ndim*ndim)*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    strcpy(file,filebase);
    strcat(file,"Y.dat");
    outtmp=fopen(file,"wb");

    fwrite(Yloc,sizeof(cufftDoubleComplex),N*n*(1+ndim+ndim*ndim),outtmp);

    fflush(outtmp);
    fclose(outtmp);
    return 0;
    //////////////////////////////////////////////////////////////////////////

    //initial state output
    if(!reloaded){
      if(dense>=1){
        strcpy(file,filebase);
        strcat(file,"states.dat");
        FILE *outanimation=fopen(file,"wb");
        fwrite(yloc,sizeof(double),N,outanimation);
        fflush(outanimation);
        fclose(outanimation);
        strcpy(file,filebase);
        strcat(file,"times.dat");
        FILE *outtimes=fopen(file,"wb");
        fclose(outtimes);
      }
    }

    double *y_eval;
    // y_eval=dp45_run(&t, &h, t1, &pars, &step_eval);

    //final state output with coupling appended
    parameters *p = &pars;
    y_eval=dp45_eval(t,p->t_eval[p->n_eval-1]);
    // cublasGetVector(p->N, sizeof(double), y_eval, 1, p->yloc, 1);
    cudaMemcpy(p->yloc, y_eval, p->N*sizeof(double), cudaMemcpyDeviceToHost);

    strcpy(file,p->filebase);
    strcat(file,"fs.dat");
    FILE *outlast=fopen(file,"wb");

    fwrite(p->yloc,sizeof(double),p->N,outlast);
    fwrite(&t,sizeof(double),1,outlast);
    fwrite(&h,sizeof(double),1,outlast);

    fflush(outlast);
    fclose(outlast);

    strcpy(file,filebase);
    strcat(file,".out");
    out = fopen(file,"ab");
    gettimeofday(&end,NULL);
    printf("\nruntime: %f\n",end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec));
    fprintf(out,"\nsteps: %i\n",pars.steps);
    fprintf(out,"runtime: %f\n",end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec));
    fflush(out);
    fclose(out);

    free(yloc);
    free(Yloc);
    free(yfftloc);
    free(Nsloc);
    cudaFree(y);
    cudaFree(yfft);
    // cudaFree(yfft2);
    // cudaFree(yfft3);
    cudaFree(Y);
    cudaFree(Ns);

    dp45_destroy();

    cublasDestroy(handle);

    return 0;
}
