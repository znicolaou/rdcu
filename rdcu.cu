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
#define PI 3.14159265358979323846

typedef struct parameters
{
  cublasHandle_t handle;
  cufftHandle *plans;
  int N;
  int n;
  int ndim;
  int *Ns;
  double *Ls;
  cufftDoubleComplex *Y;
  cufftDoubleComplex *yfft;
  int *c;
  double *C;
  int nterms;
  double t0;
  double t1;
  int steps;
  int verbose;
  int dense;
  double *t_eval;
  int n_eval;
  int eval_i;
  double *yloc;
  cufftDoubleComplex *Yloc;
  char *filebase;
  struct timeval start;
}parameters;

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
    double freq;
    if (na<Ns[axis]/2){
      freq=na*2*PI/Ls[axis];
    }
    else{
      freq=(na-Ns[axis])*2*PI/Ls[axis];
    }
    Yout[i].x=-freq*Yin[i].y;
    Yout[i].y=freq*Yin[i].x;
  }
}


void makeY (double *y, void *pars){
  parameters *p = (parameters *)pars;
  cublasDcopy(p->handle, p->N*p->n, y, 1, (double *)(p->Y), 2);

  cufftExecZ2Z(p->plans[0], p->Y, p->yfft, CUFFT_FORWARD);
  for (int i=0; i<p->ndim; i++){
    d1<<<(p->N*p->n+255)/256, 256>>>(p->yfft, &(p->yfft[(i+1)*p->N*p->n]), p->N, p->n, p->Ns, p->Ls, p->ndim, i);
    for (int j=0; j<p->ndim; j++){
      d1<<<(p->N*p->n+255)/256, 256>>>(&(p->yfft[(i+1)*p->N*p->n]), &(p->yfft[(j+i*p->ndim+p->ndim+1)*p->N*p->n]), p->N, p->n, p->Ns, p->Ls, p->ndim, j);
    }
  }
  cufftExecZ2Z(p->plans[1], p->yfft, p->Y, CUFFT_INVERSE);
  double scale=1.0/p->N;
  cublasDscal(p->handle, 2*p->N*p->n*(1+p->ndim+p->ndim*p->ndim), &scale, (double *)p->Y, 1);
  //Add the length scales for first and second derivatives too
}

__global__ void add_term (double* f, cufftDoubleComplex* Y, const int N, int eta, double C) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  //this is currently only a power of a single index of Y. We'd like to multiple multiple indices.
  //rethink how to import
  //each -c can add a list of nonzero exponents and their indices, along with a -C for the coupling constant
  //we'd probably want a make_term to loop over products in each term
  //then send the result as Y and add it here.
  //if we sent all Y here, could also loop over each product terms pow(Y[N*n[j]+i], eta[j]) within this kernel 
  //if we use i<n*N, we could do this for all fields in a single kernel. We'l loop over nprods for each nterm 
  // if (i<n*N){
      // int ifield=i/N;
  if (i<N){           
      f[i]+=C*pow(Y[i].x,eta);
  }
}

void dydt (double t, double *y, double *f, void *pars){
  parameters *p = (parameters *)pars;
  makeY(y, pars);
  //zero out f
  double scale=0.0;
  cublasDscal(p->handle, p->N*p->n, &scale, f, 1);
  //add terms
  for (int i=0; i<p->nterms; i++){
    add_term<<<(p->N+255)/256, 256>>>(&(f[p->N*p->c[3*i]]), &(p->Y[p->N*p->c[3*i+1]]), p->N, p->c[3*i+2], p->C[i]);
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

  FILE *outstates, *outstates2;
  if(p->dense>=1){
    strcpy(file,p->filebase);
    strcat(file, "states.dat");
    outstates = fopen(file,"ab");
  }
  if(p->dense>=2){
    strcpy(file,p->filebase);
    strcat(file,"Y.dat");
    outstates2 = fopen(file,"ab");
  }

  while (t >= p->t_eval[p->eval_i] && p->eval_i<p->n_eval){
    double *y_eval;
    y_eval=dp45_eval(t,p->t_eval[p->eval_i]);
    if(p->dense>=1){
      cudaMemcpy(p->yloc, y_eval, p->N*sizeof(double), cudaMemcpyDeviceToHost);
      fwrite(p->yloc,sizeof(double),p->N,outstates);
      fflush(outstates);
    }
    if(p->dense>=2){
      makeY(y_eval, pars);
      cudaMemcpy(p->Yloc, p->Y, p->N*p->n*(1+p->ndim+p->ndim*p->ndim)*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
      fwrite(p->Yloc,sizeof(cufftDoubleComplex),p->N*p->n*(1+p->ndim+p->ndim*p->ndim),outstates2);
      fflush(outstates2);
    }

    p->eval_i++;
  }
  if(p->dense>=1){
    fclose(outstates);
  }
  if(p->dense>=2){
    fclose(outstates2);
  }

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
  *len=0;
  if (optarg != NULL) {
    char *optarg_copy = strdup(optarg);
    char *token = strtok(optarg_copy, delim);
    while (token != NULL) {
      lst[(*len)++]=(int)atoi(token);
      token = strtok(NULL, delim);
      if ((*len)>max_len){
        printf("List is too long!\n");
        exit(0);
      }
    }
  }
}

void fparse_list(const char *optarg, const char* delim, double *lst, int *len, int max_len){
  *len=0;
  if (optarg != NULL) {
    char *optarg_copy = strdup(optarg);
    char *token = strtok(optarg_copy, delim);
    while (token != NULL) {
      lst[(*len)++]=(double)atof(token);
      token = strtok(NULL, delim);
      if ((*len)>max_len){
        printf("List is too long!\n");
        exit(0);
      }
    }
  }
}

int main (int argc, char* argv[]) {
    struct timeval start,end;
    gettimeofday(&start,NULL);

    double t1=1e2, dt=1e0, atl=1e-6, rtl=0, A=1.0;
    int gpu=0, seed=1, fixed=0, n=1, ndim=0, nterms=0, ndim2=0, nterms2=0, verbose=0, help=1, dense=1, reload=0;
    int Nsloc[3]={128,128,128}, cloc[300]={0};
    double Lsloc[3]={1.0,1.0,1.0}, Cloc[100]={0};
    char ch;
    const char delim[] = ",";
    char* filebase;
  
    while (optind < argc) {
      if ((ch = getopt(argc, argv, "hvFRn:N:L:c:C:A:t:d:s:D:g:r:a:")) != -1) {
        switch (ch) {
          case 'h': {
            help=1;
            break;
          }
          case 'v': {
            verbose=1;
            break;
          }
          case 'F': {
            fixed = 1;
            break;
          }
          case 'R': {
            reload = 1;
            break;
          }
          case 'n': {
            if (optarg != NULL) {
              n = (int)atoi(optarg);
            }
            break;
          }
          case 'N': {
            if (optarg != NULL) {
              parse_list(optarg, delim, Nsloc, &ndim, 3);
            }
            break;
          }
          case 'L': {
            if (optarg != NULL) {
              fparse_list(optarg, delim, Lsloc, &ndim2, 3);
            }
            break;
          }
          case 'c': {
            if (optarg != NULL) {
              parse_list(optarg, delim, cloc, &nterms, 300);
              nterms=nterms/3;
            }
            break;
          }
          case 'C': {
            if (optarg != NULL) {
              fparse_list(optarg, delim, Cloc, &nterms2, 100);
            }
            break;
          }
          case 'A': {
            if (optarg != NULL) {
              A = (double)atof(optarg);
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
      printf("\t[-A A] [-r rtol] [-a atol] [-g gpu] filebase \n\n");
      printf("-h for help \n");
      printf("-v for verbose \n");
      printf("-F for fixed timestep \n");
      printf("-R to reload initial conditions from files if possible\n");
      printf("n is number of fields. \n");
      printf("N is number of grid points in each dimension, separated by commas. \n");
      printf("c is a list of indices and exponents, separated by commas \n");
      printf("C is a list of coupling constants, separated by commas \n");
      printf("A is uniform random initial condition amplitude. Default 1.0. \n");
      printf("D is the output density\n");
      printf("t is total integration time. Default 1e2. \n");
      printf("dt is the time between outputs. Default 1e0. \n");
      printf("gpu is index of the gpu. Default 0.\n");
      printf("seed is random seed. Default 1. \n");
      printf("rtol is relative error tolerance. Default 0.\n");
      printf("atol is absolute error tolerance. Default 1e-6.\n");
      printf("filebase is base file name for output. \n");

      printf("Indices for %i fields in %i dimensions:\n", n, ndim);
      int l=0;
      for (int i=0; i<n; i++){
        printf("%i: u_%i\n",l++,i);
      }
      for(int i=0; i<n; i++){
        for (int j=0; j<ndim; j++){
          printf("%i: du_%i/dx_%i\n",l++, i, j);
        }
      }
      for(int i=0; i<n; i++){
        for (int j=0; j<ndim; j++){
          for (int k=0; k<ndim; k++){
              printf("%i: d^2u_%i/dx_%idx_%i\n",l++, i, j, k);
          }
        }
      }
      exit(0);
    }
    if(ndim2<ndim){
      printf("Warning: Number of length scales %i smaller than number of dimensions %i. Using defaults.\n", ndim2, ndim);
    }
    if(nterms2<nterms){
      printf("Warning: Number of coupling constants %i smaller than number of exponents and indices %u. Using defaults.\n", nterms2, 3*nterms);
    }

    double t=0,h=1;
    int i=0,j=0,k=0;
    FILE *out, *in;

    char file[256];
    strcpy(file,filebase);
    strcat(file,".out");
    out = fopen(file,"ab");

    int *Ns, *c;
    double *yloc, *y, *Ls, *C;
    cufftDoubleComplex *Yloc, *Y, *yfft, *yfftloc;
    int N=Nsloc[0];
    for (i=1; i<ndim; i++){
      N*=Nsloc[i];
    }

    //cuda handles and plans
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

    cufftPlanMany(&(plans[0]), ndim, Nsloc, Nsloc, 1, N, Nsloc, 1, N, CUFFT_Z2Z, n);
    cufftPlanMany(&(plans[1]), ndim, Nsloc, Nsloc, 1, N, Nsloc, 1, N, CUFFT_Z2Z, n*(1+ndim+ndim*ndim));

    //host vector allocation
    yloc = (double*)calloc(N*n,sizeof(double));
    yfftloc = (cufftDoubleComplex*)calloc(N*n*(1+ndim+ndim*ndim),sizeof(cufftDoubleComplex));
    Yloc = (cufftDoubleComplex*)calloc(N*n*(1+ndim+ndim*ndim),sizeof(cufftDoubleComplex));

    //device vector allocation
    cudaMalloc ((void**)&Ns, ndim*sizeof(int));
    cudaMalloc ((void**)&c, 3*nterms*sizeof(int));
    cudaMalloc ((void**)&C, nterms*sizeof(double));
    cudaMalloc ((void**)&Ls, ndim*sizeof(double));
    cudaMalloc ((void**)&y, N*n*sizeof(double));
    cudaMalloc ((void**)&yfft, N*n*(1+ndim+ndim*ndim)*sizeof(cufftDoubleComplex));
    cudaMalloc ((void**)&Y, N*n*(1+ndim+ndim*ndim)*sizeof(cufftDoubleComplex));
    
    
    //host vector initialization
    for (int  i=0; i<argc; i++){
      fprintf(out, "%s ", argv[i]);
    }
    fprintf(out, "\n");
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
          yloc[N*i+j] = A/2*(2.0/RAND_MAX*rand()-1);
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

    //device vector initialization
    cudaMemcpy(yfft, yfftloc, N*n*(1+ndim+ndim*ndim)*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice); 
    cudaMemcpy(Ns, Nsloc, ndim*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Ls, Lsloc, ndim*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c, cloc, 3*nterms*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(C, Cloc, nterms*sizeof(double), cudaMemcpyHostToDevice);

    //parameters and dp45 initialization
    parameters pars={
      .handle=handle,
      .plans=plans,
      .N=N,
      .n=n,
      .ndim=ndim,
      .Ns=Ns,
      .Ls=Ls,
      .Y=Y,
      .yfft=yfft,
      .c=cloc,
      .C=Cloc,
      .nterms=nterms,
      .t0=t,
      .t1=t1,
      .steps=0,
      .verbose=verbose,
      .dense=dense,
      .t_eval=t_eval,
      .n_eval=n_eval,
      .eval_i=0,
      .yloc=yloc,
      .Yloc=Yloc,
      .filebase=filebase,
      .start=start
    };
    y=dp45_init(n*N, atl, rtl, fixed, yloc, handle, &dydt);

    //initial state output
    if(!reloaded){
      if(dense>=1){
        strcpy(file,filebase);
        strcat(file,"states.dat");
        FILE *outstates=fopen(file,"wb");
        fwrite(yloc,sizeof(double),N,outstates);
        fflush(outstates);
        fclose(outstates);

        strcpy(file,filebase);
        strcat(file,"times.dat");
        FILE *outtimes=fopen(file,"wb");
        fclose(outtimes);
      }
      if(dense>=2){
        makeY(y, &pars);
        strcpy(file,filebase);
        strcat(file,"Y.dat");
        FILE *outstates=fopen(file,"wb");
        fwrite(Yloc,sizeof(cufftDoubleComplex),N*n*(1+ndim+ndim*ndim),outstates);
        fflush(outstates);
        fclose(outstates);

      }
    }

    double *y_eval;
    y_eval=dp45_run(&t, &h, t1, &pars, &step_eval);

    //final state output 
    y_eval=dp45_eval(t,t_eval[n_eval-1]);
    cudaMemcpy(yloc, y_eval, N*sizeof(double), cudaMemcpyDeviceToHost);

    strcpy(file,filebase);
    strcat(file,"fs.dat");
    FILE *outlast=fopen(file,"wb");

    fwrite(yloc,sizeof(double),N,outlast);
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
    cudaFree(y);
    cudaFree(yfft);
    cudaFree(Y);
    cudaFree(Ns);
    cudaFree(Ls);
    cudaFree(c);
    cudaFree(C);

    dp45_destroy();

    cublasDestroy(handle);

    return 0;
}
