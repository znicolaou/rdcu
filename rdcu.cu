//Zachary G. Nicolaou 2/10/2024
//Integrate the Kuramoto model with adaptive Runke Kutta timestepping on a gpu
//Default adjacency, initial conditions, and frequencies follow volcano
//nvcc -lcuda -lcublas -lcurand -O3 -o kuramoto_64 dp45_64.cu kuramoto_64.cu
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
  unsigned long int N;
  double *Y;
  double *yfft;
  double *theta;
  int *eta;
  int *nu;
  int N_eta;
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


__global__ void make_dict (double* Y, const unsigned long int N, double* dict, const unsigned int N_eta, int* eta, int* nu) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N){
    for (int j=0; j<N_eta; j++){
      dict[i+N*j]*=pow(Y[nu[j]],eta[j]);
    }
  }
}

void dydt (double t, double *y, double *f, void *pars){
  parameters *p = (parameters *)pars;
  // makeY(y);
  make_dict<<<(p->N+255)/256, 256>>>(p->Y, p->N, p->theta, p->N_eta, p->eta, p->nu);

}


void planfft(double *y, int ndim, int *Ns, cufftHandle *plans){
  int N=1;
  for (int i=0; i<ndim; i++){
    N=N*Ns[i];
  }
  for (int i=0; i<ndim; i++){
    int stride=1;
    int dist=Ns[0];
    int batch_size=int(N/Ns[i]);
    for (int j=0; j<i; j++){
      stride=stride*Ns[j];
      dist=dist*Ns[j+1];
    }
    int *n=&Ns[i];
    cufftPlanMany(&plans[i], 1, n, NULL, stride, dist, NULL, stride, dist, CUFFT_D2Z, batch_size);
  }
}

void execfft(double *y, cufftDoubleComplex *yfft, int ndim, cufftHandle *plans){
    for (int i=0; i<ndim; i++){
      cufftExecD2Z(plans[i], y, yfft);
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
    strcat(file, "thetas.dat");
    outanimation = fopen(file,"ab");
  }


  double X,Y;
  static double *r=(double*)calloc(p->n_eval,sizeof(double));
  int eval_j=p->eval_i;
  while (t >= p->t_eval[eval_j] && eval_j<p->n_eval){
    eval_j++;
  }
  int num=eval_j-p->eval_i;
  int ind=0;

  while (t >= p->t_eval[p->eval_i] && p->eval_i<p->n_eval){
    double *y_eval;
    y_eval=dp45_eval(t,p->t_eval[p->eval_i]);
    if(p->dense>=1){
      cublasGetVector(p->N, sizeof(double), y_eval, 1, p->yloc, 1);
      fwrite(p->yloc,sizeof(double),p->N,outanimation);
      fflush(outanimation);
    }

    p->eval_i++;
  }
  if(p->dense>=1){
    fclose(outanimation);
  }

  cublasGetVector(p->N, sizeof(double), y, 1, p->yloc, 1);

  strcpy(file,p->filebase);
  strcat(file,"fs.dat");
  FILE *outlast=fopen(file,"wb");

  fwrite(p->yloc,sizeof(double),p->N,outlast);
  fwrite(&t,sizeof(double),1,outlast);
  fwrite(&h,sizeof(double),1,outlast);
  fflush(outlast);
  fclose(outlast);
}

int main (int argc, char* argv[]) {
    struct timeval start,end;
    gettimeofday(&start,NULL);

    double t1, dt;
    double atl, rtl, I;
    int gpu, seed, fixed;
    unsigned long int N;
    char* filebase;

    N=5000;
    I=0;

    t1=1e2;
    dt=1e0;
    gpu=0;
    seed=1;
    int verbose=0;
    rtl=0;
    atl=1e-6;
    fixed=0;
    char c;
    int help=1;
    int dense=3;
    int normal=0;
    int reload=0;
    int Ns[100];
    const char delim[] = ",";
    char* token;
    int ndim=0;

    while (optind < argc) {
      if ((c = getopt(argc, argv, "N:I:D:g:t:d:s:r:a:hvFnR")) != -1) {
        switch (c) {
          case 'N':
              token = strtok(optarg, delim);
              while (token != NULL) {
                Ns[ndim++]=(int)atoi(token);
                token = strtok(NULL, delim);
                if (ndim>=100){
                  printf("Too many dimension!");
                  return 0;
                }
              }
              break;
          case 'I':
              I = (double)atof(optarg);
              break;
          case 'g':
              gpu = (double)atof(optarg);
              break;
          case 't':
              t1 = (double)atof(optarg);
              break;
          case 'd':
              dt = (double)atof(optarg);
              break;
          case 's':
              seed = (int)atoi(optarg);
              break;
          case 'r':
              rtl = (double)atof(optarg);
              break;
          case 'a':
              atl = (double)atof(optarg);
              break;
          case 'D':
              dense = (int)atoi(optarg);
              break;
          case 'R':
              reload = 1;
              break;
          case 'F':
              fixed = 1;
              break;
          case 'h':
              help=1;
              break;
          case 'v':
              verbose=1;
              break;
          case 'n':
              normal=1;
              break;
        }
      }
      else {
        filebase=argv[optind];
        optind++;
        help=0;
      }
    }
    if (help) {
      printf("usage:\trdcu [-hvnRFA] [-N N] [-K K] [-D D]\n");
      printf("\t[-c c] [-t t] [-d dt] [-f f] [-s seed] \n");
      printf("\t[-I init] [-r rtol] [-a atol] [-g gpu] filebase \n\n");
      printf("-h for help \n");
      printf("-v for verbose \n");
      printf("-n for normal random frequencies (default is cauchy) \n");
      printf("-R to reload initial conditions from files if possible\n");
      printf("-F for fixed timestep \n");
      printf("D is the dimension\n");
      printf("N is number of oscillators. Default 5000. \n");
      printf("t is total integration time. Default 1e2. \n");
      printf("dt is the time between outputs. Default 1e0. \n");
      printf("f is the scale of the frequencies. Default 1e0. \n");
      printf("seed is random seed. Default 1. \n");
      printf("init is uniform random initial condition scale. Default 0. \n");
      printf("rtol is relative error tolerance. Default 0.\n");
      printf("atol is absolute error tolerance. Default 1e-6.\n");
      printf("gpu is index of the gpu. Default 0.\n");
      printf("filebase is base file name for output. \n");


      exit(0);
    }

    double t=0,h;
    int j=0;
    FILE *out, *in;

    char file[256];
    strcpy(file,filebase);
    strcat(file,".out");
    out = fopen(file,"ab");

    double *yloc, *y, *yfft, *Y, *theta;
    yloc = (double*)calloc(N,sizeof(double));

    cublasStatus_t stat;
    cublasHandle_t handle;
    cufftHandle *plans = (cufftHandle*) malloc(ndim * sizeof(cufftHandle));

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

    printf("%i",ndim);
    return 0;

//     size_t fr, total, req;
//     cudaMemGetInfo (&fr, &total);
//     req=(100+N)*N*sizeof(double);
//
//     printf("GPU Memory: %lu %lu %lu\n", fr, total, req);
//     fprintf(out,"GPU Memory: %lu %lu %lu\n", fr, total, req);
//     if(fr < req) {
//       printf("GPU Memory low!\n");
//       fprintf(out,"GPU Memory low!\n");
//       return 0;
//     }
//     fflush(out);

    cudaMalloc ((void**)&y, 2*N*sizeof(double));
    cudaMalloc ((void**)&yfft, 2*N*sizeof(double));
    cudaMalloc ((void**)&Y, 2*N*sizeof(double));
    cudaMalloc ((void**)&theta, 2*N*sizeof(double));

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
      size_t read=fread(yloc,sizeof(double),N,in);
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
      for(j=0; j<N; j++) {
        yloc[j] = I/2*(2.0/RAND_MAX*rand()-1);
      }
      in=fopen(file,"wb");
      fwrite(yloc,sizeof(double),N,in);
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


    parameters pars={.handle=handle, .plans=plans, .N=N, .Y=Y, .yfft=yfft, .theta=theta, .t0=t, .t1=t1, .steps=0, .verbose=verbose, .dense=dense, .t_eval=t_eval, .n_eval=n_eval, .eval_i=0, .yloc=yloc, .filebase=filebase,.start=start};

    y=dp45_init(N, atl, rtl, fixed, yloc, handle, &dydt);

    planfft(y, ndim, Ns, plans);

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
    cublasGetVector(p->N, sizeof(double), y_eval, 1, p->yloc, 1);

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
    cudaFree(y);
    cudaFree(yfft);
    cudaFree(Y);
    cudaFree(theta);

    dp45_destroy();

    cublasDestroy(handle);

    return 0;
}
