//Zachary G. Nicolaou 6/16/2026
//nvcc -lcufft -lcublas -O3 -o rdcu dp45_64.cu rdcu.cu
//./rdcu -N 128,128 -L 100.0,100.0 -n 2 -c 1.0,0,0,1 -c 1.0,0,6,1 -c 1.0,0,12,1 -c -2.0,0,7,1 -c -2.0,0,13,1 -c -1.0,0,0,3 -c -1.0,0,0,1,1,2 -c -0.8,0,0,2,1,1 -c -0.8,0,1,3 -c 1.0,1,1,1 -c 1.0,1,7,1 -c 1.0,1,13,1 -c 2.0,1,6,1 -c 2.0,1,12,1 -c -1.0,1,0,2,1,1 -c -1.0,1,1,3 -c 0.8,1,0,3 -c 0.8,1,0,1,1,2 -v -D3 2dcgle
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "dp45_64.h"
#include "cublas_v2.h"
#include <cufft.h>
#include <cusparse.h>
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
  double *term;
  double *ones;
  double *f;
  cufftDoubleComplex *Y;
  cufftDoubleComplex *yfft;
  int *cols;
  int *rows;
  double *vals;
  int **c;
  int *nprods;
  double *C;
  int nterms;
  double t0;
  double t1;
  int steps;
  int verbose;
  int dense;
  int stiff;
  double *t_eval;
  int n_eval;
  int eval_i;
  double *yloc;
  cufftDoubleComplex *Yloc;
  char *filebase;
  struct timeval start;
  FILE *out;
  FILE *outtimes;
  FILE *outstates;
  FILE *outf;
  FILE *outY;
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

__global__ void make_term (double* term, cufftDoubleComplex* Y, const int N, int eta) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N){
      term[i]*=pow(Y[i].x,1.0*eta);
  }
}

__global__ void make_jac_term (double* vals, int* rows, int* cols, double* term, const int N, int coloffset, int rowoffset) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<3*N){
    vals[i]*=(term[rows[i]]);
    rows[i]+=rowoffset;
    cols[i]+=coloffset;
  }
}


void makeY (double *y, void *pars){
  parameters *p = (parameters *)pars;
  cublasDcopy(p->handle, p->N*p->n, y, 1, (double *)(p->Y), 2);

  if(p->stiff){
    //find finite differences
  }

  else{
    //pseudospectral
    cufftExecZ2Z(p->plans[0], p->Y, p->yfft, CUFFT_FORWARD);
    //in principle, we could track which derivatives are necessary and only calculate those
    for (int i=0; i<p->ndim; i++){
      d1<<<(p->N*p->n+255)/256, 256>>>(p->yfft, &(p->yfft[(i+1)*p->N*p->n]), p->N, p->n, p->Ns, p->Ls, p->ndim, i);
      for (int j=0; j<p->ndim; j++){
        d1<<<(p->N*p->n+255)/256, 256>>>(&(p->yfft[(i+1)*p->N*p->n]), &(p->yfft[(j+i*p->ndim+p->ndim+1)*p->N*p->n]), p->N, p->n, p->Ns, p->Ls, p->ndim, j);
      }
    }
    cufftExecZ2Z(p->plans[1], p->yfft, p->Y, CUFFT_INVERSE);
    double scale=1.0/p->N;
    cublasDscal(p->handle, 2*p->N*p->n*(1+p->ndim+p->ndim*p->ndim), &scale, (double *)p->Y, 1);
  }
}

void dydt (double t, double *y, double *f, void *pars){
  parameters *p = (parameters *)pars;
  makeY(y, pars);
  //zero out f
  double zero=0.0;
  cublasDscal(p->handle, p->N*p->n, &zero, f, 1);
  //add terms
  for (int i=0; i<p->nterms; i++){
    cudaMemcpy(p->term, p->ones, p->N*sizeof(double), cudaMemcpyDeviceToDevice);
    //accumulate products
    for (int j=1; j<p->nprods[i]; j+=2){
      make_term<<<(p->N+255)/256, 256>>>(p->term, &(p->Y[p->N*p->c[i][j]]), p->N, p->c[i][j+1]);
    }
    //scale and add
    cublasDaxpy(p->handle, p->N, &(p->C[i]), p->term, 1, &(f[p->N*p->c[i][0]]),1);
  }
}

void jac (double t, double *y, cusparseSpMatDescr_t *Jdesc, void *pars){
  parameters *p = (parameters *)pars;
  makeY(y, pars);  //necessary? will dydt always be called first?
  //zero out J and make temporary term sparse matrix
  double zero=0.0;
  cusparseSpMatDescr_t *termdesc;
  // cublasDscal(p->handle, p->N*p->n, &zero, f, 1);
  //add terms
  for (int i=0; i<p->nterms; i++){
    //chain rule for each term 
    for (int j=0; j<p->nprods[i]; j+=2){
      cudaMemcpy(p->term, p->ones, p->N*sizeof(double), cudaMemcpyDeviceToDevice);
      //product of powers
      for (int k=0; k<p->nprods[i]; k+=2){
        int pow=p->c[i][k+1];
        if (k==j){
          pow--;
          //multiply by the current derivative power and term coefficient
          double coeff=p->C[i]*pow;
          cublasDaxpy(p->handle, p->N, &(coeff), p->term, 1, p->term,1);
        }
        make_term<<<(p->N+255)/256, 256>>>(p->term, &(p->Y[p->N*p->c[i][k]]), p->N, pow);
      }

      //copy vals of dY/dj and multiply by the corresponding term 
      cudaMemcpy(valstemp, &(p->vals[3*p->N*p->c[i][j]]), 3*p->N*sizeof(double), cudaMemcpyDeviceToDevice);
      cudaMemcpy(rowstemp, &(p->rows[3*p->N*p->c[i][j]]), 3*p->N*sizeof(double), cudaMemcpyDeviceToDevice);
      cudaMemcpy(colstemp, &(p->cols[3*p->N*p->c[i][j]]), 3*p->N*sizeof(double), cudaMemcpyDeviceToDevice);
      int coloffset=p->N*p->c[i][0];
      int rowoffset=p->N*(p->c[i][j]%p->nterms);
      make_jac_term<<<(3*p->N+255)/256, 256>>>(valstemp, rowstemp, colstemp, p->term, p->N*p->n, coloffset, rowoffset);

      //create a sparse matrix for the term
      // cusparseCreateCoo(termdesc,p->n*p->N,p->n*p->N,3*p->N,p->rowstemp,p->colstemp,p->valstemp,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
      //convert to csr
      cusparseDcoo2csr(p->sphandle, p->rowoffettemp, 3*p->N, p->n*p->N, p->rowstemp, CUSPARSE_INDEX_BASE_ZERO);
      cusparseCreateCsr(termdesc,p->n*p->N,p->n*p->N,3*p->N,p->csrrwostemp,p->colstemp,p->valstemp,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
      size_t bufferSize = 0;
      void* buffer = NULL;
      //Add to J and store in Jtemp
      cusparseSpGEAM_bufferSize(p->sphandle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,Jdesc,&one,termdesc,Jtempdesc,CUDA_R_64F,CUSPARSE_SPGEAM_ALG1,spgeamDescr,&bufferSize);
      cudaMalloc(&externalBuffer, bufferSizeInBytes);
      cusparseSpGEAM(p->sphandle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,Jdesc,&one,termdesc,Jtempdesc,CUDA_R_64F,CUSPARSE_SPGEAM_ALG1,spgeamDescr,buffer);
      cusparseSpGEAM(p->sphandle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,Jdesc,&one,termdesc,Jtempdesc,CUDA_R_64F,CUSPARSE_SPGEAM_ALG1,spgeamDescr,buffer);

      //copy vals, rows, and cols from Jtemp to 
      //Destroy the J descriptor and c

      cusparseDestroySpMat(matA);

    }
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
    fprintf(p->out,"%.3f\t%1.3e\t%1.3e\t%f\t%i\t\n",(t-p->t0)/(p->t1-p->t0), end.tv_sec-p->start.tv_sec + 1e-6*(end.tv_usec-p->start.tv_usec), (end.tv_sec-p->start.tv_sec + 1e-6*(end.tv_usec-p->start.tv_usec))/((t-p->t0+h)/(p->t1-p->t0))*(1-(t-p->t0)/(p->t1-p->t0)), h, p->steps);
  }

  if(p->dense>=1){
    fwrite(&t,sizeof(double),1,p->outtimes);
  }

  int eval=0;
  while (t >= p->t_eval[p->eval_i] && p->eval_i<p->n_eval){
    double *y_eval;
    y_eval=dp45_eval(t,p->t_eval[p->eval_i]);
    if(p->dense>=1){
      cudaMemcpy(p->yloc, y_eval, p->N*p->n*sizeof(double), cudaMemcpyDeviceToHost);
      fwrite(p->yloc,sizeof(double),p->N*p->n,p->outstates);
      // fflush(p->outstates);
    }
    if(p->dense>=2){
      dydt (p->t_eval[p->eval_i], y_eval, p->f, pars);
      cudaMemcpy(p->yloc, p->f, p->N*p->n*sizeof(double), cudaMemcpyDeviceToHost);
      fwrite(p->yloc,sizeof(double),p->N*p->n,p->outf);
      // fflush(p->outf);
    }
    if(p->dense>=3){
      cudaMemcpy(p->Yloc, p->Y, p->N*p->n*(1+p->ndim+p->ndim*p->ndim)*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
      fwrite(p->Yloc,sizeof(cufftDoubleComplex),p->N*p->n*(1+p->ndim+p->ndim*p->ndim),p->outY);
      // fflush(p->outY);
    }
    p->eval_i++;
    eval=1;
  }
  //only save fs after a eval step, in case dt is very small
  if(eval){
    cudaMemcpy(p->yloc, y, p->N*p->n*sizeof(double), cudaMemcpyDeviceToHost);

    strcpy(file,p->filebase);
    strcat(file,"fs.dat");
    FILE *outlast=fopen(file,"wb");

    fwrite(p->yloc,sizeof(double),p->N*p->n,outlast);
    fwrite(&t,sizeof(double),1,outlast);
    fwrite(&h,sizeof(double),1,outlast);
    fflush(outlast);
    fclose(outlast);
  }
}

int parse_list(const char *optarg, const char* delim, int *lst, int *len, int max_len){
  int chars=0;
  if (optarg != NULL) {
    char *optarg_copy = strdup(optarg);
    char *token = strtok(optarg_copy, delim);
    while (token != NULL) {
      lst[(*len)++]=(int)atoi(token);
      token = strtok(NULL, delim);
      if ((*len)>=max_len){
        return chars;
      }
    }
  }
  return chars;
}

int fparse_list(const char *optarg, const char* delim, double *lst, int *len, int max_len){
  int chars=0;
  if (optarg != NULL) {
    char *optarg_copy = strdup(optarg);
    char *token = strtok(optarg_copy, delim);
    while (token != NULL) {
      chars+=strlen(token)+1;
      lst[(*len)++]=(double)atof(token);
      token = strtok(NULL, delim);
      if ((*len)>=max_len){
        return chars;
      }
    }
  }
  return chars;
}

int main (int argc, char* argv[]) {
    struct timeval start,end;
    gettimeofday(&start,NULL);

    double t1=1e2, dt=1e0, atl=1e-6, rtl=0, A=1.0;
    int gpu=0, seed=1, fixed=0, n=1, ndim=1, nterms=0, ndim2=0, verbose=0, help=1, dense=1, stiff=0, reload=0;
    int Nsloc[3]={128,128,128}, *c[1024]={0}, nprods[1024]={0};
    double Lsloc[3]={1.0,1.0,1.0}, C[1024]={0};
    char ch;
    const char delim[] = ",";
    char* filebase;
  
    while (optind < argc) {
      if ((ch = getopt(argc, argv, "hvFRSn:N:L:c:A:t:d:s:D:g:r:a:")) != -1) {
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
          case 'S': {
            stiff = 1;
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
              ndim=0;
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
              c[nterms]=(int *)calloc(100,sizeof(int));
              int chars=fparse_list(optarg, delim, C, &nterms, 1+nterms);
              parse_list(&(optarg[chars]), delim, c[nterms-1], &(nprods[nterms-1]), 1024);
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
      printf("usage:\trdcu [-hvFR] [-n NFIELDS] [-N NUMS] [-L LENGTHS]\n");
      printf("\t[-c COUPLING] [-A AMPLITUDE] [-t TIME] [-d DT] [-s SEED] \n");
      printf("\t[-D DENSITY] [-g GPU] [-r RTOL] [-a ATOL]  FILEBASE \n\n");
      printf("-h for help \n");
      printf("-v for verbose output, including progress \n");
      printf("-F for fixed timestep \n");
      printf("-R to reload initial conditions from files if possible\n");
      printf("NFIELDS is the number of fields. Default 1\n");
      printf("NUMS is number of grid points in each dimension (up to three), separated by commas (no spaces). Default 128\n");
      printf("LENGTHS is domain length in each dimension (up to three), separated by commas (no spaces). Default 1.0\n");
      printf("COUPLING is a list of coupling terms, separated by commas (no spaces). Default empty\n");
      printf("\tThe first value is the term coefficent. \n");
      printf("\tThe second value is an integer specifying the field that the term appears in. \n");
      printf("\tThe following 2N values are pairs of integers specifing the indices and powers for each factor that appear in the term. \n");
      printf("\tFactor indices for the given -n and -N values appear below. \n");
      printf("\tYou may specify additional coupling terms on separate lines in this format in the input file FILEBASEcoupling.dat. \n");
      printf("AMPLITUDE is uniform random initial condition amplitude. Default 1.0 \n");
      printf("\tYou may also provide a binary input file FILEBASEic.dat with the initial condition\n");
      printf("TIME is total integration time. Default 1e2 \n");
      printf("DT is the time between outputs. Default 1e0 \n");
      printf("SEED is random seed. Default 1 \n");
      printf("GPU is index of the gpu. Default 0\n");
      printf("DENSITY is the output density. Default 1\n");
      printf("\t1 for the timesteps (FILEBASEtimes.dat) and evaluated state values (FILEBASEstates.dat), \n");
      printf("\t2 to include time derivatives (FILEBASEf.dat), and 3 to include factors (FILEBASEY.dat). \n");
      printf("RTOL is relative error tolerance. Default 1E-6\n");
      printf("ATOL is absolute error tolerance. Default 1E-6\n");
      printf("FILEBASE is base file name for output. \n");
      printf("\n");
      printf("Example: ./rdcu -N 128,128 -L 100.0,100.0 -n 2 -c 1.0,0,0,1 -c 1.0,0,6,1 -c 1.0,0,12,1 -c -2.0,0,7,1 -c -2.0,0,13,1 -c -1.0,0,0,3 -c -1.0,0,0,1,1,2 -c -0.8,0,0,2,1,1 -c -0.8,0,1,3 -c 1.0,1,1,1 -c 1.0,1,7,1 -c 1.0,1,13,1 -c 2.0,1,6,1 -c 2.0,1,12,1 -c -1.0,1,0,2,1,1 -c -1.0,1,1,3 -c 0.8,1,0,3 -c 0.8,1,0,1,1,2 -v -D3 2dcgle\n");
      printf("\n");
      printf("Indices for %i field(s) in %i dimension(s):\n", n, ndim);
      int l=0;
      for (int i=0; i<n; i++){
        printf("%i: u%i\n",l++,i);
      }
      for (int j=0; j<ndim; j++){
        for(int i=0; i<n; i++){
          printf("%i: u%i_%i\n",l++, i, j);
        }
      }
      for (int j=0; j<ndim; j++){
        for (int k=0; k<ndim; k++){
          for(int i=0; i<n; i++){
              printf("%i: u%i_%i%i\n",l++, i, j, k);
          }
        }
      }
      exit(0);
    }
    char sterms[1024][1024];    
    int l=0;
    for (int i=0; i<n; i++){
      sprintf(sterms[l++],"u%i",i);
    }
    for (int j=0; j<ndim; j++){
      for(int i=0; i<n; i++){
        sprintf(sterms[l++],"u%i_%i", i, j);
      }
    }
    for (int j=0; j<ndim; j++){
      for (int k=0; k<ndim; k++){
        for(int i=0; i<n; i++){
            sprintf(sterms[l++],"u%i_%i%i", i, j, k);
        }
      }
    }

    if(ndim2<ndim){
      printf("Warning: Number of length scales %i smaller than number of dimensions %i. Using defaults.\n", ndim2, ndim);
    }

    FILE *out, *in;
    char file[1024];
    strcpy(file,filebase);
    strcat(file,".out");
    out = fopen(file,"w");
    for (int  i=0; i<argc; i++){
      fprintf(out, "%s ", argv[i]);
    }
    fprintf(out, "\n");
    fflush(out);

    //load couplings if present
    strcpy(file,filebase);
    strcat(file, "coupling.dat");
    if (in = fopen(file,"r")){
      if (verbose){
        printf("Using coupling terms from file\n");
        fprintf(out, "Using coupling terms from file\n");
      }
      char line[1024];
      while(fscanf(in, "%s",line)==1){
        c[nterms]=(int *)calloc(1024,sizeof(int));
        int chars=fparse_list(line, delim, C, &nterms, 1+nterms);
        parse_list(&(line[chars]), delim, c[nterms-1], &(nprods[nterms-1]), 1024);
      } 
    }
    fclose(in);

    if(verbose){
      //print equations
      for (int k=0; k<n; k++){
        printf("%s'=",sterms[k]);
        fprintf(out,"%s'=",sterms[k]);
        for (int i=0; i<nterms; i++){
          if(c[i][0]==k){
            printf("%.3f",C[i]);
            fprintf(out,"%.3f",C[i]);
            for (int j=1; j<nprods[i]; j+=2){
              printf("(%s)^%i",sterms[c[i][j]], c[i][j+1]);
              fprintf(out,"(%s)^%i",sterms[c[i][j]], c[i][j+1]);
            }
            printf(" + ");
            fprintf(out," + ");
          }
        }
        printf("0\n");
        fprintf(out,"0\n");
      }
    }

    double t=0,h=1;
    int i=0,j=0,k=0;
    int *Ns;
    double *yloc, *y, *f, *onesloc, *ones, *term, *Ls;
    cufftDoubleComplex *Yloc, *Y, *yfftloc, *yfft;
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
    onesloc = (double*)calloc(N,sizeof(cufftDoubleComplex));
    yfftloc = (cufftDoubleComplex*)calloc(N*n*(1+ndim+ndim*ndim),sizeof(cufftDoubleComplex));
    Yloc = (cufftDoubleComplex*)calloc(N*n*(1+ndim+ndim*ndim),sizeof(cufftDoubleComplex));

    //device vector allocation
    cudaMalloc ((void**)&Ns, ndim*sizeof(int));
    cudaMalloc ((void**)&Ls, ndim*sizeof(double));
    cudaMalloc ((void**)&y, N*n*sizeof(double));
    cudaMalloc ((void**)&f, N*n*sizeof(double));
    cudaMalloc ((void**)&term, N*sizeof(double));
    cudaMalloc ((void**)&ones, N*sizeof(double));
    cudaMalloc ((void**)&yfft, N*n*(1+ndim+ndim*ndim)*sizeof(cufftDoubleComplex));
    cudaMalloc ((void**)&Y, N*n*(1+ndim+ndim*ndim)*sizeof(cufftDoubleComplex));
    
    //host vector initialization
    if (fixed){
      h = dt;
    }
    else{
      h = dt/100;
    }
    strcpy(file,filebase);
    strcat(file, "ic.dat");
    int loaded=0;
    if (in = fopen(file,"r")){
      loaded=1;
      printf("Using initial conditions from file\n");
      fprintf(out, "Using initial conditions from file\n");
      size_t read=fread(yloc,sizeof(double),n*N,in);
      if (read!=n*N){
        loaded=0;
        printf("initial conditions file not compatible with N!\n");
        fprintf(out,"initial conditions file not compatible with N!\n");
      }
      fclose(in);
    }

    strcpy(file,filebase);
    strcat(file, "fs.dat");
    int reloaded=0;
    if (reload && (in = fopen(file,"r"))){
      reloaded=1;
      printf("Reloading final conditions from file\n");
      fprintf(out, "Reloading final conditions from file\n");
      size_t read=fread(yloc,sizeof(double),n*N,in);
      if (read!=n*N){
        printf("final conditions file not compatible with N!\n");
        fprintf(out,"final conditions file not compatible with N!\n");
        reloaded=0;
      }
      if(reloaded){
        read=fread(&t,sizeof(double),1,in);
        read=fread(&h,sizeof(double),1,in);
        if (read!=1){
          printf("Couldn't read start time and step!\n");
          fprintf(out,"Couldn't read start time and step!\n");
          reloaded=0;
        }
      }
      fclose(in);

      printf("Restarting at t=%f with h=%f\n",t,h);
      fprintf(out,"Restarting at t=%f with h=%f\n",t,h);
    }
    if (!reloaded && !loaded) {
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

    FILE *outtimes,*outstates,*outf,*outY;
    char writetype[3]="wb";
    if(reloaded){
      writetype[0]='a';
    }
    if(dense>=1){
      strcpy(file,filebase);
      strcat(file,"times.dat");
      outtimes=fopen(file,writetype);

      strcpy(file,filebase);
      strcat(file,"states.dat");
      outstates=fopen(file,writetype);      
    }
    if(dense>=2){
      strcpy(file,filebase);
      strcat(file,"f.dat");
      outf=fopen(file,writetype);
    }
    if(dense>=3){
      strcpy(file,filebase);
      strcat(file,"Y.dat");
      outY=fopen(file,writetype);
    }

    int n0=int(t/dt)+1;
    int n1=int(t1/dt)+1;
    int n_eval=n1-n0;
    double *t_eval=(double *)calloc(n_eval,sizeof(double));
    int ind=0;
    for(int n=n0; n<n1; n++){
      t_eval[ind++]=dt*n;
    }

    for(int i=0; i<N; i++){
      onesloc[i]=1;
    }

    //device vector initialization
    cudaMemcpy(yfft, yfftloc, N*n*(1+ndim+ndim*ndim)*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice); 
    cudaMemcpy(Ns, Nsloc, ndim*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Ls, Lsloc, ndim*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ones, onesloc, N*sizeof(double), cudaMemcpyHostToDevice);

    //parameters and dp45 initialization
    parameters pars={
      .handle=handle,
      .plans=plans,
      .N=N,
      .n=n,
      .ndim=ndim,
      .Ns=Ns,
      .Ls=Ls,
      .term=term,
      .ones=ones,
      .f=f,
      .Y=Y,
      .yfft=yfft,
      .c=c,
      .nprods=nprods,
      .C=C,
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
      .start=start,
      .out=out,
      .outtimes=outtimes,
      .outstates=outstates,
      .outf=outf,
      .outY=outY,
    };
    y=dp45_init(n*N, atl, rtl, fixed, yloc, handle, &dydt);
    cudaMemcpy(y, yloc, N*n*sizeof(double), cudaMemcpyHostToDevice);

    //initial state output
    if (dense>=1 && !reload){
      fwrite(yloc,sizeof(double),N*n,outstates);
      fflush(outstates);
    }
    if (dense>=2 && !reload){
      dydt(t,y,f,&pars);
      cudaMemcpy(yloc, f, N*n*sizeof(double), cudaMemcpyDeviceToHost);
      fwrite(yloc,sizeof(double),N*n,outf);
      fflush(outf);
    }
    if (dense>=2 && !reload){
      makeY(y, &pars);
      fwrite(Yloc,sizeof(cufftDoubleComplex),N*n*(1+ndim+ndim*ndim),outY);
      fflush(outY);
    }
    double *y_eval;
    y_eval=dp45_run(&t, &h, t1, &pars, &step_eval);

    //final state output 
    y_eval=dp45_eval(t,t_eval[n_eval-1]);
    cudaMemcpy(yloc, y_eval, N*n*sizeof(double), cudaMemcpyDeviceToHost);

    strcpy(file,filebase);
    strcat(file,"fs.dat");
    FILE *outlast=fopen(file,"wb");

    fwrite(yloc,sizeof(double),N*n,outlast);
    fwrite(&t,sizeof(double),1,outlast);
    fwrite(&h,sizeof(double),1,outlast);

    fflush(outlast);
    fclose(outlast);

    gettimeofday(&end,NULL);
    printf("\nruntime: %f\n",end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec));
    fprintf(out,"\nsteps: %i\n",pars.steps);
    fprintf(out,"runtime: %f\n",end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec));
    fflush(out);

    fclose(out);
    if(dense>=1){
      fclose(outtimes);
      fclose(outstates);
    }
    if(dense>=2){
      fclose(outf);
    }
    if(dense>=3){
      fclose(outY);
    }

    free(yloc);
    free(Yloc);
    free(onesloc);
    free(yfftloc);
    cudaFree(y);
    cudaFree(f);
    cudaFree(yfft);
    cudaFree(Y);
    cudaFree(Ns);
    cudaFree(Ls);
    cudaFree(term);
    cudaFree(ones);

    dp45_destroy();
    cublasDestroy(handle);
    cufftDestroy(plans[0]);
    cufftDestroy(plans[1]);

    return 0;
}
