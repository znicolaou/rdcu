## Description
Integrate reaction diffusion equations of the form

$$
\frac{\partial u_i}{\partial t} = \Sigma_{ij} \Theta_{j},
$$

where $u_i$ are the local concentration fields, $\Theta_{j}$ is a collection of dictionary of terms, and $\Sigma_{ij}$ are (potentially sparse) coefficients.
We assume a monomial form for the dictionary terms 

$$
\Theta_j = \prod_k (Y_k)^{\eta_{jk}}
$$

where 

$$
Y = (u_i,\partial_{x_n}u_i, \partial_{x_n}\partial_{x_m}u_i)
$$

is the collection of the fields and their spatial derivatives up to second order.

The integration is performed on a GPU using CUDA, employing the 4/5 Dormand-Prince Runge-Kutta method with adaptive timestepping, based on code from https://github.com/znicolaou/kuramoto_dmd. (n.b. it would be good to implement an adaptive BDF method for stiff systems as well)

## Usage
Compile with
```
nvcc -lcublas -lcufft -O3 -o rdcu dp45_64.cu rdcu.cu
```
Running `./rdcu -h` produces the following message:
```
usage:	rdcu [-hvFR] [-n NFIELDS] [-N NUMS] [-L LENGTHS]
	[-c COUPLING] [-A AMPLITUDE] [-t TIME] [-d DT] [-s SEED] 
	[-D DENSITY] [-g GPU] [-r RTOL] [-a ATOL]  FILEBASE 

-h for help 
-v for verbose output, including progress 
-F for fixed timestep 
-R to reload initial conditions from files if possible
NFIELDS is the number of fields. Default 1
NUMS is number of grid points in each dimension (up to three), separated by commas (no spaces). Default 128
LENGTHS is domain length in each dimension (up to three), separated by commas (no spaces). Default 1.0
COUPLING is a list of coupling terms, separated by commas (no spaces). Default empty
	The first value is the term coefficent. 
	The second value is an integer specifying the field that the term appears in. 
	The following 2N values are pairs of integers specifing the indices and powers for each factor that appear in the term. 
	Factor indices for the given -n and -N values appear below. 
	You may specify additional coupling terms on separate lines in this format in the input file FILEBASEcoupling.dat. 
AMPLITUDE is uniform random initial condition amplitude. Default 1.0 
	You may also provide a binary input file FILEBASEic.dat with the initial condition
TIME is total integration time. Default 1e2 
DT is the time between outputs. Default 1e0 
SEED is random seed. Default 1 
GPU is index of the gpu. Default 0
DENSITY is the output density. Default 1
	1 for the timesteps (FILEBASEtimes.dat) and evaluated state values (FILEBASEstates.dat), 
	2 to include time derivatives (FILEBASEf.dat), and 3 to include factors (FILEBASEY.dat). 
RTOL is relative error tolerance. Default 1E-6
ATOL is absolute error tolerance. Default 1E-6
FILEBASE is base file name for output. 

Example:
./rdcu -N 128,128 -L 100.0,100.0 -n 2 -c 1.0,0,0,1 -c 1.0,0,6,1 -c 1.0,0,12,1 \
-c -2.0,0,7,1 -c -2.0,0,13,1 -c -1.0,0,0,3 -c -1.0,0,0,1,1,2 -c -0.8,0,0,2,1,1 \
-c -0.8,0,1,3 -c 1.0,1,1,1 -c 1.0,1,7,1 -c 1.0,1,13,1 -c 2.0,1,6,1 -c 2.0,1,12,1 \
-c -1.0,1,0,2,1,1 -c -1.0,1,1,3 -c 0.8,1,0,3 -c 0.8,1,0,1,1,2 -v 2dcgle

Indices for 1 field(s) in 1 dimension(s):
0: u0
1: u0_0
2: u0_00
```
