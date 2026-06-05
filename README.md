Integrate reaction diffusion equations of the form

$$
\frac{\partial u_i}{\partial t} = \Sigma_{ij} \Theta_{j},
$$

where $u_i$ are the local concentration fields, $\Theta_{j}$ is a collection of dictionary of terms, and $\Sigma_{ij}$ are (potentially sparse) coefficients.
The general form for the dictionary terms is a monomial

$$
\Theta_{j} = \prod_k (U_k)^{\eta_k}
$$

where 

$$
U = (u_i,\partial_{x_j}u_i, \partial_{x_j}\partial_{x_k}u_i)
$$

is the collection of the fields and their spatial derivatives up to second order.

The integration will be carried out with cuda on a GPU using the 4/5 Dormand-Prince Runge Kutta method, with adaptive timestepping, based on code from https://github.com/znicolaou/kuramoto_dmd. 
Compile with
```
nvcc -lcuda -lcublas -lcufft -O3 -o rdcu dp45_64.cu rdcu.cu
```
The code is only partially completed.
