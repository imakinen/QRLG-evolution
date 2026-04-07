### Description

The programs `main.py` and `evolution.jl` compute the time evolution of a given initial state in the one-vertex model of quantum-reduced loop gravity. They were used to perform the calculations presented in [arXiv:2604.00999](https://arxiv.org/abs/2604.00999) on the dynamics of semiclassical states in the one-vertex model.

The data used to create the plots in the article can be reproduced by running the script `run.sh`. For the Lorentzian model, computations at the chosen cutoff value require about 25 GB of RAM, and running the full set of computations included in the script takes several days on a consumer-grade CPU.

### Usage

```
python main.py j_0 c_0 t j_max T steps m
```

```
julia evolution.jl j_0 c_0 t j_max T steps m
```

#### Arguments

* `j_0`, `c_0` Parameters defining the classical phase space point on which the initial state is peaked
* `t` Semiclassicality parameter of the initial state
* `j_max` The cutoff value for the truncated Hilbert space
* `T` Length of the time interval
* `steps` Number of time steps
* `m` Dimension of the Krylov subspace. Used only by the Julia program, but is passed to both programs for consistency

Non-integer values can be given either as decimal numbers or as fractions. This applies also to the optional arguments described below.

#### Optional arguments

* `--beta beta`\
Value of the Barbero--Immirzi parameter for the Lorentzian Hamiltonian

* `--abc kind alpha beta gamma`\
Consider the 'toy' Hamiltonian of the 'first kind' (`kind = 1`) or the 'second kind' (`kind = 2`) defined respectively by Eqs. (6.1) and (6.2) in [2604.00999](https://arxiv.org/abs/2604.00999)

If neither `--beta` nor `--abc` is passed, the Euclidean Hamiltonian is used.

* `--restore`\
(Julia program only.) Continue the computation from a previously stored state vector. The program saves a backup of the state vector to disk after each time step.

### Dependencies

Python: `numpy`, `scipy`

Julia: `ExponentialUtilities`, `MatrixMarket`, `ArgParse`, `LoggingExtras`
