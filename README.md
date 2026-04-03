### Description

The programs `main.py` and `evolution.jl` compute the time evolution of a given initial state in the one-vertex model of quantum-reduced loop gravity. They were used to perform the calculations presented in [arXiv:2604.00999](https://arxiv.org/abs/2604.00999) on the dynamics of semiclassical states in the one-vertex model.

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

#### Optional arguments

* `--beta beta`\
Value of the Barbero--Immirzi parameter for the Lorentzian Hamiltonian. The value `beta` can be given as an integer, a decimal number or a fraction.

* `--abc kind alpha beta gamma`\
Consider the 'toy' Hamiltonian of the 'first kind' (`kind = 1`) or the 'second kind' (`kind = 2`) defined respectively by Eqs. (6.1) and (6.2) in [2604.00999](https://arxiv.org/abs/2604.00999). The parameters `alpha`, `beta` and `gamma` can be given as integers, decimal numbers or fractions.

If neither `--beta` nor `--abc` is passed, the Euclidean Hamiltonian is used.

* `--restore`\
(Julia program only.) Continue the computation from a previously stored state vector. The program saves a backup of the state vector after each time step.

### Dependencies

Python: `numpy`, `scipy`

Julia: `ExponentialUtilities`, `MatrixMarket` `ArgParse`, `LoggingExtras`
