# Iterative Quantum Annealing Eigensolver (IQAE)

Iterative Quantum Annealing Eigensolver (IQAE) is a solver designed to solve standard and generalized eigenvalue problems using D-Wave's quantum annealing hardware.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
---

## Installation
To use the IQAE solver, you can clone the repository and install the necessary dependencies.

```bash
# Clone the repository
git clone git@github.com:arnaudremi99/iqae.git

# Navigate to the project directory
cd iqae

# Install dependencies
pip install -r requirements.txt
```
## Usage
To use the IQAE solver for solving a generalized eigenvalue problem $H \boldsymbol u = \lambda M \boldsymbol u$, you first have to define the variables:
- load matrices $H$ and $M$ from your problem
- define data_problem(H,M,N,K): 		
	- $H$, 
	- $M$, 
	- $N$ the number of continuous variables, 
	- $K$ the number of binary variables per continuous variables
- define data_solution(u,dz,gamma) : 
	- $\boldsymbol u$ the initial solution (typically zero by default), 
	- d$\boldsymbol z$ = $\boldsymbol u_\mathrm{max} - \boldsymbol u_\mathrm{min}$ (typically a constant vector by default)
	- $\gamma$ the initial lagrange multiplier (typically zero by default)
- define solver_parameters
- define iqae(data_problem, data_solution, solver_parameters) : the solver
Afterwards, you can run the nested box algorithm iterations and gamma-search iterations (see examples)

## Examples
Example 1 - Helmholtz problem in $\mathbb{R}^3$ in homogeneous air:
```bash
cd example1
python3 example1.py
```

Example 2 - Helmholtz problem in $\mathbb{R}^{25}$ in homogeneous air:
```bash
cd example2
python3 example2.py
```

Example 3 - Helmholtz problem in $\mathbb{R}^{25}$ with heterogeneous air-SiO$_2$ medium:
```bash
cd example3
python3 example3.py
```



