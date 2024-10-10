import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from iqae import *  # Import specific function

def main():
    #---
    # Load generalized eigenvalue problem (gEVP) "H x = \lambda M x"
    #---
    H = np.loadtxt('H.txt')
    M = np.loadtxt('M.txt')
    N = len(H) # number of real variables

    data_prob = Data_problem(H=H, M=M, N=N, K=2) # Initializes the data related to the problem. Here we choose to set K=2, i.e. we have 2 binary variables per continuous variables
    data_sol  = Data_solution(u=np.zeros(N), dz=2*np.ones(N), gamma=0) # Initialize the data related to the problem solution
    solver_params = Solver_parameters(solver_type='QA', maxiter=100, maxiter_gamma=10, annealingtime=100, N_iter_per_reads=None, numreads=1000, alpha=np.sqrt(2)/2, linesearch_method='bisection', perturbation=0) # Solver parameters
    iqae = Solver(data_sol, data_prob, solver_params)


    #---
    # iqae iterations (box-algorithm, i.e. bounds refinement,indexed by i ; gamma-search algorithm indexed by j)
    #---
    for i in tqdm(range(iqae.solver_parameters.maxiter)):
        iqae.init_gamma()
        if iqae.data_solution.gamma_bounds[0] is None:
            break
        for j in range(iqae.solver_parameters.maxiter_gamma):
            iqae.data_solution.u_updt = iqae.solve()
            iqae.update_gamma()
        iqae.data_solution.u = iqae.solve()
        iqae.update_dz()
        iqae.update_cost()
        tqdm(range(iqae.solver_parameters.maxiter)).set_postfix(cost=f"{iqae.data_solution.cost[-1]}")

    #---
    # save result
    #---
    save_results(iqae)
    plt.rcParams['figure.dpi'] = 1000
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(range(len(M)), iqae.data_solution.u, ".",  markersize=5, color='#5ec962')
    plt.savefig(f'solution.png', transparent=True, bbox_inches = 'tight')

if __name__ == '__main__':
    print('Solves the 1D Helmholtz problem at high precision (100 box-algorithm iterations)')
    main()
