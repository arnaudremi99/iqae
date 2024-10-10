# General
#---
import numpy as np #
import math
import matplotlib.pyplot as plt #
from IPython.display import set_matplotlib_formats
import xlrd
from functools import partial
from scipy.optimize import curve_fit #
from scipy.interpolate import interp1d
from scipy.linalg import eig
import itertools
from itertools import product
from collections import defaultdict
import random
from tqdm import tqdm #

# Qutip
#---
from qutip.solver import Options
import qutip as qt
from qutip import tensor, identity, sigmaz, sigmax, Qobj, QobjEvo, mesolve, krylovsolve, basis, rand_ket, measurement

# D-Wave
#---
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler
import dimod
import neal
#from dwave_qbsolv import QBSolv





class Solver_parameters:
    def __init__(self, solver_type='QA', maxiter=50, maxiter_gamma=10, annealingtime=None, nsteps=None, numreads=10, N_iter_per_reads=None, alpha=0.5, linesearch_method='bisection', perturbation=0):
        self.solver_type       = solver_type
        self.numreads          = numreads
        self.maxiter           = maxiter
        self.maxiter_gamma     = maxiter_gamma
        self.linesearch_method = linesearch_method
        self.alpha             = alpha
        self.perturbation      = perturbation  # For simulating Integrated Control Errors (ICE) in quantum hardwares
        # SQA
        self.nsteps            = nsteps
        self.annealingtime     = annealingtime #in microseconds
        # SA
        self.N_iter_per_reads  = N_iter_per_reads
        
class Data_problem:
    def __init__(self, H, M, N, K):
        self.H = H
        self.M = M
        self.N = N
        self.K = K
    
class Data_solution:
    def __init__(self, u, dz, gamma=0, n=0):
        self.u     = u
        self.u_updt= u
        self.dz    = dz
        self.gamma = gamma
        self.gamma_store = []
        self.n         = n
        self.cost      = []
        self.u_store   = []
        self.gamma_bounds  = [None,None]
        self.energy_bounds = [None,None] # for secant method
        self.psi   = None

class Solver():
    def __init__(self, data_solution, data_problem, solver_parameters):
        self.data_solution     = data_solution
        self.data_problem      = data_problem
        self.solver_parameters = solver_parameters

    def solve(self):
        if self.solver_parameters.solver_type == 'SQA':
            u = self.solve_SQA()
        elif self.solver_parameters.solver_type == 'QA':
            u = self.solve_QA()
        elif self.solver_parameters.solver_type == 'SA':
            u = self.solve_SA()
        elif self.solver_parameters.solver_type == 'customSA':
            u = self.solve_customSA()
        else:
            print('error : unknown solver type')
        return u

    def init_gamma(self):
        Q_H     = assemble_Q(self.data_problem.N, self.data_problem.K, self.data_problem.H, self.data_solution.u, self.data_solution.dz)
        Q_M     = assemble_Q(self.data_problem.N, self.data_problem.K, self.data_problem.M, self.data_solution.u, self.data_solution.dz)
        factor  = 1
        if self.solver_parameters.linesearch_method == 'bisection':
            gamma_min =  self.data_solution.gamma - factor*abs(Q_H).max()
            gamma_max =  self.data_solution.gamma + factor*abs(Q_H).max()
            iteration = 0
            while True:
                self.data_solution.gamma = gamma_min
                u_min = self.solve()
                E_min = u_min@(self.data_problem.H-gamma_min*self.data_problem.M)@u_min
                self.data_solution.gamma = gamma_max
                u_max = self.solve()
                E_max = u_max@(self.data_problem.H-gamma_max*self.data_problem.M)@u_max
                self.data_solution.gamma = (gamma_min + gamma_max)/2
                # Stopping criterion
                if E_min >= 0 and E_max < 0:
                    break
                factor *= 2
                if E_min < 0:
                    gamma_min  =  self.data_solution.gamma - factor*abs(Q_H).max()
                if E_max >= 0:
                    gamma_max  =  self.data_solution.gamma + factor*abs(Q_H).max()
        
                iteration += 1
                if iteration > 10:
                    print('Bounds cannot be found !')
                    self.data_solution.gamma_bounds = [None, None]
                    return 
            self.data_solution.gamma_bounds = [gamma_min, gamma_max]            
        else:
            print('error : unknown linesearch method')

    def update_gamma(self):
        if self.solver_parameters.linesearch_method == 'bisection':
            if self.data_solution.u_updt@(self.data_problem.H-self.data_solution.gamma*self.data_problem.M)@self.data_solution.u_updt >= 0:
                self.data_solution.gamma_bounds[0] = self.data_solution.gamma
            else:
                self.data_solution.gamma_bounds[1] = self.data_solution.gamma
            self.data_solution.gamma = 0.5 * (self.data_solution.gamma_bounds[0] + self.data_solution.gamma_bounds[1])
        elif self.solver_parameters.linesearch_method == 'secant':
            print('warning : secant works in theory for the continuous objective function but once discretized it may not converge')
            # apply formula x_{n+1} = x_n - (x_n - x_{n-1})/(y_n - y_{n-1}) * y_n
            x0 = self.data_solution.gamma_bounds[0]
            y0 = self.data_solution.energy_bounds[0]
            x1 = self.data_solution.gamma_bounds[1]
            y1 = self.data_solution.energy_bounds[1]
            x2 = x1 - (x1-x0)/(y1-y0) * y1
            print(f'x0 = {x0}')
            print(f'x1 = {x1}')
            print(f'x2 = {x2}')
            self.data_solution.gamma_bounds[1] = x2
            self.data_solution.gamma_bounds[0] = x1
            self.data_solution.gamma = x2
            u2 = self.solve()
            if np.linalg.norm(u2) > 0:
                y2 = np.linalg.norm(u2)**(-2) * u2@(self.data_problem.H-self.data_solution.gamma*self.data_problem.M)@u2
                print(f'y2 = {y2}')
            else:
                y2 = 0
            print(f'y0 = {y0}')
            print(f'y1 = {y1}')
            print(f'y2 = {y2}')
            self.data_solution.energy_bounds[1] = y2
            self.data_solution.energy_bounds[0] = y1
        else:
            print('error : unknown linesearch method')
        return 

    def update_dz(self):
        self.data_solution.dz *= self.solver_parameters.alpha

    def update_cost(self):
        self.data_solution.cost.append(np.linalg.norm(self.data_solution.u)**(-1) * np.linalg.norm(self.data_problem.H@self.data_solution.u - self.data_solution.gamma*self.data_problem.M@self.data_solution.u))
        self.data_solution.gamma_store.append(self.data_solution.gamma)
    def solve_SQA(self):
        Q   = self.get_qubo()
        J,h = self.get_ising()
        
        oper_partial = partial(oper, J=J, h=h, t_f=self.solver_parameters.annealingtime)
        H_t = QobjEvo(oper_partial)
        
        # Simulate Schrodinger equation
        psi0     = transverse_ground_state(self.data_problem.N * self.data_problem.K)
        tlist    = np.linspace(0.0, 0.99*self.solver_parameters.annealingtime, 100)
        options  = {"nsteps": self.solver_parameters.nsteps}
        output   = mesolve(H_t, rho0=psi0, tlist=tlist, options=options)
        self.data_solution.psi = output.states[-1].full()
        q = get_minimum_energy_config( sample_psi(self.data_solution.psi, self.solver_parameters.numreads) , Q)
        v = to_decimal(q, self.data_problem.N, self.data_problem.K, self.data_solution.u, self.data_solution.dz)
        
        return v


    def solve_QA(self):
        # Checks annealing type
        if self.solver_parameters.annealingtime < 0.5:
            print(f'annealing time = {1000*self.solver_parameters.annealingtime} ns < 500 ns, fast annealing protocol is selected')
            fast_anneal=True
        else:
            fast_anneal=False

        def logical_solution(sample_set,Q,N,K):
            q = list( sample_set.first.sample.values() )
            k = 0
            Q_mapping = defaultdict(int)  # keys are the logical variables, values are the index at which they are stored
            for vars in list(sample_set.variables):
                if vars in range(0,N*K):
                    Q_mapping[vars] += k
                k+=1
            q_logical = np.zeros([N*K])
            for i in range(N*K):
                q_logical[i] = q[Q_mapping[(i)]]
            return q_logical

        
        Q = self.get_qubo()

        noise = True
        if noise==True:
            maxval = max([abs(Q.max()), abs(Q.min())])
            for i in range(len(Q)):
                for j in range(len(Q)):
                    Q[i,j] += (np.random.normal(0, self.solver_parameters.perturbation)) * maxval


        Q_dict   = self.matrix_to_dict(Q)
        sampler  = EmbeddingComposite(DWaveSampler())

        S        = sampler.sample_qubo(Q_dict, num_reads=self.solver_parameters.numreads, annealing_time=self.solver_parameters.annealingtime, fast_anneal=fast_anneal, label='IQAE')
        q        = logical_solution(S,Q_dict,self.data_problem.N,self.data_problem.K)
        v        = to_decimal(q, self.data_problem.N, self.data_problem.K, self.data_solution.u, self.data_solution.dz)
        return v

    def solve_SA(self):
        # Checks annealing type
        if self.solver_parameters.annealingtime < 0.5:
            print(f'annealing time = {1000*self.solver_parameters.annealingtime} ns < 500 ns, fast annealing protocol is selected')
            fast_anneal=True
        else:
            fast_anneal=False

        def logical_solution(sample_set,Q,N,K):
            q = list( sample_set.first.sample.values() )
            k = 0
            Q_mapping = defaultdict(int)  # keys are the logical variables, values are the index at which they are stored
            for vars in list(sample_set.variables):
                if vars in range(0,N*K):
                    Q_mapping[vars] += k
                k+=1
            q_logical = np.zeros([N*K])
            for i in range(N*K):
                q_logical[i] = q[Q_mapping[(i)]]
            return q_logical

        
        Q = self.get_qubo()

        noise = True
        if noise==True:
            maxval = max([abs(Q.max()), abs(Q.min())])
            for i in range(len(Q)):
                for j in range(len(Q)):
                    Q[i,j] += (np.random.normal(0, self.solver_parameters.perturbation)) * maxval


        Q_dict   = self.matrix_to_dict(Q)
        sampler = neal.SimulatedAnnealingSampler()

        S        = sampler.sample_qubo(Q_dict, num_reads=self.solver_parameters.numreads, annealing_time=self.solver_parameters.annealingtime, fast_anneal=fast_anneal, label='IQAE')
        q        = logical_solution(S,Q_dict,self.data_problem.N,self.data_problem.K)
        v        = to_decimal(q, self.data_problem.N, self.data_problem.K, self.data_solution.u, self.data_solution.dz)
        return v


    def solve_customSA(self):
        
        def update(s):
            i = np.random.randint(len(s))
            
            s_proposal = np.copy(s)
            s_proposal[i] *= -1
    
            return s_proposal
        
        def objective_function(J,h,s):
            Energy = s@J@s + h@s
            return Energy

        def simulated_annealing(J, h, N_spins, n_iter_per_reads, num_reads):
            s   = np.ones([num_reads, N_spins])
            for read in range(num_reads):
                s[read,:] = np.ones(N_spins) # not random for "better" reproductibility
                
            E   = np.zeros(num_reads)
            E_iter = np.zeros(n_iter_per_reads)
            p   = 1                     # geometric scheduling parameter
            T_i =  2*abs(s[0,:]@J@s[0,:] + h@s[0,:])   # Initial temperature
            T_f = T_i/100                              # Final   temperature
            t   = np.linspace(0, (T_i/T_f - 1)**(1/p) , n_iter_per_reads)
            T   = T_i/(t**p + 1)
            
            for read in range(num_reads):
                E[read] = objective_function(J,h,s[read,:])
                
                for i in range(n_iter_per_reads):
                    s_proposal = update(s[read,:])
                    E_proposal = objective_function(J,h,s_proposal)
                    dE = E_proposal - E[read]
                    
                    if np.exp(-dE/T[i]) > random.random():
                        s[read,:] = s_proposal
                        E[read] = objective_function(J,h,s[read,:])
                    E_iter[i] = objective_function(J,h,s[read,:])
                if False:
                    plt.figure(figsize=(4, 3))
                    plt.plot(t,E_iter, linestyle = '-', linewidth=2, color='blue')
                    plt.plot(t,T, linestyle = '-', linewidth=2, color='red')
                    plt.xlabel(r'$t$')
                    plt.ylabel(r'Energy')
                    plt.legend([r'$E$', r'$k_BT$'])
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.show()
        
            opt_index = np.argmin(E)
            E_opt = E[opt_index]
            s_opt = s[opt_index,:]
        
            return s_opt, E_opt

        J,h = self.get_ising()
        s, E = simulated_annealing(J,h,self.data_problem.N * self.data_problem.K, self.solver_parameters.N_iter_per_reads, self.solver_parameters.numreads)
        q    = 0.5 * (s+1)
        v    = to_decimal(q, self.data_problem.N, self.data_problem.K, self.data_solution.u, self.data_solution.dz)   

        return v

    def get_qubo(self):
        Q_H = assemble_Q(self.data_problem.N, self.data_problem.K, self.data_problem.H, self.data_solution.u, self.data_solution.dz)
        Q_M = assemble_Q(self.data_problem.N, self.data_problem.K, self.data_problem.M, self.data_solution.u, self.data_solution.dz)
        Q = Q_H - self.data_solution.gamma * Q_M
        return Q

    def get_ising(self):
        Q = self.get_qubo()
        J,h = self.qubo_to_ising(Q)
        return J,h

    def qubo_to_ising(self,Q):
        n = len(Q)
        J = np.zeros((n, n))
        h = np.zeros(n)
    
        for i in range(n):
            h[i] = (sum(Q[:, i]) + sum(Q[i,:]))/4
    
        for i in range(n):
            for j in range(i+1, n):
                J[i, j] = Q[i, j]/4
                J[j, i] = Q[j, i]/4
                
        maxval   = max([abs(J.max()), abs(J.min()), abs(h.max()), abs(h.min())])
        J       /= maxval
        h       /= maxval

        return J, h

    def matrix_to_dict(self,mat):
        Q = defaultdict(float)
        num_rows, num_columns = mat.shape
        for i in range(num_rows):
            for j in range(num_columns):
                if mat[i,j] != 0:
                    Q[(i,j)] += mat[i,j]
        return Q


#---
# functions
#---
def ising_hamiltonian(J, h):
    n = len(h)  # Number of spins
    H_lin = sum(h[i] * qt.tensor([qt.identity(2)] * i + [qt.sigmaz(),] + [qt.identity(2)] * (n - i - 1)) for i in range(n))
    H_quad = sum((J[i, j] + J[j,i]) * qt.tensor([qt.identity(2)] * i + [qt.sigmaz(),] + [qt.identity(2)] * (j - i - 1) +
                                    [qt.sigmaz(),] + [qt.identity(2)] * (n - j - 1)) for i in range(n) for j in range(i+1,n))

    H = H_lin + H_quad
    return H

def transverse_hamiltonian(h_i):
    n   = len(h_i)  # Number of spins
    H = sum(h_i[i] * tensor([identity(2)] * i + [sigmax()] + [identity(2)] * (n - i - 1)) for i in range(n))

    return H

def transverse_ground_state(N_qubits):
    # Define the basis states |0> and |1> for each qubit
    qubit_states = [ [basis(2, 0), basis(2, 1)] for _ in range(N_qubits) ]

    # Create the tensor product of basis states to represent the desired state
    psi = sum(tensor(*qs) for qs in product(*qubit_states)) / 2.0**N_qubits

    return psi


def fun_A(time, t_f):
    xls_file_path = 'scheduling_params.xls'
    workbook = xlrd.open_workbook(xls_file_path)
    sheet    = workbook.sheet_by_index(0)
    N_data   = sheet.nrows - 1

    s = np.zeros(N_data)
    A = np.zeros(N_data)
    for i in range(N_data):
        s[i] = sheet.cell_value(i+1, 0)
        A[i] = sheet.cell_value(i+1, 1)

    t = s*t_f
    return interpolate_array(t,A,time)

def fun_B(time, t_f):
    xls_file_path = 'scheduling_params.xls'
    workbook = xlrd.open_workbook(xls_file_path)
    sheet    = workbook.sheet_by_index(0)
    N_data   = sheet.nrows - 1

    s = np.zeros(N_data)
    B = np.zeros(N_data)
    for i in range(N_data):
        s[i] = sheet.cell_value(i+1, 0)
        B[i] = sheet.cell_value(i+1, 2)

    t = s*t_f
    return interpolate_array(t,B,time)

def oper(t, J, h, t_f):
    
    H_0 = transverse_hamiltonian(np.ones(len(h)))
    H_p = ising_hamiltonian(J, h)

    return - 0.5 * 1000 * fun_A(t, t_f) * H_0 + 0.5 * fun_B(t, t_f) * 1000 * H_p


def interpolate_array(s,array,S):
    interp_func = interp1d(s, array, kind='cubic')
    array_interp = interp_func(S)
    return array_interp



# Q matrix assembly
#---
def assemble_Q(N, K, M, v0 = None, dz = None):
    if v0 is None:
        v0 = np.zeros(N)
    if dz is None:
        dz = 2 * np.ones(N)
    eta = v0 - dz/2 + 2**(-K) * dz/2
    matrix_Q = np.zeros([N*K, N*K])
    for i in range(N):
        bound_i = [v0[i] - dz[i]/2, v0[i] + dz[i]/2]
        for j in range(N):
            bound_j = [v0[j] - dz[j]/2, v0[j] + dz[j]/2]
            bloc    = binarization_bloc(K, bound_i, bound_j)
            matrix_Q[K*i:K*(i+1), K*j:K*(j+1)] += (M[i,j])*bloc
            if i == j:
                bloc_diag = np.sqrt(np.diag(np.diag(bloc)))
                matrix_Q[K*i:K*(i+1), K*j:K*(j+1)] += (M[i,:] @ eta)*bloc_diag
                matrix_Q[K*i:K*(i+1), K*j:K*(j+1)] += (eta @ M[:,i])*bloc_diag
    return matrix_Q


# binarization
#---
def binarization_bloc(K, bound_k=[-1,1], bound_l=[-1,1]):
    bloc = np.ones([K,K])
    for k in range(K):
        bloc[k,:] *= (bound_k[1]-bound_k[0]) * 2**(k-K)
    for l in range(K):
        bloc[:,l] *= (bound_l[1]-bound_l[0]) * 2**(l-K)
    return bloc


# binary to decimal array converter
#---
def to_decimal(q, N, K, v0=None, dz=None):
    if v0 is None:
        v0 = np.zeros(N)
    if dz is None:
        dz = 2 * np.ones(N)
    eta = v0 - dz/2 + 2**(-K) * dz/2
    v = eta
    for i in range(N):
        for k in range(K):# q_i    *       b_i
            v[i] +=       q[K*i+k] * dz[i] * 2**(k-K)
    return v


def get_probabilities(psi):
    P = np.zeros(len(psi))
    for k in range(len(psi)):
        P[k] = np.abs(psi[k].squeeze())**2
    return P

def simulateMeasurement(P, n):
    return np.random.choice(range(len(P)), n, p=P)

def sample_psi(psi, numreads):
    N_spins = int( np.log2(len(psi)) )
    P       = get_probabilities(psi)
    qq      = np.zeros([numreads, N_spins])
    index_sampled = simulateMeasurement(P, numreads)
    for k in range(numreads):
        config_k = bin(index_sampled[k])[2:].zfill(N_spins)
        q_k = np.array([int(bit) for bit in config_k])
        # spin up == 0 ; spin down == 1 : {0,1} -> {1,0}
        qq[k,:] = -(q_k-1) 
    plt.figure(figsize=(2.5, 2))
    plt.plot(range(2**N_spins), np.abs(psi)**2, color='blue')
    plt.xlabel('spin config')
    plt.ylabel(r'$\psi^\dag\psi$')
    plt.show()
    return qq

def get_minimum_energy_config(qq, Q):
    numreads, N_spins = qq.shape
    E = np.zeros(numreads)
    for k in range(numreads):
        E[k] = qq[k,:] @ Q @ qq[k,:]
    min_ind = np.argmin(E)
    return qq[min_ind,:]


def get_projector(vec, M):
    P = np.zeros([len(vec), len(vec)])
    for i in range(len(vec)):
        for j in range(len(vec)):
            psi = M @ vec
            P[i,j] = float(np.conjugate(vec[i]) * vec[j])
            #P[i,j] = float(psi[i] * psi[j])
    return P


def save_results(iqae):
    filename_cost = f"results/cost_N={iqae.data_problem.N}_K={iqae.data_problem.K}_maxiter={iqae.solver_parameters.maxiter}_maxiter_gamma={iqae.solver_parameters.maxiter_gamma}_numreads={iqae.solver_parameters.numreads}_alpha={iqae.solver_parameters.alpha}_n={0}_annealingtime={iqae.solver_parameters.annealingtime}_solvertype={iqae.solver_parameters.solver_type}_N_flips={iqae.solver_parameters.N_iter_per_reads}_perturbation={iqae.solver_parameters.perturbation}_sym.txt"
    with open(filename_cost, 'a') as file:
        np.savetxt(file, [iqae.data_solution.cost], delimiter=' ', newline='\n', comments='')
    filename_v = f"results/v_N={iqae.data_problem.N}_K={iqae.data_problem.K}_maxiter={iqae.solver_parameters.maxiter}_maxiter_gamma={iqae.solver_parameters.maxiter_gamma}_numreads={iqae.solver_parameters.numreads}_alpha={iqae.solver_parameters.alpha}_n={0}_annealingtime={iqae.solver_parameters.annealingtime}_solvertype={iqae.solver_parameters.solver_type}_N_flips={iqae.solver_parameters.N_iter_per_reads}_perturbation={iqae.solver_parameters.perturbation}_sym.txt"
    with open(filename_v, 'a') as file:
        np.savetxt(file, [iqae.data_solution.u], delimiter=' ', newline='\n', comments='')
    filename_gamma = f"results/gamma_N={iqae.data_problem.N}_K={iqae.data_problem.K}_maxiter={iqae.solver_parameters.maxiter}_maxiter_gamma={iqae.solver_parameters.maxiter_gamma}_numreads={iqae.solver_parameters.numreads}_alpha={iqae.solver_parameters.alpha}_n={0}_annealingtime={iqae.solver_parameters.annealingtime}_solvertype={iqae.solver_parameters.solver_type}_N_flips={iqae.solver_parameters.N_iter_per_reads}_perturbation={iqae.solver_parameters.perturbation}_sym.txt"
    with open(filename_gamma, 'a') as file:
        np.savetxt(file, [iqae.data_solution.gamma_store], delimiter=' ', newline='\n', comments='')
    filename_u_store = f"results/u_store_N={iqae.data_problem.N}_K={iqae.data_problem.K}_maxiter={iqae.solver_parameters.maxiter}_maxiter_gamma={iqae.solver_parameters.maxiter_gamma}_numreads={iqae.solver_parameters.numreads}_alpha={iqae.solver_parameters.alpha}_n={0}_annealingtime={iqae.solver_parameters.annealingtime}_solvertype={iqae.solver_parameters.solver_type}_N_flips={iqae.solver_parameters.N_iter_per_reads}_perturbation={iqae.solver_parameters.perturbation}_sym.txt"
    with open(filename_u_store, 'a') as file:
        np.savetxt(file, [iqae.data_solution.u_store], delimiter=' ', newline='\n', comments='')



