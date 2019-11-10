import pandas as pd
import numpy as np
from scipy.stats import norm
# import matplotlib.pyplot as plt
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank==0:
    starttime = time.time()

# set parameters for target/proposal distribution
mu0 = 2
mu1 = 1

target_sigma0 = 1
target_sigma1 = 2

proposal_sigma = 1

# target distribution: normal case
f = lambda x: norm(mu0, target_sigma0).pdf(x)
# energy = lambda x: -np.log(f(x))

def energy(x):
    if f(x) < 1e-20:
        return -1e10
    else:
        return -np.log(f(x))
# target distribution: binormal case
# f = lambda x: norm(mu0, target_sigma0).pdf(x) + norm(mu1, target_sigma1).pdf(x)
# energy = lambda x: -np.log(f(x))

#Proposal distribution: bivariate normal distribution
proposal = lambda x: np.random.normal(x, proposal_sigma)

## simulated annealing 
# codes adapted from AM207 Lecture 11 slides

#initialization for simulated annealing
init_params = {'solution':10, 'min_length':1000, 'max_temp':30}
num_epochs=10000
temp=1


def parallel_tempering(energy, proposal, init_params, epochs, temp):
    accumulator = []
    
    old_solution = init_params['solution']
    old_energy = energy(old_solution)
  
    accepted=0
    total=0

    
    for epoch in range(epochs):
        
        total += 1

        #propose new solution based on current solution
        new_solution = proposal(old_solution)
        
        #compute energy of new solution
        new_energy = energy(new_solution)

        #compute a probability for accpeting new solution
        alpha = min(1, np.exp((old_energy - new_energy) / temp))

        #MH sampling
        if np.random.uniform() < alpha: 
            #update everything if new solution accepted
            accepted += 1
            accumulator.append([new_solution, new_energy])

            old_energy = new_energy
            old_solution = new_solution

        else:
            # Keep the old stuff if new solution not accepted
            accumulator.append([old_solution, old_energy])

    return np.array(accumulator), accepted * 1. / total


accumulator, ratio = parallel_tempering(energy, proposal, init_params, epochs=num_epochs, temp=rank+0.1)
print(f"acceptance for temp={rank+0.1}: {ratio*100:.2f}%")


comm.barrier()

if rank==0:
    endtime = time.time() - starttime
    print(endtime)