import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

plot = True
chosen_rank = 0

if rank==chosen_rank:
    starttime = time.time()

# set parameters for target/proposal distribution
mu0 = 1
mu1 = 50

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

init_sol = 10
num_epochs=50000

def metropolis_hastings(old_solution, new_solution, old_energy, new_energy, temp):
    #compute a probability for accpeting new solution
    alpha = min(1, np.exp((old_energy - new_energy) / temp))

    #MH sampling
    if np.random.uniform() < alpha:
        #update everything if new solution accepted
        accepted = 1
        return accepted, new_solution, new_energy

    else:
        # Keep the old stuff if new solution not accepted
        accepted = 0
        return accepted, old_solution, old_energy

def exchange(old_solution, energy, old_energy, temp):
    # new_solution = np.array([0])
    exchanged = 0
    if rank > 0:
        comm.send(old_solution, dest=rank-1, tag=13)
    if rank < 3:
        new_solution = comm.recv(source=rank+1, tag=13)
        new_energy = energy(new_solution)
        exchanged, old_solution, old_energy = metropolis_hastings(old_solution, new_solution, old_energy, new_energy, temp)

    return exchanged, old_solution, old_energy

def parallel_tempering(energy, proposal, init_sol, epochs, temp):
    accumulator = []

    old_solution = init_sol
    old_energy = energy(old_solution)

    total_exchanged=0
    total_accepted=0
    exchanged_interval = int(epochs/1)

    for epoch in range(epochs):
        if epoch % exchanged_interval == 0 and epoch > 0:
            comm.barrier()
            exchanged, old_solution, old_energy = exchange(np.array(old_solution), energy, old_energy, temp)
            total_exchanged += exchanged

        #propose new solution based on current solution
        new_solution = proposal(old_solution)

        #compute energy of new solution
        new_energy = energy(new_solution)

        # metropolis-hastings step
        accepted, mh_solution, mh_energy = metropolis_hastings(old_solution, new_solution, old_energy, new_energy, temp)
        total_accepted += accepted
        accumulator.append([mh_solution, mh_energy])
        old_energy = mh_energy
        old_solution = mh_solution

    return np.array(accumulator), float(total_accepted/epochs), float(total_exchanged/(epochs/exchanged_interval))

temp = 2*rank + 0.1
accumulator, ratio_accept, ratio_exchange = parallel_tempering(energy, proposal, init_sol, epochs=num_epochs, temp=temp)
print(f"acceptance for temp={temp}: {ratio_accept*100:.2f}%, exchange: {ratio_exchange*100:.2f}%")

comm.barrier()

if rank==chosen_rank:
    endtime = time.time() - starttime
    print(f'Total time: {endtime}')

    if plot == True:
        fig, ax = plt.subplots(ncols=3, figsize=(20,5))

        ax[0].plot(range(num_epochs), accumulator[:,0])
        ax[0].set_title("traceplot of samples")
        ax[0].set_ylabel("sample value")
        ax[0].set_xlabel("number of iterations")

        ax[1].plot(range(num_epochs), accumulator[:,1])
        ax[1].set_title("traceplot of energy function")
        ax[1].set_ylabel("energy function value")
        ax[1].set_xlabel("number of iterations")

        ax[2].hist(accumulator[:,0], bins=30,density=True)
        xgrid = np.linspace(np.min(accumulator[:,0]),np.max(accumulator[:,0]),200)
        ax[2].plot(xgrid, f(xgrid), label="f(x)")
        ax[2].set_title("Samples histogram")
        ax[2].set_ylabel("normalized frequency")
        ax[2].set_xlabel("sample value")

        ax[2].legend()
        plt.show()
