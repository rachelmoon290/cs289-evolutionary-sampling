import numpy as np
from scipy.stats import norm
from mpi4py import MPI
import time
import sys

# set parameters for target/proposal distribution
mu0 = 1
mu1 = 50

target_sigma0 = 1
target_sigma1 = 2

init_sol = 10
proposal_sigma = 1

# initialization of functions

# target distribution: normal case
f = lambda x: norm(mu0, target_sigma0).pdf(x)
# energy = lambda x: -np.log(f(x))

def energy(x):
    if f(x) < 1e-20:
        return -1e10
    else:
        return -np.log(f(x))

# target distribution: bimodal case
# f = lambda x: norm(mu0, target_sigma0).pdf(x) + norm(mu1, target_sigma1).pdf(x)
# energy = lambda x: -np.log(f(x))

# Proposal distribution: normal distribution
proposal = lambda x: np.random.normal(x, proposal_sigma)

def metropolis_hastings(old_solution, new_solution, old_energy, new_energy, temp):
    """
    Metropolis-Hastings accept-reject framework
    """
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
    """
    Implementation of exchange
    """
    exchanged = 0
    if rank > 0:
        comm.send(old_solution, dest=rank-1, tag=13)
    if rank < 3:
        new_solution = comm.recv(source=rank+1, tag=13)
        new_energy = energy(new_solution)
        exchanged, old_solution, old_energy = metropolis_hastings(old_solution, new_solution, old_energy, new_energy, temp)

    return exchanged, old_solution, old_energy

def parallel_tempering(energy, proposal, init_sol, epochs, temp):
    """
    Implementation of parallel tempering with exchange using a metropolis-hastings accept-reject framework
    """
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

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    chosen_rank = 0

    if rank == chosen_rank:
        starttime = time.time()

    try:
        num_epochs = int(sys.argv[1])
        assert num_epochs > 0

    except:
        print('err')
        num_epochs = 10

    temp = 2*rank + 0.1
    accumulator, ratio_accept, ratio_exchange = parallel_tempering(energy, proposal, init_sol, epochs=num_epochs, temp=temp)
    print(f'acceptance for temp={temp}: {ratio_accept*100:.2f}%, exchange: {ratio_exchange*100:.2f}%')

    comm.barrier()

    if rank == chosen_rank:
        endtime = time.time() - starttime
        print(f'Total time: {endtime}')

    # writing files
    np.save(f'results/process_{rank}.npy', accumulator)
