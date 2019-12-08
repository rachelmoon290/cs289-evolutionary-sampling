import numpy as np
from scipy.stats import norm
from mpi4py import MPI
import time
import sys

np.random.seed(289)


"""Initialize target distribution"""


# target distribution: normal case
mu0 = 1
target_sigma0 = 1
f = lambda x: norm(mu0, target_sigma0).pdf(x)

# target distribution: bimodal case
# mu0, mu1 = 1, 10
# target_sigma0, target_sigma1 = 1,2
# f = lambda x: 0.5*norm(mu0, target_sigma0).pdf(x) + 0.5*norm(mu1, target_sigma1).pdf(x)


# target distribution: difficult normal case
# mu0, mu1, mu2, mu3 = -10,0,10,30
# sig0, sig1, sig2, sig3 = 1, 2, 2, 1
# f = lambda x: 0.25*norm(mu0, sig0).pdf(x) + 0.25*norm(mu1, sig1).pdf(x) + 0.25*norm(mu2, sig2).pdf(x) + 0.25*norm(mu3, sig3).pdf(x)


def energy(x):
    if f(x) < 1e-20:
        return -np.log(1e-10)
    else:
        return -np.log(f(x))

"""Proposal distribution"""
proposal_sigma = 1
proposal = lambda x: np.random.normal(x, proposal_sigma)


def metropolis_hastings(old_solution, new_solution, old_energy, new_energy, temp):
    """
    Metropolis-Hastings accept-reject framework
    """

    #compute a probability for accepting new solution
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

def single_chain_sampling(energy, proposal, init_sol, epochs, temp):
    """
    Implementation of single chain sampling with exchange using a metropolis-hastings accept-reject framework
    """
    accumulator = []

    old_solution = init_sol[rank]
    old_energy = energy(old_solution)

    total_accepted=0

    for epoch in range(epochs):

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

    return np.array(accumulator), float(total_accepted/epochs)



if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    num_chains = comm.Get_size()


    # initial solutions (samples)    
    init_sol = np.random.randint(-10,10,size=num_chains)

    # choose rank to report time elapsed
    chosen_rank = 0
    if rank == chosen_rank:
        starttime = time.time()


    # get cmd argument inputs
    try:
        num_epochs = int(sys.argv[1])
        assert num_epochs > 0

    except:
        print('Invalid number of epochs, should be non-zero positive integer.')
        num_epochs = 10

    data_folder = sys.argv[2]

    # intialize temperature
    # temp_list = [1,1.2,1.4,1.6,1.8,2,2.2,]
    # temp = temp_list[rank]


    temp_list = [1, 1.5, 3, 4.5, 6, 7.5]
    temp = temp_list[rank]
    
    exchange_rate = 100
    
    # if rank==0: 
    #     temp = 1
    # else:
    #     temp = rank * 1.5
    
    accumulator, ratio_accept = single_chain_sampling(energy, proposal, init_sol, epochs=num_epochs, temp=temp)
    print(f'Agent {rank} [T ={temp}] accepted: {ratio_accept*100:.2f}%')

    # wait until everyone is finished
    comm.barrier()

    if rank == chosen_rank:
        endtime = time.time() - starttime
        print(f'Total time elapsed: {endtime} seconds')

    # write results
    np.save(f'results/{data_folder}/single_chain/process_{rank}.npy', accumulator)
