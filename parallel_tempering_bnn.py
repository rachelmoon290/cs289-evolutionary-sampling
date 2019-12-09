import numpy as np
from scipy.stats import norm
from mpi4py import MPI
import time
import sys
from neural_network import Feedforward
import pandas as pd
from autograd import numpy as grad_np

np.random.seed(289)

def log_joint(W):
    X, Y, W = x.reshape(1,-1), y.reshape(1,-1), W.reshape(1,-1)
    N = Y.shape[0]
    pred = nn.forward(W,X).reshape(1,-1)
    p_w = -(1/2.)*np.log(2*np.pi) -1*np.log(5) -(0.02)*np.dot(W, W.T)
    cond = -(N/2)*np.log(0.5) -2*np.sum((Y-pred)**2)
    return (p_w + cond)[0,0]

energy = lambda w: -log_joint(w)

# Proposal distribution: normal distribution
proposal = lambda p: np.random.normal(p.flatten(), 0.001*temp*np.ones(shape=nn.D), size=nn.D).reshape(1,-1)

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

def exchange(old_solution, energy, old_energy, temp):
    """
    Implementation of exchange
    """
    exchanged = 0
    if num_chains == 1:
        return exchanged, old_solution, old_energy

    if rank > 0:
        comm.send(old_solution, dest=rank-1, tag=13)
    if rank < num_chains -1:
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

    total_exchanged = 0
    total_accepted = 0
    total_exchange_attempts = 0 if epochs > 1000 else 1

    for epoch in range(epochs):
        if epoch % 5 == 0 and epoch > 0:
            comm.barrier()
            exchanged, old_solution, old_energy = exchange(np.array(old_solution), energy, old_energy, temp)
            total_exchanged += exchanged
            total_exchange_attempts += 1

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

    return np.array(accumulator), float(total_accepted/epochs), float(total_exchanged/total_exchange_attempts)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    num_chains = comm.Get_size()

    def gramacy_lee_1d(x):
        return np.sin(10*np.pi*x)/(2*x) + (x-1)**4

    xmin = -0.5
    xmax = 2.5
    x = np.linspace(-0.5,2.5,10000)
    y = gramacy_lee_1d(x)

    # define rbf activation function
    alpha = 1
    c = 0
    h = lambda x: grad_np.exp(-alpha * (x - c)**2)

    # neural network model design choices
    width = 15
    hidden_layers = 2
    input_dim = 1
    output_dim = 1

    architecture = {'width': width,
                   'hidden_layers': hidden_layers,
                   'input_dim': input_dim,
                   'output_dim': output_dim,
                   'activation_fn_type': 'rbf',
                   'activation_fn_params': 'c=0, alpha=1',
                   'activation_fn': h}

    # set random state to make the experiments replicable
    rand_state = 0
    random = np.random.RandomState(rand_state)

    # instantiate a Feedforward neural network object
    nn = Feedforward(architecture, random=random)

    # define design choices in gradient descent
    params = {'step_size':1e-3,
              'max_iteration':4000,
              'random_restarts':1}

    # fit my neural network to minimize MSE on the given data
    nn.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)

    # initial solution
    init_sol = np.random.normal(size=nn.D)

    chosen_rank = 0

    if rank == chosen_rank:
        starttime = time.time()

    try:
        num_epochs = int(sys.argv[1])
        assert num_epochs > 0

    except:
        print('err')
        num_epochs = 10

    # intialize temperature
    if rank==0:
        temp = 1
    else:
        temp = 1.5*rank

    accumulator, ratio_accept, ratio_exchange = parallel_tempering(energy, proposal, init_sol, epochs=num_epochs, temp=temp)
    print(f'acceptance for temp={temp}: {ratio_accept*100:.2f}%, exchange: {ratio_exchange*100:.2f}%')

    comm.barrier()

    if rank == chosen_rank:
        endtime = time.time() - starttime
        print(f'Total time: {endtime}')

        values = accumulator[:,0]
        energy = accumulator[:,1]
        final_samples = np.array([val[0] for val in values])
        n = 100
        num_pts = 1000
        all_ys = np.zeros(shape=(n, num_pts))

        for i in range(n):
            ind = np.random.randint(0, final_samples.shape[0])
            sample = final_samples[ind]
            x_test = np.linspace(xmin, xmax, num_pts)
            noise = np.random.normal(scale=0.5, size=num_pts)

            y_test = nn.forward(sample.reshape(1,-1), x_test.reshape((1, -1))).flatten() + noise
            all_ys[i] = y_test

        # saving this file
        np.save(f'results/posterior_predictive.npy', all_ys)

    # writing files
    np.save(f'results/process_{rank}.npy', accumulator)
