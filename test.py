from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# passing MPI datatypes explicitly
# if rank == 0:
#     data = np.arange(1000, dtype='i')
#     comm.Send([data, MPI.INT], dest=1, tag=77)
# elif rank == 1:
#     data = np.empty(1000, dtype='i')
#     comm.Recv([data, MPI.INT], source=0, tag=77)

# automatic MPI datatype discovery

iterations = 100

if rank == 0:
    data = np.arange(100, dtype=np.float64)

elif rank == 1:
    data = np.empty(100, dtype=np.float64)


for i in range(iterations):
    if rank == 0:
        comm.Send(np.array(data[i]), dest=1, tag=13)

    elif rank == 1:
        newData = np.zeros(1)
        comm.Recv(newData, source=0, tag=13)

        data[i] = newData

if rank == 1:
    print(data)


#print(rank)
#print(data)
print("I am finished my job")