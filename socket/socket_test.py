import os
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import recv
import torch.multiprocessing as mp

def run0(rank, size):
    """ Distributed function to be implemented later. """
    rank = torch.tensor(rank)
    for i in range(10):
        i = torch.tensor(i)
        dist.send(i,1)
        print("send", rank)

    pass
def run1(rank,size):
    x = torch.tensor(0)
    for i in range(10):
        dist.recv(x,0)
        print("recv",x)

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29530'
    print("init")
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        if rank == 0:
            p = mp.Process(target=init_process, args=(rank, size, run0))
        else:
            p = mp.Process(target=init_process, args=(rank, size, run1))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()