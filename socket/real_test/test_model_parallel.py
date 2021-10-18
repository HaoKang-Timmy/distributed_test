import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding = 1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3,padding = 1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64,128,3,padding = 1)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(128*3*3,625)
        self.fc2 = nn.Linear(625,10)
          
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1,128*3*3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
       
        return x
class Net1(nn.Module):
    def __init__(self):
        super(Net1,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding = 1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3,padding = 1)
        self.pool2 = nn.MaxPool2d(2,2)
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x
class Net2(nn.Module):
    def __init__(self):
        super(Net2,self).__init__()
        self.conv3 = nn.Conv2d(64,128,3,padding = 1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128*3*3,625)
        self.fc2 = nn.Linear(625,10)
    def forward(self,x):
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1,128*3*3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def run0(rank, size):
    """ Distributed function to be implemented later. """
    for i in range(5):
        x0 = torch.rand([1,1,28,28])
        print(x0.shape)
        net1 = Net1()
        x1 = net1(x0)
        x1 = torch.tensor(x1)
        print("run0",x1.shape)
        dist.send(x1,1)
def run1(rank,size):
    for i in range(5):
        print("run1begin")
        x2 = torch.rand([1,64,7,7])
        dist.recv(x2,0)
        print("run1",x2.shape)
        net2 = Net2()
        result = net2(x2)


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("rank",rank)
    print("__________")
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        if rank ==0:
            p = mp.Process(target=init_process, args=(rank, size, run0))
        elif rank == 1:
            p = mp.Process(target=init_process, args=(rank, size, run1))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()