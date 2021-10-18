import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


learning_rate = 0.01
momentum = 0.5


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128*3*3, 625)
        self.fc2 = nn.Linear(625, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128*3*3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*3*3, 625)
        self.fc2 = nn.Linear(625, 10)

    def forward(self, x):
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128*3*3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net1 = Net1()
net2 = Net2()
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=learning_rate, momentum=momentum)
optimizer2 = optim.SGD(net2.parameters(), lr=learning_rate, momentum=momentum)


def run0(rank, size, net, optimizer,criterion):
    """ Distributed function to be implemented later. """
    for i in range(20):
        print("round", i, "in run", rank,"begin")
        x0 = torch.rand([1, 1, 28, 28])
        x1 = net(x0)
        x1 = torch.tensor(x1)
        print("round", i, "in run", rank,"run0", x1.shape)
        dist.send(x1, 1)
        print("send",i)


def run1(rank, size, net, optimizer,criterion):
    for i in range(20):
        print("round", i, "in run", rank,"begin")
        x2 = torch.rand([1, 64, 7, 7])
        dist.recv(x2, 0)
        print("receive",i)
        result = net(x2)
        print("round", i, "in run", rank,"run1",result.shape)
        label = torch.rand([10])
        #loss = criterion(result,label)
        #print(loss)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()


def init_process(rank, size, fn, net, optimizer,criterion, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5676'
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("rank", rank)
    print("__________")
    fn(rank, size, net,optimizer,criterion)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        if rank == 0:
            p = mp.Process(target=init_process, args=(
                rank, size, run0, net1, optimizer1,None))
        elif rank == 1:
            p = mp.Process(target=init_process, args=(
                rank, size, run1, net2, optimizer2,criterion))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
