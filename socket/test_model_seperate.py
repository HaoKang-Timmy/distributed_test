import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
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
          
    def forward(self,x,rank):
        if rank == 0:
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            dist.send(x,1)
        elif rank == 1:
            dist.recv(x,0)
            x = x.view(-1,128*3*3)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
       
        return x
