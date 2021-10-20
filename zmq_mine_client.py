
import time
import zmq
import io
import torch.optim as optim
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader
import torch as th
from buffer import serialize, de_serialize
import torchvision.transforms as transforms
from torchvision.datasets import mnist

print("Connecting to hello world server…")
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

class NetClient(nn.Module):
    def __init__(self):
        super(NetClient, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128*3*3)

        return x
net_client = NetClient()
class RemotePassBegin(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        buffer = io.BytesIO()
        th.save({
            "flag": 0, # forward
            "data": input
        }, buffer)
        
        socket.send(buffer.getvalue())
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # return grad_output
        return de_serialize(socket.recv())

class RemotePassEnd(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.input = input
        return de_serialize(socket.recv())

    @staticmethod
    def backward(ctx, grad_output):
        buffer = io.BytesIO()
        th.save({
            "flag": 1, # backward
            "data": grad_output
        }, buffer)
        
        socket.send(buffer.getvalue())
        return None

train_batch_size = 128
test_batch_size = 64
learning_rate = 0.01
momentum = 0.5
num_epoches = 10
# create two datasets
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = mnist.MNIST('./data', train=True, transform=transforms, download=True)
test_dataset = mnist.MNIST('./data', train=False, transform=transforms)
# iterate
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


optimizer = optim.SGD(net_client.parameters(), lr=learning_rate, momentum=momentum)
b = th.rand([128,1,28,28],requires_grad=True)
x = net_client(b)
x = RemotePassBegin.apply(x)
x = RemotePassEnd.apply(x)
label = torch.rand([128]).type(torch.LongTensor)
criterion = nn.CrossEntropyLoss()
loss = criterion(x,label)
optimizer.zero_grad()
loss.backward()
optimizer.step()
train_losses = []
train_accs = []
def train():
    for epoch in range(num_epoches):
        train_loss = 0
        train_acc = 0
        for img, label in train_loader:
            img = img 
            label = label
            x = net_client(img)
            x = RemotePassBegin.apply(x)
            x = RemotePassEnd.apply(x)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(x,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #     # 前向传播
        #     out = model(img)
        #     loss = criterion(out, label)
        #     print(out.shape,label.shape)
        #     print(loss)
        #     # 反向传播
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

            # 计算损失
            train_loss += loss.item()
            # 计算准确率
            pred = x.argmax(dim=1)
            train_acc += (pred == label).sum().item() / img.size(0)
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_accs)   
        # 日志输出
        print("Epoch: {}, Train loss: {:.4f}, Train acc: {:.4f}".format(epoch, train_loss, train_acc))
train()