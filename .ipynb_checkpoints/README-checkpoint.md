# remote-train
Use dist and zmq to implement model-parallel training.
Dataset:mnist
network:
```python
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
            dist.send(x,1)
            dist.recv(x,0)
            x = x.view(-1,128*3*3)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
       
            return x
```
# results
## local
![](./pictures/local.jpg)
## zmq
![](./pictures/zmq.jpg)
## dist
![](./pictures/dist.jpg)
already have fix the error that server did not change its network's parameters.
# methods explaining
zmq and local are the same method.\
When it comes to `dist.send` and `dist.recv`\
There are two problems:
1. these apis does not have a tensor like return, so we must initiate a tensor which is same to the tensor being received
```
        x = torch.rand([128, 10])
        dist.recv(x, 0)
```
Which is uncomfortable, I don't think there are any other apis in torch.distributed.
2. Both apis can not send and receive gpu.tensors!!! also can only send and receive cpu.tensors.\
Dictionary and other are not available!!!