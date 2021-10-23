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
```
time cost 99.72529292106628 s
Epoch: 0, Train loss: 1.5572, Train acc: 0.5758 
time cost 100.45221376419067 s
Epoch: 1, Train loss: 0.2332, Train acc: 0.9280 
time cost 101.30737400054932 s
Epoch: 2, Train loss: 0.1274, Train acc: 0.9609 
time cost 100.07063317298889 s
Epoch: 3, Train loss: 0.0948, Train acc: 0.9713 
time cost 100.05869507789612 s
Epoch: 4, Train loss: 0.0782, Train acc: 0.9753 
time cost 100.9523868560791 s
Epoch: 5, Train loss: 0.0675, Train acc: 0.9792 
time cost 100.32389807701111 s
Epoch: 6, Train loss: 0.0596, Train acc: 0.9818 
time cost 101.30834674835205 s
Epoch: 7, Train loss: 0.0540, Train acc: 0.9831 
time cost 99.85591912269592 s
Epoch: 8, Train loss: 0.0491, Train acc: 0.9848 
time cost 101.23964381217957 s
Epoch: 9, Train loss: 0.0448, Train acc: 0.9862

```
## zmq
```
time cost 106.2454879283905 s
Epoch: 0, Train loss: 1.5686, Train acc: 0.5585
time cost 106.19063806533813 s
Epoch: 1, Train loss: 0.2414, Train acc: 0.9258
time cost 106.75395107269287 s
Epoch: 2, Train loss: 0.1345, Train acc: 0.9585
time cost 107.25808787345886 s
Epoch: 3, Train loss: 0.1001, Train acc: 0.9695
time cost 107.05887007713318 s
Epoch: 4, Train loss: 0.0807, Train acc: 0.9754
time cost 106.43184208869934 s
Epoch: 5, Train loss: 0.0690, Train acc: 0.9792
time cost 107.17323017120361 s
Epoch: 6, Train loss: 0.0602, Train acc: 0.9815
time cost 106.48555111885071 s
Epoch: 7, Train loss: 0.0529, Train acc: 0.9842
time cost 106.2051100730896 s
Epoch: 8, Train loss: 0.0486, Train acc: 0.9846
time cost 106.44140076637268 s
Epoch: 9, Train loss: 0.0438, Train acc: 0.9860
```
## dist
```
time cost 100.2899158000946 s
Epoch: 0, Train loss: 1.5495, Train acc: 0.5503
time cost 100.98278594017029 s
Epoch: 1, Train loss: 0.2115, Train acc: 0.9328
time cost 100.67520999908447 s
Epoch: 2, Train loss: 0.1204, Train acc: 0.9590
time cost 101.46299695968628 s
Epoch: 3, Train loss: 0.0909, Train acc: 0.9674
time cost 100.97617888450623 s
Epoch: 4, Train loss: 0.0761, Train acc: 0.9722
time cost 101.1002140045166 s
Epoch: 5, Train loss: 0.0661, Train acc: 0.9757
time cost 101.0083999633789 s
Epoch: 6, Train loss: 0.0588, Train acc: 0.9777
time cost 100.96474599838257 s
Epoch: 7, Train loss: 0.0536, Train acc: 0.9792
time cost 100.96987318992615 s
Epoch: 8, Train loss: 0.0482, Train acc: 0.9811
time cost 101.51815581321716 s
Epoch: 9, Train loss: 0.0447, Train acc: 0.9821
```
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