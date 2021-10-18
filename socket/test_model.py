import torch
import os
from torch import nn
import torchvision
 
class Model_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,10,3,1,1)
        self.conv2 = nn.Conv2d(10,20,3,2,1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
 
class Model_B(nn.Module):
    def __init__(self,num_class=5):
        super().__init__()
        self.conv1 = nn.Conv2d(20,40,3,2,1)
        self.conv2 = nn.Conv2d(40,10,3,2,1)
        self.adpool = nn.AdaptiveAvgPool2d([1,1])
        self.linear = nn.Linear(10,num_class)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adpool(x) # n*c*1*1
        x = self.linear(x.view(x.size(0),-1)) #需要reshape
        return x
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = Model_A().to('cuda:0')
        self.model_b = Model_B().to('cuda:1')
    def forward(self, x):
        x = self.model_a(x.to('cuda:0'))
        x = self.model_b(x.to('cuda:1'))
        return x
 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
 
softmax_func = nn.CrossEntropyLoss() # 采用cpu计算  也可以采用 .cuda(num)
 
batch = 4
num_class = 5
inputs = torch.rand([batch,3,224,224])  # cpu
labels = torch.randint(0,num_class,[batch,])    # cpu
model = Model()
model.train()
 
optimizer=torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.99),weight_decay=0.0005)
 
for i in range(1000):
    optimizer.zero_grad()
 
    inputs = inputs.cuda(0)
    labels = labels.cuda(1)
 
    out = model(inputs)
 
    loss = softmax_func(out, labels)
    print('loss: %.4f'%loss.item())
    loss.backward()
    optimizer.step()
