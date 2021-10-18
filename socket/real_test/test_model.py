from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torch
import torch.optim as optim

# 超参数
train_batch_size = 128
test_batch_size = 64
learning_rate = 0.01
momentum = 0.5
num_epoches = 10

# 数据集加载
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = mnist.MNIST('./data', train=True, transform=transforms, download=True)
test_dataset = mnist.MNIST('./data', train=False, transform=transforms)

# 创建数据集迭代对象，但不是迭代器，可以使用for循环，不能使用next
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# 构建模型
class Net(nn.Module):
    
    def __init__(self, in_dim, n_hidden1, n_hidden2, out_dim):
        super(Net, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden1), nn.BatchNorm1d(n_hidden1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden1, n_hidden2), nn.BatchNorm1d(n_hidden2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden2, out_dim))
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# 实例化模型
model = Net(28 * 28, 300, 100, 10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 训练过程
train_losses = []
train_accs = []

for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    for img, label in train_loader:
        img = img.to(device)    
        label = label.to(device)
        
        img = img.view(img.size(0), -1)
        
        # 前向传播
        out = model(img)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算损失
        train_loss += loss.item()
        # 计算准确率
        pred = out.argmax(dim=1)
        train_acc += (pred == label).sum().item() / img.size(0)
    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)
    train_losses.append(train_loss)
    train_accs.append(train_accs)   
    # 日志输出
    print("Epoch: {}, Train loss: {:.4f}, Train acc: {:.4f}".format(epoch, train_loss, train_acc))