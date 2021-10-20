
import time
import zmq
import io
import torch.optim as optim
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from buffer import serialize, de_serialize
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")



class NetServer(nn.Module):
    def __init__(self):
        super(NetServer, self).__init__()

        self.fc1 = nn.Linear(128*3*3, 625)
        self.fc2 = nn.Linear(625, 10)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

net_server = NetServer()
train_batch_size = 128
test_batch_size = 64
learning_rate = 0.01
momentum = 0.5
num_epoches = 10
optimizer = optim.SGD(net_server.parameters(), lr=learning_rate, momentum=momentum)

print("Initialization finished")
while True:
    #  Wait for next request from client
    message = socket.recv()
    info = th.load(io.BytesIO(message))
    if info['flag'] == 0:
        input_data = info['data']
        print("[begin] Received request: ", input_data.shape)
        out = net_server(input_data)
        socket.send(serialize(out))
        print("[begin] Send output: ",  out.shape)
    elif info['flag'] == 1:
        grad_outputs = info['data']
        print("[end] Received grad_output: ", grad_outputs.shape)
        optimizer.zero_grad()
        grad_i = th.autograd.grad(out, input_data, grad_outputs=grad_outputs)[0]
        optimizer.step()
        print("[end] Send dy/dx: ", grad_i.shape)
        socket.send(serialize(grad_i))
    elif info['flag'] == -1:
        socket.send(b'finish')
        break
    time.sleep(1)
