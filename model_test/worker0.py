import torch
import torch.distributed.rpc as rpc
import os
from torch import nn, Tensor
from torch.distributed.nn.api.remote_module import RemoteModule
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
print("good")
rpc.init_rpc("worker0", rank=0, world_size=2)
remote_linear_module = RemoteModule(
    "worker1/cpu", nn.Linear, args=(20, 30),
)
input = torch.randn(128, 20)
ret_fut = remote_linear_module.forward_async(input)
ret = ret_fut.wait()
print(ret.shape)
rpc.shutdown()