import torch
import torch.distributed.rpc as rpc
import os
import torch.nn as nn
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
print("good")
rpc.init_rpc("worker0", rank=0, world_size=2)
linear1 = nn.Linear(10,100)
fut1 = rpc.rpc_async("worker1", linear1, args=(torch.ones(10),))
linear2 = nn.Linear(100,10)
fut2 = linear2(fut1.wait())

result = fut2
print(result)
rpc.shutdown()