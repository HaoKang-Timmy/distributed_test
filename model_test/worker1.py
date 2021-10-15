import torch.distributed.rpc as rpc
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
rpc.init_rpc("worker1", rank=1, world_size=2)
rpc.shutdown()