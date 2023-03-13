import torch
import torch.multiprocessing as mp
import os
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank,world_size):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    dist.init_process_group("gloo",rank=rank,world_size=world_size)

class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1=nn.Linear(10,10)
        self.relu=nn.ReLU()
        self.net2=nn.Linear(10,5)

    def forward(self,x):
        return self.net2(self.relu(self.net1(x)))

def clean_up():
    dist.destroy_process_group()


def basic(rank,world_size):

    print(f'running basic ddp in rank:{rank}')
    setup(rank,world_size)
    model=ToyModel().to(rank)

    ddp_model=DDP(model,device_ids=[rank])

    loss_fn=nn.MSELoss()
    optimizer=optim.SGD(ddp_model.parameters(),lr=0.001)
    dist.barrier()

    optimizer.zero_grad()
    ouput=ddp_model(torch.rand(20,10))
    labels=torch.rand(20,5).to(rank)
    loss_fn(ouput,labels).backward()
    optimizer.step()

    dist.barrier()

    clean_up()

def run_demo(demo_fn,world_size):

    #  mp.spawn enable share memory
    mp.spawn(demo_fn,
            args=(world_size,),
            nprocs=world_size,
            join=True)

if __name__ == "__main__":
    run_demo(basic,4)