import logging
import sys
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp


# 打印日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - Line:(%(lineno)d) - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)

def print_rank0(msg, *args, **kwargs):
    rank = dist.get_rank()
    kwargs.setdefault('stacklevel', 2)
    if rank == 0:
        logging.info(msg, *args, **kwargs)

# 测试dist.scatter
"""
测试结果：
Scatter input tensor:
tensor([[0., 1.],
        [2., 3.]], device='cuda:0')
Rank:0 Scatter output tensor:
tensor([0., 1.], device='cuda:0')
Rank:1 Scatter output tensor:
tensor([2., 3.], device='cuda:1')
"""
def test_scatter():
    # 进程同步
    dist.barrier()
    # 获取rank和world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 创建接收tensor
    # 大小为world_size的一维张量
    output_tensor = torch.zeros(world_size, dtype=torch.float32)
    # 只有进程0创建输入tensor
    if rank == 0:
        # 先创建一个world_size*world_size的一维张量，然后再reshape成world_size行world_size列的二位张量
        input_tensor = torch.arange(world_size * world_size, dtype=torch.float32).reshape(world_size, world_size)
        # 打印输入tensor
        print_rank0(f"Scatter input tensor:\n{input_tensor}")
        # 注意，scatter要求的输入是一个List[Tensor]，而不是一个Tensor
        scatter_list = list(input_tensor)
    else:
        # 注意，其余进程也要创建scatter_list变量，但可以赋值为None
        scatter_list = None
    
    # 执行分发
    # 根节点为0,将input_tensor按行分发到各个进程的output_tensor中
    dist.scatter(output_tensor, scatter_list, src=0)
    dist.barrier()
    logging.info(f"Rank:{rank} Scatter output tensor:\n{output_tensor}")

def test_gather():
    dist.barrier()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 创建发送tensor
    # 发送tensor为world_size大小的一维张量
    send_tensor = torch.arange(world_size, dtype=torch.float32)
    # 创建接收tensor
    if rank == 0:
        # 等价于创建一个空的list，然后循环world_size次，每次创建一个张量并且将其append到list中
        recv_tensor = [torch.zeros(world_size, dtype=torch.float32) for _ in range(world_size)]
    else:
        recv_tensor = None
    # 执行收集
    dist.gather(send_tensor, recv_tensor, dst=0)
    dist.barrier()
    logging.info(f"rank: {rank}, send_tensor: {send_tensor}")
    print_rank0(f"Gather recv_tensor: {recv_tensor}")






def main():
    print(torch.cuda.nccl.version())
    print(torch.cuda.device_count())

    # Rank:进程的编号，从0开始，一般情况下是一张显卡对应一个进程
    # world_size:进程总数  processGroup:进程组，多个进程可以组成一个进程组，进程组内的进程可以相互通信
    # Tensor：张量，可以理解为多维数组，pytorch中的核心数据结构

    # 1. 初始化通信进程组
    dist.init_process_group("nccl")
    # 2. 获取当前进程的编号
    # 使用dist.get)rank()可以获取当前的进程编号
    rank = dist.get_rank() 
    # 3. 将当前进程和某张显卡进行绑定
    # torch.cuda.device_count()可以获取当前机器的显卡数
    # 一般一个显卡对应一个进程，进程号就是显卡号
    local_rank = rank % torch.cuda.device_count()
    # torch.cuda.set_default_device()
    # torch.cuda.set_default_device("cuda")设置默认设备为GPU
    # torch.cuda.set_default_device("cuda:<local_rank>")设置默认设备为指定编号的GPU
    # torch.cuda.set_default_device("cpu")设置默认设备为CPU
    torch.set_default_device(f"cuda:{local_rank}")

    """
    torch.tensor()等创建的tensor默认是创建到CPU上
    如果想要构建到GPU上,需要进行搬运，如：
    device = torch.device("cuda:0")
    a = torch.tensor([1,2,3])
    a = a.to(device)
    这里创建的a默认创建在cpu上，device = torch.device("cuda:0")表示GPU0，a = a.to(device)将a搬运到GPU0上

    torch.cuda.set_default_device()可以指定默认的设备，这样对应的rank的tensor就会创建在指定的device上
    """
    # 4. 测试scatter通信原语
    print_rank0("Testing dist.scatter ...")
    test_scatter()


    
if __name__ == "__main__":
    main()