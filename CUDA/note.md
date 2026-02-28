## 第一章

* GPU:图形处理器，有更多的计算核心，适合于数据计算

### 性能指标

  1. 核心数：核心是用于提供计算能力的单元，核心数越多，并行计算的线程就越多
  2. GPU显存容量：显存是用于临时储存GPU计算时所需的数据，与主机的内存是分开的，直接集成在显卡上，传输的延迟较低。显存越大，传输的延迟就越低
  3. GPU计算峰值：理论的最大计算能力
  4. 显存带宽：运算单元和显存之间的数据交换速率，带宽越大，传输越快

### 架构特点

#### CPU与GPU的特点

  ![image-20260228195202039](https://raw.githubusercontent.com/hhr2449/pictureBed/main/img/image-20260228195202039.png)

  CPU有少数几个快速的计算核心，较多的晶体管用于数据缓存和流程控制

  GPU有很多的不那么快速的计算核心，但用于控制和缓存的部分就比较少

  所以GPU其实是用更多的计算核心来获取更高的计算性能

  CPU和GPU都有DRAM，两者通过PCIe总线连接

#### 计算架构

GPU不能单独计算，GPU+CPU组成异构计算架构

CPU起到控制作用，称为主机(Host)

GPU可以看作CPU的协处理器，称为设备(Device)

主机和设备之间内存访问通过PCle总线连接

流程如下：

1. 数据准备：CPU在Host Memory中准备输入数据
2. 数据传输：通过PCIe总线将数据从Host Memory拷贝到Device Mmory中
3. 启动核函数计算：CPU会调用GPU上的核函数，由GPU的计算核心进行计算
4. **结果回传**：计算完成后，将结果从 Device Memory 拷贝回 Host Memory。

### CUDA编译器

cuda使用nvcc进行编译，nvcc支持纯c++代码的编译

使用`nvidia-msi`可以查看显卡等相关信息

  



## 第一个cuda程序与线程结构

### hello.cu

```c++
#include <stdio.h>

// GPU kernel函数，CPU调用，在GPU上执行
// 需要在前面加上__global__修饰，类型只能是void，没有返回值
__global__  void hello_from_gpu() {
    // 核函数中可以调用stdio中的peintf函数，但是不能调用iostream中的函数
    printf("hello world from GPU\n");
}

int main() {
    // 核函数调用的时候需要加上<<<grid_size, block_size>>>
    // grid_size是网格大小，即block数。block_size是每个block中的线程数，需要指定核函数执行时的线程组织方式
    hello_from_gpu<<<2,2>>>();
    // 注意，GPU中的输出函数会先输出到缓冲区中，直到缓冲区满或是调用同步函数才进行输出
    cudaDeviceSynchronize();
    return 0;
}
```

![image-20260228211926047](https://raw.githubusercontent.com/hhr2449/pictureBed/main/img/image-20260228211926047.png)

使用nvcc进行编译，发现输出了四个`hello world from GPU`

因为设定了`<<<2,2>>>`，也就是总共有四个线程，这四个线程会各自执行一次核函数

### 线程结构

CUDA将线程分为三层结构：

| 层级  | 名称            | 特点                                                         | 硬件映射                              |
| ----- | --------------- | ------------------------------------------------------------ | ------------------------------------- |
| 第1层 | Grid（网格）    | 一个核函数启动时创建一个 grid，包含多个 blocks               | 逻辑概念，不直接对应硬件              |
| 第2层 | Block（线程块） | 每个 block 包含多个 threads，block 内线程可协作（共享内存、同步） | 映射到 SM（Streaming Multiprocessor） |
| 第3层 | Thread（线程）  | 最小执行单元，每个 thread 执行核函数的一次实例               | 映射到 CUDA Core / 流处理器           |

GPU中有大量的计算核心，一个线程最多只能使用一个核心，想要发挥GPU的计算能力需要很多的线程

实际上，一般需要数倍于核心数的线程才能完全发挥计算能力

执行核函数的时候需要指定`<<<grid_size, block_size>>>`也就是指定网格中的线程块数量和每个块中的线程数量

一共有`grid_size * block_size`个线程，每个线程都会执行一遍核函数

* grid和block的维度最大为三维



### 内置变量与线程定位

#### 内置变量

CUDA提供了以下内置变量，均为`dim3`类型，有`.x,.y,.z`三个分量

| 变量名      | 含义                            | 说明                               |
| ----------- | ------------------------------- | ---------------------------------- |
| `threadIdx` | 当前线程在其所属 block 中的索引 | 从 `(0,0,0)` 开始                  |
| `blockIdx`  | 当前 block 在整个 grid 中的索引 | 从 `(0,0,0)` 开始                  |
| `blockDim`  | 每个 block 的 线程数量维度      | 编译时已知，由 kernel 启动参数决定 |
| `gridDim`   | 整个 grid 的 block 数量维度     | 同上                               |

#### 线程定位

##### 一维

`int tid = blockIdx.x * blockDim.x + threadIdx`

