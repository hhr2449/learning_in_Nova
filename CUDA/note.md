## 第一章：简介

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

思路：

![554636c6c2961694f02207965e3e642e](https://raw.githubusercontent.com/hhr2449/pictureBed/main/img/554636c6c2961694f02207965e3e642e.jpg)

##### 一维

`int blockId = blockIdx.x;`

`int tid = blockIdx.x * blockDim.x + threadIdx`

##### 二维

```cpp
// 1. 计算当前 Block 在 Grid 中的一维绝对序号
int blockId = blockIdx.x + blockIdx.y * gridDim.x;

// 2. 计算当前 Thread 的全局唯一 ID
// 公式：前面所有 Block 的 Thread 总数 + 自身在当前 Block 内的二维展平偏移
int threadId = blockId * (blockDim.x * blockDim.y) 
             + threadIdx.y * blockDim.x 
             + threadIdx.x;
```

##### 三维

```cpp
// 1. 计算当前 Block 在 Grid 中的一维绝对序号
int blockId = blockIdx.x 
            + blockIdx.y * gridDim.x 
            + blockIdx.z * gridDim.x * gridDim.y;

// 2. 计算当前 Thread 的全局唯一 ID
// 公式：前面所有 Block 的 Thread 总数 + 自身在当前 Block 内的三维展平偏移
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) 
             + threadIdx.z * (blockDim.x * blockDim.y) 
             + threadIdx.y * blockDim.x 
             + threadIdx.x;
```

#### 编译简介

使用nvcc来编译cuda代码

##### 代码分离

cuda代码中既有运行在CPU上的Host Code，也有运行在GPU上的Device Code，nvcc第一步会进行代码分离

- Host Code：所有在CPU上运行的普通代码分配给系统自带的C++编译器进行编译
- Device Code：所有带有 `__global__` 或 `__device__` 标签的核函数。这些代码nvcc会自己处理

##### 两次翻译

为了提高兼容性，nvcc会进行两步翻译

![img](https://raw.githubusercontent.com/hhr2449/pictureBed/main/img/v2-094a1f867d97ca11317812f5d3fcf82a_r.jpg)

##### 第一步：编译成ptx

ptx是独立于硬件架构的伪汇编语言，只描述逻辑上的操作

nvcc会基于Virtual Architecture将cuda代码翻译为ptx

使用`-arch=compute_XY`指定虚拟架构

##### 第二部：编译为可运行的cubin

Cubin是只能在特定GPU上运行的二进制机器码

nvcc会基于Real Architecture将ptx编译为真实可运行的机器码

使用`-code=sm_ZW`指定真实架构



`-gencode arch=compute_XY,code=sm_XY`生成可运行的机器码，不保留ptx

`-gencode arch=compute_XY,code=compute_XY`只生成ptx，不生成机器码



## 第二章：简单cuda程序的基本架构



### 例子：数组相加

```cpp
// 使用 C++ 和 CUDA 编写一个程序，在 GPU 上实现两个超大型一维双精度浮点数（double）向量的加法运算：$Z = X + Y$。

// 【参数设定】
// 向量长度 ($N$)：$300000005$（3亿零5个元素）。
// 输入向量 $X$：所有元素初始化为常数 $1.23$。
// 输入向量 $Y$：所有元素初始化为常数 $2.34$。
// 精度校验阈值 (Epsilon)：$1.0 \times 10^{-15}$。


// 1. 分配内存：使用malloc()在主机中分配内存，使用cudaMalloc()在设备中分配内存
// 2. 初始化向量：在主机中初始化向量x,y
// 3. 拷贝数据：使用cudaMemcpy()将向量x,y从主机拷贝到设备
// 4. 核函数：计算线程id，获取总线程数，然后跨步执行
// 5. 拷贝数据：使用cudaMemcpy()将向量z从设备拷贝到主机
// 6. 验证结果：在主机中验证结果是否正确
// 7. 释放内存：使用free()释放内存，使用cudaFree()释放设备内存

#include <stdio.h>
#include <math.h>
// 数据总数
const size_t N = 300000005;
// 浮点数判等阈值
const double EPSILON = 1.0e-15;
// 输入，输出向量的值
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
// 声明核函数
__global__ void add(double *x, double *y, double *z, size_t n);
void check_result(double *z, size_t n);

int main() {
    // 1. 分配内存
    // 主机中内存的指针
    double *h_x, *h_y, *h_z;
    // 计算字节数
    size_t M = N * sizeof(double);
    // 使用malloc()分配内存
    h_x = (double *)malloc(M);
    h_y = (double *)malloc(M);
    h_z = (double *)malloc(M);

    // 设备中内存的指针
    double *d_x, *d_y, *d_z;
    // 使用cudaMalloc()分配内存
    // cudaMalloc(void **devPtr, size_t size)，注意第一个参数是指向 设备中内存的指针 的指针，这里可以传入指针的地址
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);

    // 2. 初始化数据
    for(size_t i = 0; i < N; i++) {
        h_x[i] = a;
        h_y[i] = b;
    }
    printf("Initialization done.\n");
    // 3. 拷贝数据
    // cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)，其中kind指定了数据传输的方向
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);
    printf("Data copy to device done.\n");

    // 4. 启动核函数计算
    // 注意，核函数是异步执行的，CPU调用核函数只是将核函数加入GPU的任务队列中，不会等待它完成
    // 想要进行同步，需要1. 显式同步：cudaDeviceSynchronize() 2. 隐式同步：cudaMemcpy()是阻塞式的
    add<<<128, 256>>>(d_x, d_y, d_z, N);
    // 5. 拷贝结果回主机
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    // 6. 验证结果
    size_t check_size = 1000;
    check_result(h_z, check_size);
    // 7. 释放内存
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

// 核函数
// 一般将数据规模作为参数传入
__global__ void add(double *x, double *y, double *z, size_t n) {
    // 当前线程id
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 线程总数，当线程数量不足时，按照stride为步长跨步执行
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = tid; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}

// 验证结果
void check_result(double *z, size_t n) {
    for(size_t i = 0; i < n; i++) {
        if(fabs(z[i] - c) > EPSILON) {
            printf("Error at index %lu: z[%lu] = %lf, expected %lf\n", i, i, z[i], c);
            return;
        }
    }
    printf("Result is correct for the first %lu elements.\n", n);
}
```

![image-20260302100327603](https://raw.githubusercontent.com/hhr2449/pictureBed/main/img/image-20260302100327603.png)



### cuda程序的基本框架

![image-20260301171931867](https://raw.githubusercontent.com/hhr2449/pictureBed/main/img/image-20260301171931867.png)

1. 分配主机和设备内存
2. 初始化主机中的数据
3. 将数据从主机复制到设备
4. 调用核函数进行计算
5. 将计算结果从设备复制到主机
6. 释放内存



### 相关知识点

#### cudaMalloc

`cudaError_t cudaMalloc(void **devPtr, size_t size)`

注意devPtr是指向 `指向要分配的内存的指针` 的指针，实际使用时应该传入指针的地址

返回值是错误信息，所以要传入void **devPtr，因为返回值用于传递错误信息了

size是分配的字节数



#### cudaMemcpy

`cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)`

kind指定了数据传输的方向

比如：`cudaMemcpyHostToDevice`是将数据从host传到device



#### cudafree

释放显卡上的内存



#### 核函数异步执行与同步初涉

CPU启动核函数实际上只是将核函数放入GPU的任务队列中，放入之后CPU不会等待GPU执行完核函数，而是继续执行

也就是说，核函数的执行是**异步**的

如果核函数同步执行，CPU在等待核函数执行完的时间就是空闲着的，会浪费计算资源



##### 如何进行同步

1. 全局显式同步`cudaDeviceSynchronize()`

   CPU执行此函数之后会被阻塞，直到GPU完成所有计算任务

2. 内存拷贝带来的隐式同步`cudaMemcpy()`

   D2H操作（从Device到Host）一定会等待之前的GPU操作全部完成



#### 核函数的基本框架

1. 获取线程id

   `size_t tid = blockIdx.x * blockDim.x + threadIdx.x;`

2. 获取计算范围

   一般使用网格跨步计算

   `size_t stride = gridDim.x * blockDim.x; for(size_t i = tid; i < N; i += stride) { ... }`

3. 搬运数据

   开始复杂的数学运算前，通常要把所需的数据从缓慢的全局内存（Global Memory）读取到极快的寄存器（Registers）中。在更高级的算法中，这一步还会把整块数据读入**共享内存 (Shared Memory)** 供同 Block 的兄弟线程共享。

4. 执行计算逻辑

5. 将结果放入存放结果的内存中



#### 一些数据传输的相关知识

##### CPU与GPU

CPU和GPU的DRAM通过PCIe总线进行连接，通过PCIe进行数据的传输

###### 二次拷贝

如果使用malloc分配内存，分配的是分页内存，由于虚拟内存机制，此时这块内存的物理地址是不连续的，并且上面的数据随时可能被换到磁盘中

如果传输到一般数据被换出就会导致崩溃，所以需要先分配一块临时的锁页内存，将数据先拷贝到临时内存中，在通过PCIe总线传输到GPU中

1. 加大了延迟
2. 由于需要CPU将数据拷贝到临时内存中，**`cudaMemcpyAsync`**退化为同步操作

###### 使用锁页内存

锁页内存：物理地址连续且固定，不会被换出到磁盘的内存

使用锁页内存可以提高效率，并且**`cudaMemcpyAsync`**是异步的

分配：`cudaMallocHost((void**)&h_pinned, size); `

释放：`cudaFreeHost(h_pinned); `



##### NVLink

想要实现GPU之间的通信，传统的方法会先将数据传入CPU的主内存，再传入对应的GPU

NVLink 是主板上真正的一根根高速排线或桥接器，它直接把多张 GPU 物理连接在了一起，完全绕开了 CPU 和 PCIe 总线。

可以实现同一主板上的GPU通信



##### RDMA

NVLink只能进行同一节点内的GPU通信，对于不同节点需要进行网络传输

传统的方法需要将GPU中的数据传入CPU主内存中，通过网卡发送到另一个节点的CPU主内存，再传入GPU

RDMA则是让网卡绕过CPU，直接从GPU中存取数据

**发送端：** 节点 A 的网卡通过主板上的 PCIe 交换机（PCIe Switch），**直接去读取 GPU 0 的显存**。

**网络传输：** 网卡将数据打成数据包（通常基于 InfiniBand 或 RoCE v2 协议），发往节点 B。

**接收端：** 节点 B 的网卡收到数据后，再次通过本机的 PCIe 交换机，**直接把数据写入 GPU 1 的显存**。



### 效率对比

```cpp
#include <stdio.h>
#include <math.h>
#include <chrono>

const size_t N = 300000005;
const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void sum_gpu(double *x, double *y, double *z, size_t n);
void sum_cpu(double *x, double *y, double *z, size_t n);
void check_result(double *z, size_t n);

int main() {
    // 2. 分配内存
    double *h_x, *h_y, *h_z_cpu, *h_z_gpu;
    size_t bytes = N * sizeof(double);
    h_x = (double *) malloc(bytes);
    h_y = (double *) malloc(bytes);
    h_z_cpu = (double *) malloc(bytes);
    h_z_gpu = (double *) malloc(bytes);
    
    double *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_z, bytes);

    // 2. 初始化数据
    for (size_t i = 0; i < N; i++) {
        h_x[i] = a;
        h_y[i] = b;
    }

    // 3. 拷贝数据到GPU
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    // 4. 运行计算并计时

    // CPU直接计时即可
    auto start = std::chrono::high_resolution_clock::now();
    sum_cpu(h_x, h_y, h_z_cpu, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("CPU time: %f ms\n", duration.count());

    // GPU需要创建事件并且插入
    // 因为GPU的计算是异步的，可以在任务前后各插入一个事件，计算时间差值

    // cudaEvent_t是cuda的事件句柄
    cudaEvent_t start_event, stop_event;
    // 创建事件
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // 插入事件
    cudaEventRecord(start_event);
    sum_gpu<<<256, 256>>>(d_x, d_y, d_z, N);
    cudaEventRecord(stop_event);

    // 同步，等待stop事件结束
    cudaEventSynchronize(stop_event);

    float elapsed_time;
    // 计算时间差
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
    printf("GPU time: %f ms\n", elapsed_time);

    // 5. 拷贝数据到CPU
    cudaMemcpy(h_z_gpu, d_z, bytes, cudaMemcpyDeviceToHost);
    check_result(h_z_gpu, N);
    check_result(h_z_cpu, N);
    // 6. 销毁
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    free(h_x);
    free(h_y);
    free(h_z_cpu);
    free(h_z_gpu);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    return 0;

    
}

// GPU计算
__global__ void sum_gpu(double *x, double *y, double *z, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}

// CPU计算
void sum_cpu(double *x, double *y, double *z, size_t n) {
    for (size_t i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}

// 验证结果
void check_result(double *z, size_t n) {
    for(size_t i = 0; i < n; i++) {
        if(fabs(z[i] - c) > EPSILON) {
            printf("Error at index %lu: z[%lu] = %lf, expected %lf\n", i, i, z[i], c);
            return;
        }
    }
    printf("Result is correct for the first %lu elements.\n", n);
}
```

运行结果：

![image-20260302100228401](https://raw.githubusercontent.com/hhr2449/pictureBed/main/img/image-20260302100228401.png)

### 函数类型

| **限定符**                | **执行位置** | **调用位置** | **关键限制**                                               | 调用方式         | 异步性                             |
| ------------------------- | ------------ | ------------ | ---------------------------------------------------------- | ---------------- | ---------------------------------- |
| **`__global__`**          | GPU (Device) | CPU (Host)   | 必须返回 `void`                                            | 需要`<<<>>>`语法 | 异步执行                           |
| **`__device__`**          | GPU (Device) | GPU (Device) | 只能在设备端被调用                                         | 直接进行调用     | 同步执行（属于调用者线程的一部分） |
| **`__host__`**            | CPU (Host)   | CPU (Host)   | 普通 C++ 函数，无法在 GPU 运行                             |                  |                                    |
| **`__host__ __device__`** | 两者皆可     | 两者皆可     | 不能包含平台特有的代码（如 `printf` 在某些旧架构下的差异） |                  |                                    |

- 核函数由CPU调用，GPU执行，返回值只能是void
- 设备函数只能被核函数或设备函数调用，返回值可以是任何类型



```cpp
// 主机+设备函数
__host__ __device__ double cal(double a, double b) {
    return a * a + b * b;
}

// GPU计算
__global__ void sum_gpu(double *x, double *y, double *z, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < n; i += stride) {
        z[i] = cal(x[i], y[i]);
    }
}

// CPU计算
void sum_cpu(double *x, double *y, double *z, size_t n) {
    for (size_t i = 0; i < n; i++) {
        z[i] = cal(x[i], y[i]);
    }
}
```





## CUDA程序的错误检测



### 错误处理宏

```cpp
#pragma once
#include <stdio.h>

#define CHECK(call)                                                       \
do                                                                        \
{\
    const cudaError_t error_code = call;\
    if (error_code != cudaSuccess) {\
        printf("CUDA Error\n");\
        printf("    FILE:    %s\n", __FILE__);\
        printf("    LINE:    %s\n", __LINE__);\
        printf("    ERROR CODE:    %d\n", error_code);\
        printf("    ERROR MESSAGE: %d\n", cudaGetErrorString(error_code))\
        exit(1);
    }\
} while(0)
```



这里使用CHECK宏捕获错误信息，作为call参数传入，如果显示有错误，则输出错误信息并且退出

`__FILE__`和`__LINE__`是内置的宏，是当前的文件和行号

cudaGetErrorString可以根据错误码输出错误信息



#### 检查API函数

对于运行时API函数，会返回一个cudaError类型的变量来传递错误信息

如果成功则返回cudaSuccess，否则返回错误码

使用CHECK宏包裹函数即可：`CHECK(cudaMalloc)`



#### 检查核函数

使用

```cpp
CHECK(cudaGetLastError());
CHECK(cudaDeviceSynchronize());
```

cudaGetLastError()：返回最近发生的错误码

cudaDeviceSynchronize()：进行同步，保证核函数执行完毕
