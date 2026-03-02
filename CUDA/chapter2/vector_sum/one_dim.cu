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