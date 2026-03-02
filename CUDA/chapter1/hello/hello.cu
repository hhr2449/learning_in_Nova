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