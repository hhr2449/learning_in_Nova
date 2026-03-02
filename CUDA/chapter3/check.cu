#include "error.cuh"

__global__ void addKernel_Error(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    // 故意制造错误：size 只有 5，但我尝试访问索引 10000
    // 这将导致非法地址访问
    c[i + 10000] = a[i] + b[i]; 
}

int main() {
    const int size = 5;
    int a[size] = {1, 2, 3, 4, 5}, b[size] = {10, 20, 30, 40, 50}, c[size] = {0};
    int *dev_a, *dev_b, *dev_c;

    CHECK(cudaMalloc((void**)&dev_a, size * sizeof(int)));
    CHECK(cudaMalloc((void**)&dev_b, size * sizeof(int)));
    CHECK(cudaMalloc((void**)&dev_c, size * sizeof(int)));

    CHECK(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

    printf("Launching kernel with intentional error...\n");
    
    // 启动核函数
    addKernel_Error<<<1, size>>>(dev_c, dev_a, dev_b);

    // 1. 启动检查：通常会通过，因为配置（1 block, 5 threads）是合法的
    CHECK(cudaGetLastError());
    printf("cudaGetLastError() passed (Launch is OK).\n");

    // 2. 同步检查：这里会报错！因为 GPU 在执行到越界内存写入时崩溃了
    printf("Synchronizing...\n");
    CHECK(cudaDeviceSynchronize()); 

    CHECK(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}