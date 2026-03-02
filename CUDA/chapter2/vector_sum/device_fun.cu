#include <stdio.h>
#include <math.h>
#include <chrono>

const size_t N = 300000005;
const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

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

