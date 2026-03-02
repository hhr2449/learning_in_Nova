#include "error.cuh"

int main() {
    size_t big_n = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
    float *test_d;
    // 分配一个超大内存
    CHECK(cudaMalloc(&test_d, big_n * sizeof(float)));
    CHECK(cudaFree(test_d));
    return 0;
}