#include "func.h"
#include <stdio.h>

__global__ void add(int n, int* c, int* a, int* b) {

    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        c[i] = a[i] + b[i];
    }
    c[0] = blockDim.x;
    return;
}

void parrel_add(int n, int* a, int* b, int* c) {
    cudaError_t er;

    //分配gpu内存
    int* da = nullptr;
    int* db = nullptr;
    int* dc = nullptr;
    er = cudaMalloc(&da, n * sizeof(int));
    assert(er == cudaSuccess);
    er = cudaMalloc(&db, n * sizeof(int));
    er = cudaMalloc(&dc, n * sizeof(int));

    //拷贝数据到gpu
    er = cudaMemcpy(da, a, n * sizeof(int), cudaMemcpyHostToDevice);


    er = cudaMemcpy(db, b, n * sizeof(int), cudaMemcpyHostToDevice);

    //触发gpu并行计算
    add << <20, 1 >> > (n, dc, da, db);

    //结果拷贝回cpu
    HANDLE_ERROR(cudaMemcpy(c, dc, n * sizeof(int), cudaMemcpyDeviceToHost));

    //释放gpu
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return;

}



