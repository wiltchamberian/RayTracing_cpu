#pragma once

#include "cuda_runtime.h"

__global__ void addKernel(int* c, const int* a, const int* b);

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
