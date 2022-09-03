#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__device__ int julia(int x, int y);

__global__ void kernel(unsigned char* ptr);

extern void julia_test();