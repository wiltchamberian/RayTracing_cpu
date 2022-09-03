#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>

#define HANDLE_ERROR(func) \
{ cudaError_t er = func; \
    assert(er == cudaSuccess);\
    if(er!=cudaSuccess){ \
        printf("%s\n",cudaGetErrorString(er));\
    }\
}

__global__ void add(int n, int* c, int* a, int* b);


extern void parrel_add(int n, int* a, int* b, int* c);

__global__ void kernal_anim(unsigned char* ptr, int ticks);





