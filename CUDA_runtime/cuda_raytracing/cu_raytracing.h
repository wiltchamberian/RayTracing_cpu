#ifndef __CUDA_RAYTRACING_H
#define __CUDA_RAYTRACING_H

#include "cu_camera.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>


/*__host__ */__device__ CHitRecord cu_raytracing_hit(const CRay& ray, HitObject* vec, int num);
void host_raytracing(CTexture tex, int x,int y, const CCamera& camera, HitObject* vec,int num);
__global__ void cu_raytracing(cudaSurfaceObject_t surface,CCamera camera, HitObject* vec, int num);

#endif