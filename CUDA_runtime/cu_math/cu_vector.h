#ifndef __SUN_CUDA_VECTOR_H
#define __SUN_CUDA_VECTOR_H

#include "device_launch_parameters.h"

#ifndef A_PI
#define A_PI 3.14159265358979
#endif

struct cvec3{
    float x;
    float y;
    float z;
};

extern __host__ __device__ cvec3 vec3_gen(float x, float y, float z);
extern __host__ __device__ cvec3 vec3_add(const cvec3& v1, const cvec3& v2);
extern __host__ __device__ cvec3 vec3_sub( const cvec3& v1, const cvec3& v2);
extern __host__ __device__ float vec3_dot(const cvec3& v1,const cvec3& v2);
extern __host__ __device__ cvec3 vec3_mul(const cvec3& v1, const cvec3& v2);
extern __host__ __device__ cvec3 vec3_nmul(const cvec3& v1, float k);
extern __host__ __device__ cvec3 vec3_ndiv(const cvec3& v1, float k);
extern __host__ __device__ cvec3 vec3_reflect(const cvec3& v, const cvec3& normal);
extern __host__ __device__ cvec3 vec3_normalize(cvec3& vec);
extern __host__ __device__ cvec3 vec3_cross(const cvec3& v1, const cvec3& v2);

#endif