#ifndef __CUDA_HITALBE_H
#define __CUDA_HITABLE_H

#include "device_launch_parameters.h"
#include "CUDA_runtime/cu_math/cu_vector.h"
#include "cu_material.h"


struct CRay {
    cvec3 ori;
    //must be normalized for easy to use
    cvec3 dir;
    float time;

};

extern __host__ __device__ cvec3 ray_pointAt(const CRay& ray, float t);

struct CHitRecord
{
    //标记光线碰撞点在光路上的尺度(从光线发出点到碰撞点的长度)
    float t;
    //光线碰撞点
    cvec3 p;
    //碰撞位置的单位法线
    cvec3 normal;
    //碰撞物体的id
    int id = 0;
    //标记碰撞点对应物体的点的u,v坐标
    float u = 0.f;
    float v = 0.f;
    bool isHit = false;
};

enum EObjType {
    OT_SPHERE,
};

struct HitObject {
    EObjType type;
    int id;
    void* data;
    CMaterial material;
};

struct CSphere {
    cvec3 center;
    float r;
};

//sphere
extern /*__host__ */__device__ void sphere_hit(const HitObject& data, const CRay& ray, float t_min, float t_max, CHitRecord& rec);



#endif 



