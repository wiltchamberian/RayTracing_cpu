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
    //��ǹ�����ײ���ڹ�·�ϵĳ߶�(�ӹ��߷����㵽��ײ��ĳ���)
    float t;
    //������ײ��
    cvec3 p;
    //��ײλ�õĵ�λ����
    cvec3 normal;
    //��ײ�����id
    int id = 0;
    //�����ײ���Ӧ����ĵ��u,v����
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



