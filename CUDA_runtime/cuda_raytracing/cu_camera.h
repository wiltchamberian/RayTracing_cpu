#ifndef __CUDA_CAMERA_H
#define __CUDA_CAMERA_H

#include "device_launch_parameters.h"
#include "cu_vector.h"
#include "cu_hitable.h"

class CCamera
{
public:
    CCamera();
    CCamera(const cvec3& lookfrom, const cvec3& lookat, const cvec3& vup, float fov, float aspect
        , float aperture, float focus_dist, float t0, float t1);
    ~CCamera();

    __host__ __device__  CRay buildRay(float u, float v) const;

    cvec3 origin_;
    cvec3 lower_left_corner_;
    cvec3 horizontal_;
    cvec3 vertical_;
    cvec3 u_, v_, w_;
    //͸���뾶
    float lens_radius_;
    //��ֹʱ��
    float time0_, time1_;

};

extern __host__ __device__ CRay camera_buildRay(const CCamera& camera,float u, float v);



#endif