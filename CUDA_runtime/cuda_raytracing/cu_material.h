#ifndef __SUN_MATERIAL_H
#define __SUN_MATERIAL_H

#include "device_launch_parameters.h"
#include "CUDA_runtime/cu_math/cu_vector.h"
#include "cu_texture.h"

extern struct CRay;
extern struct CHitRecord;

enum EMatType {
    MT_Lambertian,
    MT_Metal,
};

struct CMaterial {
    EMatType type;
    uint32_t dataSiz;
    void* data;
};

struct CLambertian {

    CTexture* tex;
};

struct CMetal {
    cvec3 albedo;
};

///////////////////////////////////////material--scattered//////////////////////////////////////
extern __host__ __device__ bool scatter(const CMaterial& material, CRay r_in, CHitRecord rec, cvec3* attenuation, CRay* scattered, int& num);



#endif