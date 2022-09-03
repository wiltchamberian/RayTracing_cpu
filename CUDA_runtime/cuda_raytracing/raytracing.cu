#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "cu_raytracing.h"
#include <float.h>
#include "cu_vector.h"
#include "surface_functions.h"

/*__host__ */__device__ CHitRecord cu_raytracing_hit(const CRay& ray, HitObject* vec, int num) {
    CHitRecord rec;
    rec.isHit = false;

    CHitRecord tmpRec;
    float tmin = 0.001;
    float tmax = FLT_MAX;

    double closest_so_far = tmax;
    for (int i = 0; i < num; ++i) {
        /*switch (vec[i].type) {
        case OT_SPHERE:
        {
            sphere_hit(vec[i], ray, tmin, tmax, tmpRec);
        }
        break;
        default:
            break;
        }*/
        sphere_hit(vec[i], ray, tmin, tmax, tmpRec);
        if (tmpRec.isHit) {
            closest_so_far = rec.t;
            rec = tmpRec;
        }
    }

    return rec;
}

void host_raytracing(CTexture texture,int x,int y, const CCamera& camera, HitObject* vec, int objNum)
{
#if 0
    //int x = blockDim.x * blockIdx.x + threadIdx.x;
    //int y = blockDim.y * blockIdx.y + threadIdx.y;
    //float u = ((float)(x)+0.5) / (float)(blockDim.x * gridDim.x);
    //float v = ((float)(y)+0.5) / (float(blockDim.y * gridDim.y));

    float u = ((float)(x)+0.5) / (float)(1024);
    float v = ((float)(y)+0.5) / (float)(1024);

    CRay ray = camera.buildRay(u, v);

    CHitRecord rec = cu_raytracing_hit(ray, vec, objNum);

    cvec3 output = vec3_gen(0, 0, 0);
    if (rec.isHit) {
        //自发光贡献
        int num = 4;
        CRay rays[4];//散射
        cvec3 attenutations[4];
        bool ok = scatter(vec[rec.id].material, ray, rec, attenutations, rays,num);

        //将递归逻辑展开
        if (ok) {
            for (int i = 0; i < num; ++i) {
                CHitRecord rec = cu_raytracing_hit(rays[i], vec, objNum);
                if (rec.isHit) {
                    //自发光贡献
                    
                }
                //环境光
                else {
                    cvec3 v = vec3_gen(1, 1, 1);
                    output = vec3_add(output, vec3_mul(attenutations[i], v));
                }
            }
            output = vec3_ndiv(output, num);


        }
    }
    else {
        output = vec3_gen(1, 1, 1);
    }

    //int k = (gridDim.x * blockIdx.y + blockIdx.x) * 4;
    int k = (1024 * y + x) * 4;
    texture.data[k] = (unsigned char)(output.x * 255.99);
    texture.data[k + 1] = (unsigned char)(output.y * 255.99);
    texture.data[k + 2] = (unsigned char)(output.z * 255.99);
    texture.data[k + 3] = 255;
#endif
}

__global__ void cu_raytracing(cudaSurfaceObject_t surface,CCamera camera, HitObject* vec, int objNum)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    float u = ((float)(x)+0.5) / (float)(blockDim.x * gridDim.x);
    float v = ((float)(y)+0.5) / (float(blockDim.y * gridDim.y));

    CRay ray = camera.buildRay(u, v);

    CHitRecord rec = cu_raytracing_hit(ray, vec, objNum);

    cvec3 output = vec3_gen(0, 0, 0);
    if (rec.isHit) {
        //自发光贡献
        int num = 4;
        CRay rays[4];//散射
        cvec3 attenutations[4];
        bool ok = scatter(vec[rec.id].material, ray, rec, attenutations, rays, num);

        //将递归逻辑展开
        if (ok) {
            for (int i = 0; i < num; ++i) {
                CHitRecord rec = cu_raytracing_hit(rays[i], vec, objNum);
                if (rec.isHit) {
                    //自发光贡献

                }
                //环境光
                else {
                    cvec3 v = vec3_gen(1, 1, 1);
                    output = vec3_add(output, vec3_mul(attenutations[i], v));
                }
            }
            output = vec3_ndiv(output, num);
        }
    }
    else {
        output = vec3_gen(1, 1, 1);
    }

    int k = (gridDim.x * blockIdx.y + blockIdx.x) * 4;
    //texture.data[k] = (unsigned char)(output.x * 255.99);
    //texture.data[k + 1] = (unsigned char)(output.y * 255.99);
    //texture.data[k + 2] = (unsigned char)(output.z * 255.99);
    //texture.data[k + 3] = 255;

    uchar4 data;
    data.x = (unsigned char)(output.x * 255.99);
    data.y = (unsigned char)(output.y * 255.99);
    data.z = (unsigned char)(output.z * 255.99);
    data.w = 255;

    surf2Dwrite(data, surface, x*sizeof(uchar4),y);

}