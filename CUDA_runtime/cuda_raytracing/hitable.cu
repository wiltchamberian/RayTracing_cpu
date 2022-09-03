#include "cu_hitable.h"
#include <cmath>

__host__ __device__ cvec3 ray_pointAt(const CRay& ray, float t) {
    cvec3 out;
    out.x = ray.ori.x + ray.dir.x * t;
    out.y = ray.ori.y + ray.dir.y * t;
    out.z = ray.ori.z + ray.dir.z * t;
    return out;
}

/*__host__ */__device__ void sphere_hit(const HitObject& obj, const CRay& ray, float t_min, float t_max, CHitRecord& rec) {
    rec.isHit = false;
    CSphere* sphere = (CSphere*)(obj.data);
    cvec3 oc =  vec3_sub(ray.ori, sphere->center);
    float A = vec3_dot(ray.dir,ray.dir);
    float B = vec3_dot(ray.dir,oc);
    float C = vec3_dot(oc,oc) - sphere->r * sphere->r;
    float discriminant = B * B - A * C;
    if (discriminant > 0) {
        float sqr = __fsqrt_rd(discriminant);
        //float sqr = sqrt(discriminant);
        float t = (-B - sqr) / A;
        if (t<t_max && t>t_min) {
            rec.t = t;
            rec.p = ray_pointAt(ray, rec.t);
            rec.normal = vec3_ndiv(vec3_sub(rec.p, sphere->center), sphere->r);
            rec.id = obj.id;
            rec.isHit = true;
        }
    }
    return;
}



