#include "cu_material.h"
#include "cu_hitable.h"
#include "cu_vector.h"

////////////////////////////////material//////////////////////////////////
__host__ __device__ bool scatter(const CMaterial& material, CRay r_in, CHitRecord rec, cvec3* attenuations, CRay* scattered, int& num) {
    if (material.type == EMatType::MT_Metal) {
        CMetal* metal = (CMetal*)(material.data);
        num = 1;
        cvec3 reflected = vec3_reflect(r_in.dir, rec.normal);
        scattered[0].ori = rec.p;
        scattered[0].dir = reflected;
        attenuations[0] = metal->albedo;
        return  vec3_dot(scattered[0].dir, rec.normal) > 0;
    }

    return false;
}
