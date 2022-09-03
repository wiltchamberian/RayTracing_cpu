#pragma once

#include "Math/Vector3D.h"
#include "Math/Ray.h"
#include "Hitable.h"


namespace Sun {

    extern vec3 rayTracing(const Ray& ray, Hitable* world, int depth);

    //∑¥…‰
    extern vec3 reflect(const vec3& input, const vec3& normal);

    //’€…‰
    extern bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted);

    extern float schlick(float cosine, float ref_idx);

}


