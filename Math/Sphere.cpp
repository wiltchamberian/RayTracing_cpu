#include "Sphere.h"

namespace Sun {

    Sphere::Sphere() {

    }

    Sphere::Sphere(const vec3& c, float R)
        :center(c)
        ,r(R)
    {


    }

    vec3& Sphere::getCenter() {
        return center;
    }

    float& Sphere::getRadius() {
        return r;
    }
}