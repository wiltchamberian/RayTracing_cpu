#include "Ray.h"

namespace Sun {
    Ray::Ray() {

    }

    Ray::Ray(const vec3& origin, const vec3& direction, float time)
        :ori(origin)
        ,dir(direction)
        ,time_(time)
    {

    }

    vec3& Ray::origin() {
        return ori;
    }

    vec3& Ray::direction() {
        return dir;
    }

    vec3 Ray::pointAt(float t) const {
        return ori + dir * t;
    }

}