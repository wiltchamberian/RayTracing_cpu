#ifndef __SPHERE_RAY_H
#define __SPHERE_RAY_H

#include "Vector3D.h"
#include "Sphere.h"
#include "Ray.h"

namespace Sun {

    float hit(const Sphere& sphere, const Ray& ray);

}


#endif