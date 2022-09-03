#include "Sphere_Ray.h"


namespace Sun {

    float hit(const Sphere& sphere, const Ray& ray) {
        vec3 oc = ray.ori - sphere.center;
        float A = ray.dir.dotProduct(ray.dir);
        float B = 2 * ray.dir.dotProduct(oc);
        float C = oc.dotProduct(oc) - sphere.r * sphere.r;
        float discriminant = B * B - 4 * A * C;
        if (discriminant < 0) {
            return -1.;
        }
        return (-B - sqrt(discriminant)) / (2 * A);
    }

}