/*****************************************************************************
* @brief : Sphere
* 3Œ¨«ÚÃÂ¿‡
* @author : acedtang
* @date : 2021/4/26
* @version : ver 1.0
* @inparam :
* @outparam :
*****************************************************************************/

#ifndef __SPHERE_H
#define __SPHERE_H

#include "Vector3D.h"

namespace Sun {

    struct Sphere {
        Sphere();
        Sphere(const Sphere& other) {
            center = other.center;
            r = other.r;
        }
        Sphere(const vec3& center, float r);
        
        vec3& getCenter();
        float& getRadius();
        float getSquareRadius() const {
            return r * r;
        }

        vec3 center;
        float r;
    };

}

#endif