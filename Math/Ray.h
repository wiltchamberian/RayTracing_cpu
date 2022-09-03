/*****************************************************************************
* @brief : Ray
* …‰œﬂ
* @author : acedtang
* @date : 2021/4/26
* @version : ver 1.0
* @inparam :
* @outparam :
*****************************************************************************/

#ifndef __RAY_H
#define __RAY_H

#include "Vector3D.h"

namespace Sun {

    class Ray {
    public:
        Ray();
        Ray(const vec3& origin, const vec3& direction, float time = 0.f);
        vec3& origin();
        vec3& direction();
        vec3 pointAt(float t) const;
        vec3 ori;
        //must be normalized for easy to use
        vec3 dir;

        float time_;
    };

}


#endif