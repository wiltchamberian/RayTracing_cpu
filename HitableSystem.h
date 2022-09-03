#pragma once

#include "Math/Ray.h"
#include "Math/Box.h"
#include "HitRecord.h"
#include "SphereObj.h"

namespace Sun {

    class HitableSystem
    {
    public:
        //判断是否碰撞，如果碰撞，碰撞信息输出在HitRecord
        bool hit(const SphereObj& sphereObj, const Ray& ray, float t_min, float t_max, HitRecord& rec);
        //计算boundingBox
        virtual bool boundingBox(float t0, float t1, AABB& box) const = 0;

    protected:

    };

}

