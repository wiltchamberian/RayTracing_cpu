#pragma once

#include "Hitable.h"
#include <vector>

namespace Sun {

    class HitableList : public Hitable
    {
    public:
        HitableList();
        HitableList(const std::vector<Hitable*>& vec);
        virtual bool hit(const Ray& ray, float tmin, float tmax, HitRecord& rec) const override;
        virtual bool boundingBox(float t0, float t1, AABB& box) const override;
    public:
        std::vector<Hitable*> vec_;
    };

}

