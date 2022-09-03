#include "Box.h"

namespace Sun {
    AABB::AABB() {

    }

    vec3 AABB::center() const {
        return (mins_ + maxs_) * 0.5;
    }

    AABB::AABB(const vec3& a, const vec3& b)
    {
        mins_ = a;
        maxs_ = b;
    }

    bool AABB::hit(const Ray& ray, float tmin, float tmax) const {
        float* p = (float*)(&mins_);
        float* q = (float*)(&maxs_);
        float* r = (float*)(&ray.dir);
        float invD = 1.0;
        float d1 = 0.0;
        float d2 = 0.0;
        float t1 = 0.0;
        float t2 = 0.0;
        for (int i = 0; i < 3; ++i, p++, q++, r++) {
            invD = 1 / (*r);
            d1 = (*p - *r) * invD;
            d2 = (*q - *r) * invD;
            float t1 = d1 < d2 ? d1 : d2;
            float t2 = d1 < d2 ? d2 : d1;

            tmin = t1 < tmin ? tmin : t1;
            tmax = t2 < tmax ? t2 : tmax;

            if (tmax <= tmin)
                return false;
        }
        return true;
    }

    bool AABB::hit2(const Ray& r, float tmin, float tmax) const {
        return true;
    }

    AABB AABB::surroundingBox(const AABB& box1, const AABB& box2){
        AABB box;
        box.mins_.x = Math::lowerBound(box1.mins_.x, box2.mins_.x);
        box.mins_.y = Math::lowerBound(box1.mins_.y, box2.mins_.y);
        box.mins_.z = Math::lowerBound(box1.mins_.z, box2.mins_.z);

        box.maxs_.x = Math::upperBound(box1.maxs_.x, box2.maxs_.x);
        box.maxs_.y = Math::upperBound(box1.maxs_.y, box2.maxs_.y);
        box.maxs_.z = Math::upperBound(box1.maxs_.z, box2.maxs_.z);

        return box;
    }
}

