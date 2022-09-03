#ifndef __BVH_H
#define __BVH_H

#include <vector>
#include "Hitable.h"
#include "Math/Box.h"

namespace Sun {

    class BVH_Node : public Hitable
    {
    public:
        BVH_Node();
        BVH_Node(std::vector<Hitable*>& vec, float time0, float time1);
        virtual bool hit(const Ray& ray, float tmin, float tmax, HitRecord& rec) const;
        virtual bool boundingBox(float t0, float t1, AABB& box) const;

        Hitable* left_;
        Hitable* right_;
        AABB box_;
    };

}

#endif

