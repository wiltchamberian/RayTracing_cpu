#include "HitableList.h"

namespace Sun {

    HitableList::HitableList() {

    }

    HitableList::HitableList(const std::vector<Hitable*>& vec) {
        vec_ = vec;
    }

    bool HitableList::hit(const Ray& ray, float tmin, float tmax, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        double closest_so_far = tmax;
        for (int i = 0; i < vec_.size(); ++i) {
            if (vec_[i]->hit(ray, tmin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    bool HitableList::boundingBox(float t0, float t1, AABB& box) const {
        if (vec_.empty()) return false;
        bool ok = vec_[0]->boundingBox(t0, t1, box);
        if (!ok) return false;
        AABB tmp;
        for (int i = 1; i < vec_.size(); ++i) {
            bool ok = vec_[i]->boundingBox(t0, t1, tmp);
            if (!ok) return false;
            box = AABB::surroundingBox(box, tmp);
        }
        return true;
    }

}
