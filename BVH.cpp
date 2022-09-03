#include "BVH.h"
#include <algorithm>
#include <cassert>
#include "Math/Random.h"

namespace Sun {

    BVH_Node::BVH_Node()
        :left_(0)
        , right_(0)
    {

    }

    BVH_Node::BVH_Node(std::vector<Hitable*>& vec, float time0, float time1)
    {
        if (vec.empty())return;
        int axis = int(3 * rand48());
        if (axis == 0) {
            std::sort(vec.begin(), vec.end(), [](Hitable* a, Hitable* b)->bool {
                AABB box1, box2;
                if (!a->boundingBox(0, 0, box1) || !b->boundingBox(0, 0, box2)) {
                    assert(false);
                    return false;
                }
                if (box1.mins_.x - box2.mins_.x < 0) {
                    return true;
                }
                return false;
                });
        }
        else if (axis == 1) {
            std::sort(vec.begin(), vec.end(), [](Hitable* a, Hitable* b)->bool {
                AABB box1, box2;
                if (!a->boundingBox(0, 0, box1) || !b->boundingBox(0, 0, box2)) {
                    assert(false);
                    return false;
                }
                if (box1.mins_.y - box2.mins_.y < 0) {
                    return true;
                }
                return false;
                });
        }
        else if (axis == 2) {
            std::sort(vec.begin(), vec.end(), [](Hitable* a, Hitable* b)->bool {
                AABB box1, box2;
                if (!a->boundingBox(0, 0, box1) || !b->boundingBox(0, 0, box2)) {
                    assert(false);
                    return false;
                }
                if (box1.mins_.z - box2.mins_.z < 0) {
                    return true;
                }
                return false;
                });
        }
        if (vec.size() == 1) {
            left_ = vec[0];
            right_ = nullptr;
        }
        else if (vec.size() == 2) {
            left_ = vec[0];
            right_ = vec[1];
        }
        else {
            std::vector<Hitable*> v_l(vec.begin(), vec.begin() + vec.size() / 2);
            std::vector<Hitable*> v_r(vec.begin() + vec.size() / 2, vec.end());
            left_ = new BVH_Node(v_l, time0, time1);
            right_ = new BVH_Node(v_r, time0, time1);
        }
        AABB boxLeft, boxRight;
        if (!left_->boundingBox(time0, time1, boxLeft) || !right_->boundingBox(time0, time1, boxRight)) {
            assert(false);
        }
        box_ = AABB::surroundingBox(boxLeft, boxRight);
    }

    bool BVH_Node::boundingBox(float t0, float t1, AABB& box) const {
        box = box_;
        return true;
    }

    bool BVH_Node::hit(const Ray& ray, float tmin, float tmax, HitRecord& rec) const {
        if (box_.hit(ray, tmin, tmax)) {
            HitRecord leftRec, rightRec;
            bool hitLeft = false;
            bool hitRight = false;
            if (left_) {
                hitLeft = left_->hit(ray, tmin, tmax, leftRec);
            }
            if (right_) {
                hitRight = right_->hit(ray, tmin, tmax, rightRec);
            }
            if (hitLeft && hitRight) {
                if (leftRec.t <= rightRec.t)
                    rec = leftRec;
                else
                    rec = rightRec;
                return true;
            }
            else if (hitLeft) {
                rec = leftRec;
                return true;
            }
            else if (hitRight) {
                rec = rightRec;
                return true;
            }
            else
                return false;
        }
        return false;
    }

}
