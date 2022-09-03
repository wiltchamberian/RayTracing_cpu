/*****************************************************************************
* @brief : brief
* @author : acedtang
* @date : 2021/3/23
* @version : ver 1.0
* @inparam : 
* @outparam : 
*****************************************************************************/

#ifndef __BOX_H
#define __BOX_H

#include "Vector3D.h"
#include "Ray.h"
#include <algorithm>

namespace Sun {

    struct Box {
        vec3 mins_;
        vec3 maxs_;
    };

    class AABB :public Box{
    public:
        AABB();
        AABB(const vec3& a, const vec3& b);

        vec3 center() const;

        inline float volume() const {
            vec3 v = maxs_ - mins_;
            return fabs(v.x * v.y * v.z);
        }

        inline AABB combine(const AABB& box) const {
            AABB res;

            res.mins_.x = std::min<float>(box.mins_.x, mins_.x);
            res.mins_.y = std::min<float>(box.mins_.y, mins_.y);
            res.mins_.z = std::min<float>(box.mins_.z, mins_.z);

            res.maxs_.x = std::max<float>(box.maxs_.x, maxs_.x);
            res.maxs_.y = std::max<float>(box.maxs_.y, maxs_.y);
            res.maxs_.z = std::max<float>(box.maxs_.z, maxs_.z);

            return res;
        }

        //获取1/4周长
        inline float getOneQuaterOfPerimeter() const{
            return (maxs_.x - mins_.x) + (maxs_.y - mins_.y) + (maxs_.z - mins_.z);
        }

        inline void extend(const vec3& margins) {
            for (int i = 0; i < 3; ++i) {
                maxs_[i] += margins[i];
                mins_[i] -= margins[i];
            }
        }

        inline bool contains(const AABB& box) const {
            return maxs_.x > box.maxs_.x
                && maxs_.y > box.maxs_.y
                && maxs_.z > box.maxs_.z
                && mins_.x < box.mins_.x
                && mins_.y < box.mins_.y
                && mins_.z < box.mins_.z;
        }

        inline bool overLap(const AABB& box) const {
            vec3 d1 = mins_ - box.maxs_;
            vec3 d2 = box.mins_ - maxs_;

            if (d1.x > 0 || d1.y > 0 || d1.z > 0) {
                return false;
            }
            if (d2.x > 0 || d2.y > 0 || d2.z > 0) {
                return false;
            }

            return true;
        }

        inline AABB translateAABB(const vec3& v) const {
            AABB res = *this;
            res.mins_ += v;
            res.maxs_ += v;
            return res;
        }

        inline AABB expandByTranslate(const vec3& translate) const {
            AABB other = translateAABB(translate);
            other = combine(other);
            return other;
        }

        bool hit(const Ray& r, float tmin, float tmax) const;
        //another for glsl optimize
        bool hit2(const Ray& r, float tmin, float tmax) const;
        //surroundingBox
        static AABB surroundingBox(const AABB& box1, const AABB& box2);
    };

    //定向包围盒
    struct OBB {
        //三个归一化正交轴
        vec3 v0;
        vec3 v1;
        vec3 v2;

        float e0, e1, e2; //extend

        vec3 center;
    };
}

#endif