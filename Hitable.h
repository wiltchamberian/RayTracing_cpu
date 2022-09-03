#pragma once

#include "Math/Ray.h"
#include "Math/Box.h"
#include "HitRecord.h"

namespace Sun {

    class Material;

    //碰撞基类
    class Hitable
    {
    public:
        //判断是否碰撞，如果碰撞，碰撞信息输出在HitRecord
        virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const = 0;
        //计算boundingBox
        virtual bool boundingBox(float t0, float t1, AABB& box) const = 0;
        Material* material_;
    };

    class xy_rect :public Hitable {
    public:
        xy_rect() {}
        xy_rect(float x0, float x1, float y0, float y1, float k, Material* m) :
            x0_(x0), x1_(x1), y0_(y0), y1_(y1), z_(k) {
            material_ = m;
        }
        virtual bool hit(const Ray& ray, float t0, float t1, HitRecord& rec)const;
        virtual bool boundingBox(float t0, float t1, AABB& box) const {
            box = AABB({ x0_,y0_,z_ - 0.0001f }, { x1_,y1_,z_ + 0.0001f });
            return true;
        }
        float x0_, x1_, y0_, y1_, z_;
    };

    class xz_rect :public Hitable {
    public:
        xz_rect() {}
        xz_rect(float x0, float x1, float z0, float z1, float k, Material* m) :
            x0_(x0), x1_(x1), z0_(z0), z1_(z1), k_(k) {
            material_ = m;
        }
        virtual bool hit(const Ray& ray, float t0, float t1, HitRecord& rec)const;
        virtual bool boundingBox(float t0, float t1, AABB& box) const {
            box = AABB({ x0_,k_ - 0.0001f, z0_ }, { x1_,k_ + 0.0001f, z1_ });
            return true;
        }
        float x0_, x1_, z0_, z1_, k_;
    };

    class yz_rect :public Hitable {
    public:
        yz_rect() {}
        yz_rect(float y0, float y1, float z0, float z1, float k, Material* m) :
            y0_(y0), y1_(y1), z0_(z0), z1_(z1), k_(k) {
            material_ = m;
        }
        virtual bool hit(const Ray& ray, float t0, float t1, HitRecord& rec)const;
        virtual bool boundingBox(float t0, float t1, AABB& box) const {
            box = AABB({ k_ - 0.0001f,y0_,z0_ }, { k_ + 0.0001f,y1_,z1_ });
            return true;
        }
        float y0_, y1_, z0_, z1_, k_;
    };

    class box :public Hitable {
    public:
        box() {}
        box(const vec3& p0, const vec3& p1,
            Material* ptr);
        virtual bool hit(const Ray& ray, float t0, float t1,
            HitRecord& rec) const;
        virtual bool boundingBox(float t0, float t1, AABB& box)const
        {
            box = AABB(mins, maxs);
            return true;
        }
        vec3 mins, maxs;
        Hitable* listPtr;
    };

}

