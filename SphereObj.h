#pragma once

#include "Hitable.h"
#include "Math/Sphere.h"

namespace Sun {

    class SphereObj :public Sphere, public Hitable
    {
    public:
        SphereObj();
        SphereObj(const vec3& v, float f, Material* material);
        virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const override;
        virtual bool boundingBox(float t0, float t1, AABB& box) const override;
    };

    class MovingSphereObj : public Hitable {
    public:
        MovingSphereObj();
        MovingSphereObj(const vec3&, const vec3&, float t0, float t1, float r, Material* m);
        virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const override;
        virtual bool boundingBox(float t0, float t1, AABB& box) const;
        vec3& center0();
        vec3& center1();
        vec3 center(float time) const;
        vec3 center0_, center1_;
        float time0_, time1_;
        float radius_;
        Material* m_;
    };

}

