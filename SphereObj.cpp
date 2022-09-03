#include "SphereObj.h"
#include "Math/MathUtils.h"
#include <cassert>

namespace Sun {

    SphereObj::SphereObj() {

    }

    SphereObj::SphereObj(const vec3& v, float f, Material* material)
        :Sphere(v, f)
    {
        material_ = material;
    }

    //p should be a normal 
    void get_sphere_uv(const vec3& p, float& u, float& v) {
        float phi = atan2(p.z, p.x);
        float theta = asin(p.y);
        u = 1 - (phi + A_PI) / (2 * A_PI);
        v = (theta + A_PI / 2) / A_PI;
    }

    bool SphereObj::hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const {
        if (!(r < 0 || r>0)) { //FIXME
            return false;
        }
        vec3 oc = ray.ori - center;
        float A = ray.dir.dotProduct(ray.dir);
        float B = ray.dir.dotProduct(oc);
        float C = oc.dotProduct(oc) - r * r;
        float discriminant = B * B - A * C;
        if (discriminant > 0) {
            float sqr = sqrt(discriminant);
            float t = (-B - sqr) / A;
            if (t<t_max && t>t_min) {
                rec.t = t;
                rec.p = ray.pointAt(rec.t);
                rec.normal = (rec.p - center) / r;
                rec.material = material_;
                get_sphere_uv(rec.normal, rec.u, rec.v);
                return true;
            }
            t = (-B + sqr) / A;
            if (t<t_max && t>t_min) {
                rec.t = t;
                rec.p = ray.pointAt(rec.t);
                rec.normal = (rec.p - center) / r;
                rec.material = material_;
                get_sphere_uv(rec.normal, rec.u, rec.v);
                return true;
            }
        }
        return false;
    }

    bool SphereObj::boundingBox(float t0, float t1, AABB& box) const {
        float R = fabs(Sphere::r);
        box.mins_.x = center.x - R;
        box.mins_.y = center.y - R;
        box.mins_.z = center.z - R;
        box.maxs_.x = center.x + R;
        box.maxs_.y = center.y + R;
        box.maxs_.z = center.z + R;
        return true;
    }

    ////////////////////////movingSphere//////////////////
    MovingSphereObj::MovingSphereObj()
    {
    }

    MovingSphereObj::MovingSphereObj(const vec3& cen0, const vec3& cen1, float t0, float t1, float r, Material* m)
        : center0_(cen0)
        , center1_(cen1)
        , time0_(t0)
        , time1_(t1)
        , radius_(r)
        , m_(m)
    {
        float dt = t0 - t1;
        dt = dt < 0 ? (-dt) : dt;
        if (dt < MIN_flt) {
            assert(false);
        }
    }

    //代码上面基本一样
    bool MovingSphereObj::hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const {
        if (!(radius_ < 0 || radius_>0)) { //FIXME
            return false;
        }
        vec3 oc = ray.ori - center(ray.time_);
        float A = ray.dir.dotProduct(ray.dir);
        float B = ray.dir.dotProduct(oc);
        float C = oc.dotProduct(oc) - radius_ * radius_;
        float discriminant = B * B - A * C;
        if (discriminant > 0) {
            float sqr = sqrt(discriminant);
            float t = (-B - sqr) / A;
            if (t<t_max && t>t_min) {
                rec.t = t;
                rec.p = ray.pointAt(rec.t);
                rec.normal = (rec.p - center(ray.time_)) / radius_;
                rec.material = material_;
                return true;
            }
            t = (-B + sqr) / A;
            if (t<t_max && t>t_min) {
                rec.t = t;
                rec.p = ray.pointAt(rec.t);
                rec.normal = (rec.p - center(ray.time_)) / radius_;
                rec.material = material_;
                return true;
            }
        }
        return false;
    }

    bool MovingSphereObj::boundingBox(float t0, float t1, AABB& box) const {
        vec3 cent0 = center(t0);
        vec3 cent1 = center(t1);

        vec3 v(radius_, radius_, radius_);

        vec3 min0 = cent0 - v;
        vec3 max0 = cent0 + v;

        vec3 min1 = cent1 - v;
        vec3 max1 = cent1 + v;

        box.mins_.x = Math::lowerBound(min0.x, min1.x);
        box.mins_.y = Math::lowerBound(min0.y, min1.y);
        box.mins_.z = Math::lowerBound(min0.z, min1.z);

        box.maxs_.x = Math::upperBound(max0.x, max1.x);
        box.maxs_.y = Math::upperBound(max0.y, max1.y);
        box.maxs_.z = Math::upperBound(max0.z, max1.z);

        return false;
    }

    vec3& MovingSphereObj::center0() {
        return center0_;
    }

    vec3& MovingSphereObj::center1() {
        return center1_;
    }

    vec3 MovingSphereObj::center(float time) const {
        return center0_ + (center1_ - center0_) * ((time - time0_) / (time1_ - time0_));
    }

}
