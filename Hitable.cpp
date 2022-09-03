#include "Hitable.h"
#include <vector>
#include "HitableList.h"

namespace Sun {

    bool xy_rect::hit(const Ray& ray, float t0, float t1, HitRecord& rec) const {
        float t = (z_ - ray.ori.z) / ray.dir.z;
        if (t<t0 || t>t1)
            return false;
        float x = ray.ori.x + t * ray.dir.x;
        float y = ray.ori.y + t * ray.dir.y;
        if (x<x0_ || x>x1_ || y<y0_ || y>y1_)
            return false;

        rec.u = (x - x0_) / (x1_ - x0_);
        rec.v = (y - y0_) / (y1_ - y0_);
        rec.t = t;
        rec.material = material_;
        rec.p = ray.pointAt(t);
        //rec.normal = { 0,0,1 };
        rec.normal = { 0,0,ray.dir.z > 0 ? (-1.f) : 1.f };

        return true;
    }

    bool xz_rect::hit(const Ray& ray, float t0, float t1, HitRecord& rec) const {
        float t = (k_ - ray.ori.y) / ray.dir.y;
        if (t<t0 || t>t1)
            return false;
        float x = ray.ori.x + t * ray.dir.x;
        float z = ray.ori.z + t * ray.dir.z;
        if (x<x0_ || x>x1_ || z<z0_ || z>z1_)
            return false;

        rec.u = (x - x0_) / (x1_ - x0_);
        rec.v = (z - z0_) / (z1_ - z0_);
        rec.t = t;
        rec.material = material_;
        rec.p = ray.pointAt(t);

        //rec.normal = { 0,1,0 };
        rec.normal = { 0,ray.dir.y > 0 ? (-1.f) : 1.f,0 };

        return true;
    }

    bool yz_rect::hit(const Ray& ray, float t0, float t1, HitRecord& rec) const {
        float t = (k_ - ray.ori.x) / ray.dir.x;
        if (t<t0 || t>t1)
            return false;
        float y = ray.ori.y + t * ray.dir.y;
        float z = ray.ori.z + t * ray.dir.z;
        if (y < y0_ || y > y1_ || z < z0_ || z > z1_)
            return false;

        rec.u = (y - y0_) / (y1_ - y0_);
        rec.v = (z - z0_) / (z1_ - z0_);
        rec.t = t;
        rec.material = material_;
        rec.p = ray.pointAt(t);
        //rec.normal = { 1,0,0 };
        rec.normal = { ray.dir.x > 0 ? (-1.f) : 1.f,0,0 };

        return true;
    }

    box::box(const vec3& p0, const vec3& p1,
        Material* ptr)
    {
        mins = p0;
        maxs = p1;
        std::vector<Hitable*> vec(6, nullptr);
        vec[0] = new xy_rect(p0.x, p1.x, p0.y, p1.y, p1.z, ptr);
        vec[1] = new xy_rect(p0.x, p1.x, p0.y, p1.y, p0.z, ptr);
        vec[2] = new xz_rect(p0.x, p1.x, p0.z, p1.z, p1.y, ptr);
        vec[3] = new xz_rect(p0.x, p1.x, p0.z, p1.z, p0.y, ptr);
        vec[4] = new yz_rect(p0.y, p1.y, p0.z, p1.z, p1.x, ptr);
        vec[5] = new yz_rect(p0.y, p1.y, p0.z, p1.z, p0.x, ptr);
        listPtr = new HitableList(vec);
    }

    bool box::hit(const Ray& ray, float t0, float t1, HitRecord& rec) const {
        return listPtr->hit(ray, t0, t1, rec);
    }

}