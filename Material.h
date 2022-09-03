#pragma once

#include <cmath>
#include "Math/Vector3D.h"
#include "Math/Ray.h"
#include "Math/Random.h"
#include "RayTracing.h"
#include "Hitable.h"
#include "texture.h"
#include "HitRecord.h"

namespace Sun {

    //材质基类
    class Material
    {
    public:
        //实现scatter函数，输入:ray, rec;输出attenuation, 反射射线scattered, attenuation表示
        virtual bool scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered) const = 0;

        virtual vec3 emitted(float u, float v, const vec3& p)const {
            return { 0,0,0 };
        }
    };

    //朗伯材质（各项同性均匀)
    class Lambertian : public Material
    {
    public:
        Lambertian(texture* a) : albedo_(a) {
        }
        ~Lambertian() {
            delete albedo_;
        }

        //可以看出 这个就是简单漫反射一条随机射线
        virtual bool scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered) const {
            //散射球内随机点 (这是原始公式，个人认为可以优化到半球内)
            vec3 target = rec.p + rec.normal + randomPointInUnitSphere();
            //散射射线
            scattered = Ray(rec.p, target - rec.p);

            //float u = 0;
            //float v = 0;
            //get_sphere_uv(rec.normal, u, v);
            attenuation = albedo_->value(rec.u, rec.v, rec.p);
            return true;
        }
        //衰减（或者纹理颜色)
        //vec3 albedo;
        texture* albedo_;
    };

    //金属材质（反射+小的抖动)
    class Metal : public Material {
    public:
        Metal(const vec3& a, float f) : albedo(a) {
            fuzz_ = f < 1 ? f : 1;
        }
        //可以看出就是简单计算反射矢量，但是当入射光与法向同向时，返回false
        virtual bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered) const {
            vec3 reflected = reflect(r_in.dir, rec.normal);
            scattered = Ray(rec.p, reflected);
            attenuation = albedo;
            return scattered.dir.dotProduct(rec.normal) > 0;
        }
        vec3 albedo;
        float fuzz_;
    };

    //电介质
    class Dielectrics : public Material
    {
    public:
        Dielectrics(float ri) : ref_idx(ri) {

        }

        //电介质（如玻璃球)的散射比较复杂，根据概率选择是折射还是反射
        virtual bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered) const
        {
            vec3 outward_normal;
            vec3 reflected = reflect(r_in.dir, rec.normal);
            float ni_over_nt;
            attenuation = vec3(1.0, 1.0, 0.0);
            vec3 refracted;
            float reflect_prob = 1.0;
            //入射角余弦
            float cosine;

            //ref_idx可以认为是介质自身的折射率

            float proj = r_in.dir.dotProduct(rec.normal);
            //从折射物体内部穿出，则需要先翻转折射法线以便于计算 
            if (proj > 0) {
                outward_normal = -rec.normal;
                ni_over_nt = ref_idx;
                cosine = proj / r_in.dir.getLength();
            }
            else {
                outward_normal = rec.normal;
                ni_over_nt = 1.0 / ref_idx; //从外部进入，输入输出折射比为介质折射率倒数(真空或空气看为1)
                cosine = -proj / r_in.dir.getLength();
            }
            //输出折射矢量
            if (refract(r_in.dir, outward_normal, ni_over_nt, refracted)) {
                //scattered = Ray(rec.p, refracted);
                reflect_prob = schlick(cosine, ref_idx);
            }
            //说明reflect_prob接近1.0，则为反射
            if (rand48() < reflect_prob) {
                scattered = Ray(rec.p, reflected);
            }
            else {
                scattered = Ray(rec.p, refracted);
            }

            return true;
        }

        float ref_idx;
    };

    //发光材质
    class diffuse_light : public Material {
    public:
        diffuse_light(texture* a) :emit_(a) {}
        virtual bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered) const
        {
            return false;
        }
        virtual vec3 emitted(float u, float v, const vec3& p)const
        {
            return emit_->value(u, v, p);
        }
        texture* emit_;
    };

}