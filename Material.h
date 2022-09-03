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

    //���ʻ���
    class Material
    {
    public:
        //ʵ��scatter����������:ray, rec;���attenuation, ��������scattered, attenuation��ʾ
        virtual bool scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered) const = 0;

        virtual vec3 emitted(float u, float v, const vec3& p)const {
            return { 0,0,0 };
        }
    };

    //�ʲ����ʣ�����ͬ�Ծ���)
    class Lambertian : public Material
    {
    public:
        Lambertian(texture* a) : albedo_(a) {
        }
        ~Lambertian() {
            delete albedo_;
        }

        //���Կ��� ������Ǽ�������һ���������
        virtual bool scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered) const {
            //ɢ����������� (����ԭʼ��ʽ��������Ϊ�����Ż���������)
            vec3 target = rec.p + rec.normal + randomPointInUnitSphere();
            //ɢ������
            scattered = Ray(rec.p, target - rec.p);

            //float u = 0;
            //float v = 0;
            //get_sphere_uv(rec.normal, u, v);
            attenuation = albedo_->value(rec.u, rec.v, rec.p);
            return true;
        }
        //˥��������������ɫ)
        //vec3 albedo;
        texture* albedo_;
    };

    //�������ʣ�����+С�Ķ���)
    class Metal : public Material {
    public:
        Metal(const vec3& a, float f) : albedo(a) {
            fuzz_ = f < 1 ? f : 1;
        }
        //���Կ������Ǽ򵥼��㷴��ʸ�������ǵ�������뷨��ͬ��ʱ������false
        virtual bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered) const {
            vec3 reflected = reflect(r_in.dir, rec.normal);
            scattered = Ray(rec.p, reflected);
            attenuation = albedo;
            return scattered.dir.dotProduct(rec.normal) > 0;
        }
        vec3 albedo;
        float fuzz_;
    };

    //�����
    class Dielectrics : public Material
    {
    public:
        Dielectrics(float ri) : ref_idx(ri) {

        }

        //����ʣ��粣����)��ɢ��Ƚϸ��ӣ����ݸ���ѡ�������仹�Ƿ���
        virtual bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered) const
        {
            vec3 outward_normal;
            vec3 reflected = reflect(r_in.dir, rec.normal);
            float ni_over_nt;
            attenuation = vec3(1.0, 1.0, 0.0);
            vec3 refracted;
            float reflect_prob = 1.0;
            //���������
            float cosine;

            //ref_idx������Ϊ�ǽ��������������

            float proj = r_in.dir.dotProduct(rec.normal);
            //�����������ڲ�����������Ҫ�ȷ�ת���䷨���Ա��ڼ��� 
            if (proj > 0) {
                outward_normal = -rec.normal;
                ni_over_nt = ref_idx;
                cosine = proj / r_in.dir.getLength();
            }
            else {
                outward_normal = rec.normal;
                ni_over_nt = 1.0 / ref_idx; //���ⲿ���룬������������Ϊ���������ʵ���(��ջ������Ϊ1)
                cosine = -proj / r_in.dir.getLength();
            }
            //�������ʸ��
            if (refract(r_in.dir, outward_normal, ni_over_nt, refracted)) {
                //scattered = Ray(rec.p, refracted);
                reflect_prob = schlick(cosine, ref_idx);
            }
            //˵��reflect_prob�ӽ�1.0����Ϊ����
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

    //�������
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