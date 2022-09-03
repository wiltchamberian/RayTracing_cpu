#include "RayTracing.h"
#include "Math/Sphere_Ray.h"
#include "Hitable.h"
#include "Material.h"

#define MAX_SCATTER_NUM 3

namespace Sun {

    //��ɫ�������ڲ�����������ײ�߼� �Լ� ���߷����߼� �����ҿ��ܵݹ�
    vec3 rayTracing(const Ray& ray, Hitable* world, int depth) {
        HitRecord rec;
        //������ײ��Ϣ
        if (world->hit(ray, 0.001, MAX_flt, rec)) {
            //1st 
            //return Vector3D(rec.normal.x + 1, rec.normal.y + 1, rec.normal.z + 1) * 0.5;

            //2st
            //Vector3D target = rec.p + rec.normal + randomPointInUintSphere();
            //return color(Ray(rec.p, target - rec.p), world) * 0.5;

            //ɢ�䣨��������� )����
            Ray scattered;
            //��ʾ��ɫ��ͬʱҲ����˥������˼������ԭʼrgb���Ը�ϵ�� ���ֳ���ɫ����,Ҳ����˥��
            vec3 attenuation;
            //�������Է���
            vec3 emitted = rec.material->emitted(rec.u, rec.v, rec.p);

            //������ײ��Ϣ �� ���䷨�ߣ�����ɢ��  ����
            //depth����� ɢ�����
            if (depth < MAX_SCATTER_NUM && rec.material->scatter(ray, rec, attenuation, scattered)) {
                //ɢ��֮�󣬼�����ײ����
                return emitted + attenuation * rayTracing(scattered, world, depth + 1);
            }
            else {
                return emitted;
            }

        }
        //Vector3D ud = ray.direction_.getNormalized();
        //float t = 0.5 * (ud.y + 1.0);
        //return Vector3D(1., 1., 1) * (1.0 - t) + Vector3D(0.5, 0.7, 1.0) * t;

        return vec3(1, 1, 1);
    }

    vec3 reflect(const vec3& input, const vec3& normal) {
        return normal * (2 * (-input.dotProduct(normal))) + input;
    }

    bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
        vec3 uv = v.getNormalized();
        float dt = uv.dotProduct(n);
        //cos^2(theta2)
        float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
        if (discriminant > 0) {
            //(uv-n*dt)*ni_over_nt�ǵ�λ����ʸ����ˮƽ������-n*sqrt(discriminant)�Ǵ�ֱ����
            refracted = (uv - n * dt) * ni_over_nt - n * sqrt(discriminant);
            return true;
        }
        return false;
    }

    float schlick(float cosine, float ref_idx) {
        float r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * std::powf((1 - cosine), 5);
    }

}



