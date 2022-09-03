#include "GpuRayTracing.h"
#include <cmath>
#include <cassert>
#include "Math/Random.h"

namespace Sun {

    void GpuRayTracing::get_sphere_uv(vec3 p, float& u,  float& v) {
        float phi = atan2(p.z, p.x); //how to make sure input not all zero?
        float theta = asin(p.y);
        u = 1 - (phi + A_PI) / (2 * A_PI);
        v = (theta + A_PI / 2) / A_PI;
    }

    int GpuRayTracing::hitRay_Sphere(int id, Ray ray, float t_min, float t_max,  HitRecord& rec) {
        //Entity ent = entities[id];
        //SphereObj sphere = sphereObjs[ent.offset];
        Sphere sphere = spheres[id];

        vec3 oc = ray.origin - sphere.center;
        float A = dot(ray.direction, ray.direction);
        float B = dot(ray.direction, oc);
        float C = dot(oc, oc) - sphere.radius * sphere.radius;
        float discriminant = B * B - A * C;
        if (discriminant > 0) {
            float sqr = sqrt(discriminant);
            float t = (-B - sqr) / A;
            if (t<t_max && t>t_min) {
                rec.t = t;
                //rec.p = ray.pointAt(rec.t);
                rec.p = ray.origin + ray.direction * rec.t;
                rec.normal = (rec.p - sphere.center) / sphere.radius;
                //rec.material = material_;
                rec.id = id;
                get_sphere_uv(rec.normal, rec.u, rec.v);
                return 1;
            }
            t = (-B + sqr) / A;
            if (t<t_max && t>t_min) {
                rec.t = t;
                rec.p = ray.origin + ray.direction * rec.t;
                rec.normal = (rec.p - sphere.center) / sphere.radius;
                //rec.material = material_;
                rec.id = id;
                get_sphere_uv(rec.normal, rec.u, rec.v);
                return 1;
            }
        }
        return 0;
    }

    int GpuRayTracing::hitRay_Polynomial(int id, Ray ray, float t_min, float t_max, HitRecord& rec) {
#if 0 //with newton iter
        Polynomial poly1({ {0,ray.origin.x,},{1,ray.direction.x} });
        Polynomial poly2({ {0,ray.origin.y},{1,ray.direction.y} });
        Polynomial poly3({ {0,ray.origin.z},{1,ray.direction.z} });

        Polynomial3D& poly3d = polynomials[id];
        Polynomial poly = poly3d.toPolynomial(poly1, poly2, poly3);

        std::complex<float> root = poly.oneRoot(t_min, 20);
        if (!equal(root.imag(), 0.f)) {
            return 0;
        }
        float t = root.real();
        //for test
        float val  =poly.value(t);
#endif 

#if 1// optimize newton iter
        Polynomial3D& poly3d = polynomials[id];
        Polynomial3D& partialX = partialXs[id];
        Polynomial3D& partialY = partialYs[id];
        Polynomial3D& partialZ = partialZs[id];
        float t = polyRoot(poly3d, partialX, partialY, partialZ, 0.5/*t_min*/, ray.origin, ray.direction, 20);
        if (isnan(t)) return 0;
#endif

#if 0  //use fix point
        Polynomial3D& poly3d = polynomials[id];
        float t = poly3d.root(t_min,ray.origin, ray.direction);
        if (isnan(t)) {
            return 0;
        }
#endif

        if (t<t_max && t>t_min) {
            rec.t = t;
            rec.p = ray.origin + ray.direction * rec.t;
            //rec.normal = (rec.p - sphere.center) / sphere.radius;
            rec.normal = poly3d.gradient(rec.p).getNormalized();

            //rec.material = material_;
            rec.id = id;
            //get_sphere_uv(rec.normal, rec.u, rec.v);
            return 1;
        }

        return 0;
    }

    int GpuRayTracing::lambertian_scatter(int id, Ray ray, HitRecord rec,  vec3& attenuation,  Ray& scattered) {
        //Entity ent = entities[id];
        //ent.material

        //散射球内随机点 (这是原始公式，个人认为可以优化到半球内)
        //vec3 randPoint = randomPointInUnitSphere();
        vec3 randPoint = vec3(0, 0, 0);
        vec3 target = rec.p + rec.normal + randPoint;//randomPointInUintSphere();
        //散射射线
        scattered.origin = rec.p;
        scattered.direction = target - rec.p;

        //float u = 0;
        //float v = 0;
        //get_sphere_uv(rec.normal, u, v);
        //attenuation = albedo_->value(rec.u, rec.v, rec.p);
        //attenuation = spheres[id].color;//ent.color;
        attenuation = vec3(0.2, 0.2, 0.8);

        return 1;
    }

    int GpuRayTracing::specular_reflect(int id, Ray ray, HitRecord rec, vec3& attenuation, Ray& scattered) {
        vec3 w = -ray.direction.getNormalized();
        float diff = /*fabs*/(w.dotProduct(rec.normal));
        vec3 r = rec.normal * diff * 2 - w;
        scattered.direction = r;
        scattered.origin = rec.p;
        if (diff < 0) {
            //assert(false);
        }
        attenuation = vec3(0.2, 0.8, 0.2) * diff;
        //attenuation = vec3(0.038, 0.155, 0.038);
        return 1;
    }

    //int hitRay_Sphere(int id, Ray ray, float t_min, float t_max, inout HitRecord rec)
    //单次光线追踪  (根据入射光  计算 hitRecord)
    int GpuRayTracing::rayTracingOneHit( Ray ray, HitRecord& rec) {

        float t_min = 0.001;
        float t_max = MAX_METRIC; // MAX_flt;
        int siz = spheres.size();
        int res = 0;
        for (int i = 0; i < siz; ++i) {
            HitRecord tmpRec;
            int hit = hitRay_Sphere(i, ray, t_min, t_max, tmpRec);
            res = res | hit;
            if (hit > 0 && (tmpRec.t <= rec.t)) {
                rec = tmpRec;
            }
        }

        int siz2 = polynomials.size();
        for (int i = 0; i < siz2; ++i) {
            HitRecord tmpRec;
            int hit = hitRay_Polynomial(i, ray, t_min, t_max, tmpRec);
            res = res | hit;
            if (hit > 0 && (tmpRec.t <= rec.t)) {
                rec = tmpRec;
            }
        }

        return res;

    }

    int GpuRayTracing::rayMarchingOneHit(Ray ray, HitRecord& rec) {
        const int MAX_ITER_NUM = 50;
        float t = 0.001;
        const float MAX_DISTANCE = 1000.0;
        const float MIN_HIT_DISTANCE = 0.001;
        float totalDis = 0;
        float dis;
        int id = 0;
        //make sure direction is normalized
        ray.direction.normalize();
        //push a small dis > MIN_HIT_DISTANE 
        ray.origin = ray.origin + ray.direction * 0.002;
        int i = 0;
        for (i = 0; i < MAX_ITER_NUM; ++i) {
            vec3 p = ray.origin + ray.direction * totalDis;
            dis =  sceneDistance(p,id);
            //p += ray.direction * dis;
            if (dis < MIN_HIT_DISTANCE) {
                rec.id = id;
                rec.p = p;
                vec4 p2 = sdf_invTransforms[id] * vec4(p.x,p.y,p.z,1);
                vec3 pp = vec3(p2.x, p2.y, p2.z);
                rec.normal = gradient(sdfs[id], pp).getNormalized();
                //将模型空间法线转为世界空间法线
#if 0
                rec.normal = sdf_transforms[id].getScale().getInverse() * rec.normal;
                rec.normal = sdf_transforms[id].getRotation().rotate(rec.normal);
                rec.normal = rec.normal.getNormalized();
#endif
                rec.normal = sdf_transforms[id] * rec.normal;
                return 1;
            }
            totalDis += dis;
            if (totalDis > MAX_DISTANCE) {
                break;
            }
        }
        return 0;
    }

    float GpuRayTracing::sceneDistance(vec3 p,int& id) {
        float dis;
        id= 0;
        dis = 1000000;
        for (int i = 0; i < sdfs.size(); ++i) {
            //先乘以该对象的模型矩阵逆，从而转化到模型空间
            vec3 q = (sdf_invTransforms[i] * vec4(p.x, p.y, p.z, 1.0)).xyz();
            //vec3 q = sdf_invTransforms[i] * p;
            float d = objDistance(sdfs[i], q);
            //如果变换保持测度不变（无scale)则d无需转换（这里先这样假设,否则需要乘以度量变换系数,how to get it? FIXME!)
            if (d < dis) {
                dis = d;
                id = i;
            }
        }
        return dis;
    }

    float GpuRayTracing::objDistance(Surface& surf, vec3 p) {
        //Surface& surf = sdfs[id];
        if (surf.num == 0) return 0;
        for (int i = 0; i < surf.num; ++i)
        {
            switch (surf.tokens[i].code) {
            case OP_ADD:
            {
                surf.tokens[i].value = surf.tokens[surf.tokens[i].left].value + surf.tokens[surf.tokens[i].right].value;
            }
            break;
            case OP_SUB:
            {
                surf.tokens[i].value = surf.tokens[surf.tokens[i].left].value - surf.tokens[surf.tokens[i].right].value;
            }
            break;
            case OP_MUL:
            {
                surf.tokens[i].value = surf.tokens[surf.tokens[i].left].value * surf.tokens[surf.tokens[i].right].value;
            }
            break;
            case OP_DIV:
            {
                surf.tokens[i].value = surf.tokens[surf.tokens[i].left].value / surf.tokens[surf.tokens[i].right].value;
            }
            break;
            case OP_POW:
            {
                surf.tokens[i].value = std::pow(surf.tokens[surf.tokens[i].left].value, surf.tokens[surf.tokens[i].right].value);
            }
            break;
            case OP_MIN:
            {
                surf.tokens[i].value = std::min(surf.tokens[surf.tokens[i].left].value, surf.tokens[surf.tokens[i].right].value);
            }
            break;
            case OP_MAX:
            {
                surf.tokens[i].value = std::max(surf.tokens[surf.tokens[i].left].value, surf.tokens[surf.tokens[i].right].value);
            }
            break;
            case OP_ATAN:
            {
                surf.tokens[i].value = atan2(surf.tokens[surf.tokens[i].left].value, surf.tokens[surf.tokens[i].right].value);
            }
            break;
            case OP_SQRT:
            {
                surf.tokens[i].value = sqrt(surf.tokens[surf.tokens[i].left].value);
            }
            break;
            case OP_ABS:
            {
                surf.tokens[i].value = abs(surf.tokens[surf.tokens[i].left].value);
            }
            break;
            case OP_SIN:
            {
                surf.tokens[i].value = sin(surf.tokens[surf.tokens[i].left].value);
            }
            break;
            case OP_COS:
            {
                surf.tokens[i].value = cos(surf.tokens[surf.tokens[i].left].value);
            }
            break;
            case OP_EXP:
            {
                surf.tokens[i].value = std::pow(A_E, surf.tokens[i].data.real);
            }
            break;
            case OP_LN:
            {
                surf.tokens[i].value = std::log(surf.tokens[i].data.real);
            }
            break;
            case OP_CLAD:
            {
                surf.tokens[i].value = surf.tokens[surf.tokens[i].left].value < surf.tokens[surf.tokens[i].right].value ? surf.tokens[surf.tokens[i].right].value : surf.tokens[surf.tokens[i].left].value;
            }
            break;
            case OP_CLAU:
            {
                surf.tokens[i].value = surf.tokens[surf.tokens[i].left].value > surf.tokens[surf.tokens[i].right].value ? surf.tokens[surf.tokens[i].right].value : surf.tokens[surf.tokens[i].left].value;
            }
            break;
            case OP_FLOAT:
            {
                surf.tokens[i].value = surf.tokens[i].data.real;
            }
            break;
            case OP_X:
            {
                surf.tokens[i].value = p.x;
            }
            break;
            case OP_Y:
            {
                surf.tokens[i].value = p.y;
            }
            break;
            case OP_Z:
            {
                surf.tokens[i].value = p.z;
            }
            break;
            case OP_SYM:
            {
                surf.tokens[i].value = surf.tokens[i].data.real;
            }
            break;
            
            
            default:
                break;
            }
        }
        return surf.tokens[surf.num - 1].value;
    }

    vec3 GpuRayTracing::gradient(Surface& surf, vec3 p) {
        float d = 0.001;
        vec3 v;
        float rd = 0.5*(1.0 / d);
        v.x = (objDistance(surf, p + vec3(d, 0, 0)) - objDistance(surf, p - vec3(d, 0, 0))) * rd;
        v.y = (objDistance(surf, p + vec3(0, d, 0)) - objDistance(surf, p - vec3(0, d, 0))) * rd;
        v.z = (objDistance(surf, p + vec3(0, 0, d)) - objDistance(surf, p - vec3(0, 0, d))) * rd;
        return v;
    }

    //光线追踪
    vec3 GpuRayTracing::rayTracing( Ray ray) {
        int i = 0;
        //太阳光
        vec3 outColor = vec3(1, 1, 1);

        Ray outRay;
        HitRecord rec;
        Ray inRay = ray;
        rec.t = MAX_METRIC;//MAX_flt;
        rec.id = 0; //为了减少if,else,这样设定有一定潜在问题，假设存在一个物体在Max_flt,
        //则会判定命中，但是因为
        for (i = 0; i < MAX_SCATTER; ++i) {
            //将入射光和场景所有物体碰撞（可利用加速结构）找到最近碰撞点信息
            //int ok = rayTracingOneHit(inRay, rec);
            int ok = rayMarchingOneHit(inRay, rec);

            if (ok == 1) {
                
                vec3 attenuation; //输出颜色
                Ray scattered;

                //根据输入光线和碰撞信息，计算散射光线
                int isScatter = specular_reflect(rec.id, inRay, rec, attenuation, scattered);
                //int isScatter = lambertian_scatter(rec.id, inRay, rec, attenuation, scattered);
                if (attenuation.y <0.5) {
                    //assert(false);
                }
                outColor = outColor * attenuation;
                if (i == 1) {
                    int x = 0; int y = x;
                }
                inRay = scattered;
            }
            else {
                ++i;
                break;
            }
        }

        return outColor;
    }


}
