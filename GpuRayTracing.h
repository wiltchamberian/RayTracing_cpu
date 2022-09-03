#pragma once

#include <vector>
#include "Math/Vector3D.h"
#include "Math/Vector2D.h"
#include "Math/Quaternion.h"
#include "Math/Polynomial3D.h"
#include "Math/Expression.h"
#include "Math/Transform3D.h"

namespace Sun {

#define MAX_SCATTER 1
#define MAX_METRIC 100000000

    using vec3 = Vector3D<float>;
    using vec2 = Vector2D<float>;

    struct Surface {
        unsigned int num;
        Token tokens[50];
    };

    class GpuRayTracing
    {
    public:
        struct Camera {
            vec3 origin;
            vec3 lower_left_corner;
            vec3 horizontal;
            vec3 vertical;
        };

        //Camera camera;

        //球体
        struct Sphere {
            vec3 center;
            vec3 color;
            float radius;
            vec2 padding; //for alignment
        };

        struct Ray {
            vec3 origin;
            vec3 direction;
        };

        //碰撞信息结构体
        struct HitRecord {
            //标记光线碰撞点在光路上的尺度
            float t;
            //光线碰撞点
            vec3 p;
            //碰撞位置的法线
            vec3 normal;
            //碰撞物体的材质索引
            int material;  //可能废弃,glsl中材质信息数据放在物体中，直接存物体索引
            //碰撞物体索引
            int id;
            //标记碰撞点对应物体的点的u,v坐标
            float u;
            float v;

        };

        void get_sphere_uv(vec3 p,  float& u,  float& v);
        int hitRay_Sphere(int id, Ray ray, float t_min, float t_max,  HitRecord& rec);
        int hitRay_Polynomial(int id, Ray ray, float t_min, float t_max, HitRecord& rec);
        int lambertian_scatter(int id, Ray ray, HitRecord rec,vec3& attenuation, Ray& scattered);
        int specular_reflect(int id, Ray ray, HitRecord rec, vec3& attenuation, Ray& scattered);
        int rayTracingOneHit(Ray ray, HitRecord& rec);
        int rayMarchingOneHit(Ray ray, HitRecord& rec);
        float sceneDistance(vec3 p , int& id);
        float objDistance(Surface& surf, vec3 p);
        //中心差分法计算梯度 
        vec3 gradient(Surface& surf, vec3 p);
        
        vec3 rayTracing(Ray ray);

    public:
        inline float dot(vec3 a, vec3 b) {
            return a.dotProduct(b);
        }
        std::vector<Surface> sdfs;
        std::vector<Matrix4x4> sdf_transforms;
        std::vector<Matrix4x4> sdf_invTransforms;

        std::vector<Sphere> spheres;
        std::vector<Polynomial3D> polynomials;
        std::vector<Polynomial3D> partialXs;
        std::vector<Polynomial3D> partialYs;
        std::vector<Polynomial3D> partialZs;
        Camera camera;
    };

}


