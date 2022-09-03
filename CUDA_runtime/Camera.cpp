#include "Camera.h"
#include "Math/Random.h"

namespace Sun {

    Camera::Camera() {
        lower_left_corner_ = { -2,-1,-1 };
        horizontal_ = { 4.f,0.f,0.f };
        vertical_ = { 0,2,0 };
        origin_ = { 0,0,0 };
        time0_ = 0;
        time1_ = 0;
    }

    //参数依次为，视点位置，目标点位置，上向量，垂直张角(degree)，宽高比,光圈，焦距
    Camera::Camera(const vec3& lookfrom, const vec3& lookat, const vec3& vup, float fov, float aspect
        , float aperture, float focus_dist, float t0, float t1) {

        time0_ = t0;
        time1_ = t1;
        lens_radius_ = aperture / 2;
        float theta = fov * A_PI / 180;
        float half_height = tan(theta / 2) * focus_dist;
        float half_width = half_height * aspect;
        origin_ = lookfrom;

        //u,v,w三个单位向量构成了相机的局部坐标骨架的x,y,z,其中w是指向远离目标的方向
        w_ = (lookfrom - lookat).getNormalized();
        u_ = vup.crossProduct(w_).getNormalized();
        v_ = w_.crossProduct(u_);

        //half_width标记相机前的矩形像素阵列的半宽，half_height*focus_dist标记半高
        //因此lower_left_corner_就是矩形像素阵列左下角的全局坐标
        lower_left_corner_ = origin_ - u_ * half_width - v_ * half_height - w_ * focus_dist;
        //矩形像素阵列的水平分量向量
        horizontal_ = u_ * (2 * half_height);
        //矩形像素阵列的垂直分量向量
        vertical_ = v_ * (2 * half_height);

    }

    Camera::~Camera() {


    }

    //返回平面(x,y,0)上单位圆盘中的随机点
    vec3 random_in_unit_disk() {
        vec3 p;
        do {
            p = vec3(rand48(), rand48(), 0) * 2 - vec3(1, 1, 0);
        } while (p.dotProduct(p) >= 1.0);
        return p;
    }

    //构建射线从相机视点 出发到 矩形像素阵列上某一点(由度量[0,1]之间的u,[0,1]之间的v 标记)
    //的射线
    Ray Camera::buildRay(float u, float v) {
        vec3 rd = random_in_unit_disk() * lens_radius_;
        vec3 offset = u_ * rd.x + v_ * rd.y;
        //为了产生模糊效果，可能给origin_加个偏移量，并由于随机数合参数lens_radius_确定

        //随机生成一个时间
        float time = time0_ + rand48() * (time1_ - time0_);

        return Ray(origin_ + offset, lower_left_corner_ + horizontal_ * u + vertical_ * v - origin_ - offset, time);
    }

}