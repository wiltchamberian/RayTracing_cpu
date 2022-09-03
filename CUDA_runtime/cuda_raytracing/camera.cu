#include "cu_camera.h"
#include "cu_vector.h"

CCamera::CCamera() {
    lower_left_corner_ = { -2,-1,-1 };
    horizontal_ = { 4.f,0.f,0.f };
    vertical_ = { 0,2,0 };
    origin_ = { 0,0,0 };
    time0_ = 0;
    time1_ = 0;
}

//参数依次为，视点位置，目标点位置，上向量，垂直张角(degree)，宽高比,光圈，焦距
CCamera::CCamera(const cvec3& lookfrom, const cvec3& lookat, const cvec3& vup, float fov, float aspect
    , float aperture, float focus_dist, float t0, float t1) {

    time0_ = t0;
    time1_ = t1;
    lens_radius_ = aperture / 2;
    float theta = fov * A_PI / 180;
    float half_height = tanf(theta / 2) * focus_dist;
    float half_width = half_height * aspect;
    origin_ = lookfrom;

    //u,v,w三个单位向量构成了相机的局部坐标骨架的x,y,z,其中w是指向远离目标的方向
    w_ = vec3_normalize(vec3_sub(lookfrom, lookat));
    u_ = vec3_normalize(vec3_cross(vup, w_));
    v_ = vec3_cross(w_, u_);

    //half_width标记相机前的矩形像素阵列的半宽，half_height*focus_dist标记半高
    //因此lower_left_corner_就是矩形像素阵列左下角的全局坐标
    lower_left_corner_ = vec3_sub(origin_, vec3_nmul(u_, half_width));
    lower_left_corner_ = vec3_sub(lower_left_corner_, vec3_nmul(v_, half_height));
    lower_left_corner_ = vec3_sub(lower_left_corner_, vec3_nmul(w_, focus_dist));

    //矩形像素阵列的水平分量向量
    horizontal_ = vec3_nmul(u_, 2 * half_height);
    //矩形像素阵列的垂直分量向量
    vertical_ = vec3_nmul(v_ ,2 * half_height);

}

CCamera::~CCamera() {


}

//构建射线从相机视点 出发到 矩形像素阵列上某一点(由度量[0,1]之间的u,[0,1]之间的v 标记)
//的射线
__host__ __device__ CRay CCamera::buildRay(float u, float v) const {
    CRay ray;
    ray.ori.x = origin_.x;
    ray.ori.y = origin_.y;
    ray.ori.z = origin_.z;

    ray.dir = lower_left_corner_;
    ray.dir = vec3_add(ray.dir, vec3_nmul(horizontal_, u));
    ray.dir = vec3_add(ray.dir, vec3_nmul(vertical_, v));
    ray.dir = vec3_sub(ray.dir, origin_);
    
    ray.time = 0;

    return ray;
}

__host__ __device__ CRay camera_buildRay(const CCamera& camera, float u, float v) {
    CRay ray;
    ray.ori.x = camera.origin_.x;
    ray.ori.y = camera.origin_.y;
    ray.ori.z = camera.origin_.z;

    ray.dir = camera.lower_left_corner_;
    ray.dir = vec3_add(ray.dir, vec3_nmul(camera.horizontal_, u));
    ray.dir = vec3_add(ray.dir, vec3_nmul(camera.vertical_, v));
    ray.dir = vec3_sub(ray.dir, camera.origin_);
    
    ray.time = 0;

    return ray;

}