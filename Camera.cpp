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

    //��������Ϊ���ӵ�λ�ã�Ŀ���λ�ã�����������ֱ�Ž�(degree)����߱�,��Ȧ������
    Camera::Camera(const vec3& lookfrom, const vec3& lookat, const vec3& vup, float fov, float aspect
        , float aperture, float focus_dist, float t0, float t1) {

        time0_ = t0;
        time1_ = t1;
        lens_radius_ = aperture / 2;
        float theta = fov * A_PI / 180;
        float half_height = tan(theta / 2) * focus_dist;
        float half_width = half_height * aspect;
        origin_ = lookfrom;

        //u,v,w������λ��������������ľֲ�����Ǽܵ�x,y,z,����w��ָ��Զ��Ŀ��ķ���
        w_ = (lookfrom - lookat).getNormalized();
        u_ = vup.crossProduct(w_).getNormalized();
        v_ = w_.crossProduct(u_);

        //half_width������ǰ�ľ����������еİ��half_height*focus_dist��ǰ��
        //���lower_left_corner_���Ǿ��������������½ǵ�ȫ������
        lower_left_corner_ = origin_ - u_ * half_width - v_ * half_height - w_ * focus_dist;
        //�����������е�ˮƽ��������
        horizontal_ = u_ * (2 * half_height);
        //�����������еĴ�ֱ��������
        vertical_ = v_ * (2 * half_height);

    }

    Camera::~Camera() {


    }

    //����ƽ��(x,y,0)�ϵ�λԲ���е������
    vec3 random_in_unit_disk() {
        vec3 p;
        do {
            p = vec3(rand48(), rand48(), 0) * 2 - vec3(1, 1, 0);
        } while (p.dotProduct(p) >= 1.0);
        return p;
    }

    //�������ߴ�����ӵ� ������ ��������������ĳһ��(�ɶ���[0,1]֮���u,[0,1]֮���v ���)
    //������
    Ray Camera::buildRay(float u, float v) {
        vec3 rd = random_in_unit_disk() * lens_radius_;
        vec3 offset = u_ * rd.x + v_ * rd.y;
        //Ϊ�˲���ģ��Ч�������ܸ�origin_�Ӹ�ƫ������������������ϲ���lens_radius_ȷ��

        //�������һ��ʱ��
        float time = time0_ + rand48() * (time1_ - time0_);

        return Ray(origin_ + offset, lower_left_corner_ + horizontal_ * u + vertical_ * v - origin_ - offset, time);
    }

}