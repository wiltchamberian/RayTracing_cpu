#pragma once

#include "Math/Vector3D.h"
#include "Math/Ray.h"

namespace Sun {

    class Camera
    {
    public:
        Camera();
        Camera(const vec3& lookfrom, const vec3& lookat, const vec3& vup, float fov, float aspect
            , float aperture, float focus_dist, float t0, float t1);
        ~Camera();

        Ray buildRay(float u, float v);

        vec3 origin_;
        vec3 lower_left_corner_;
        vec3 horizontal_;
        vec3 vertical_;
        vec3 u_, v_, w_;
        //Í¸¾µ°ë¾¶
        float lens_radius_;
        //ÆðÖ¹Ê±¼ä
        float time0_, time1_;

    };

}

