#include "Plane.h"

namespace Sun {

    Plane::Plane()
    {

    }

    Plane::~Plane()
    {

    }

    //面向平面发线方向，三点为逆时针
    Plane::Plane(vec3& v1, vec3& v2, vec3& v3)
    {
        normal = (v2 - v1).crossProduct(v3 - v1);
        normal.normalize();
        sd = v1.dotProduct(normal);
    }

    int Plane::pointSide(const vec3& v) const
    {
        vec3 u = v - normal * sd;
        float c = u.dotProduct(normal);
		if (c > 0)
			return PLANE_FRONT;
		else if (c < 0)
			return PLANE_BACK;
		else
			return PLANE_ON;

        return PLANE_ON;
    }

    bool Plane::isParell(const Plane& p2) const
    {
        return parallel(normal, p2.normal);
    }

    float Plane::distance(const vec3& v) const
    {
        vec3 v_plane = normal * sd;
        vec3 v_ = v - v_plane;
        float d = v_.dotProduct(normal);
        d = d < 0 ? (-d) : d;
        return d;
    }
}

