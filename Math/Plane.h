#ifndef __PLANE_H
#define __PLANE_H

#include "Vector3D.h"

namespace Sun
{
   
    #define   PLANE_FRONT  1
    #define    PLANE_BACK  2
    #define    PLANE_BOTH  PLANE_FRONT | PLANE_BACK
    #define    PLANE_ON  4

    inline bool isSideCross(int side) {
        return (side & PLANE_BOTH) == PLANE_BOTH;
    }

    class Plane
    {
    public:
        Plane();
        //面向平面发线方向，三点为逆时针
        Plane(vec3& v1,vec3& v2,vec3& v3);
        Plane(const vec3& Normal, float Sd):normal(Normal),sd(Sd){}
        ~Plane();
        //normal vector, should always be normalized!
        vec3 normal;
        
        //原点到平面上任意一点的矢量在平面法线方向的有向投影距离
        float sd = 0.0;
        //if the point is in front of the plane,return true,else return false;
        //
        int pointSide(const vec3& v) const;
        bool isParell(const Plane& p2) const;
        float distance(const vec3& v) const;
    };
}



#endif