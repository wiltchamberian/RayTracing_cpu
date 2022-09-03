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
        //����ƽ�淢�߷�������Ϊ��ʱ��
        Plane(vec3& v1,vec3& v2,vec3& v3);
        Plane(const vec3& Normal, float Sd):normal(Normal),sd(Sd){}
        ~Plane();
        //normal vector, should always be normalized!
        vec3 normal;
        
        //ԭ�㵽ƽ��������һ���ʸ����ƽ�淨�߷��������ͶӰ����
        float sd = 0.0;
        //if the point is in front of the plane,return true,else return false;
        //
        int pointSide(const vec3& v) const;
        bool isParell(const Plane& p2) const;
        float distance(const vec3& v) const;
    };
}



#endif