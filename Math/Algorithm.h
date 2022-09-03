/*****************************************************************************
* @brief : ��������ƽ����ص�һЩ�㷨����
* @author : acedtang
* @date : 2021/2/9
* @version : ver 1.0
* @inparam : 
* @outparam : 
*****************************************************************************/

#include "Vector3D.h"
#include "Sphere.h"

namespace Sun
{
    //�ж�������������ɵ�ƽ���ϣ��ڶ�������ָ���һ����������໹���Ҳ࣬��෵��1,�Ҳ�-1,
    //ƽ�з���0
    inline int direction(const vec3& v1, const vec3& v2) {
        return 0;
    }

    //�����һ�������ڵڶ����������ͶӰ���򳤶�
    inline float directionProject(const vec3& v1, const vec3& v2) {
        if (v2.isZero()) return 0;
        float d = v1.dotProduct(v2);
        d /= v2.getLength();
        return d;
    }

    

}

