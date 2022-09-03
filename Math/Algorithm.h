/*****************************************************************************
* @brief : 向量矩阵平面相关的一些算法集合
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
    //判断在两个向量组成的平面上，第二个向量指向第一个向量的左侧还是右侧，左侧返回1,右侧-1,
    //平行返回0
    inline int direction(const vec3& v1, const vec3& v2) {
        return 0;
    }

    //计算第一个向量在第二向量方向的投影有向长度
    inline float directionProject(const vec3& v1, const vec3& v2) {
        if (v2.isZero()) return 0;
        float d = v1.dotProduct(v2);
        d /= v2.getLength();
        return d;
    }

    

}

