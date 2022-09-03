#pragma once

#include "Math/Vector3D.h"

namespace Sun {

    class Material;
    struct HitRecord 
    {
        //标记光线碰撞点在光路上的尺度(从光线发出点到碰撞点的长度)
        float t;
        //光线碰撞点
        vec3 p;
        //碰撞位置的单位法线
        vec3 normal;
        //碰撞物体的材质
        Material* material;
        //碰撞物体的id
        int id;
        //标记碰撞点对应物体的点的u,v坐标
        float u = 0.f;
        float v = 0.f;
    };

}

