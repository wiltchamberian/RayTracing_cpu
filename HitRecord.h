#pragma once

#include "Math/Vector3D.h"

namespace Sun {

    class Material;
    struct HitRecord 
    {
        //��ǹ�����ײ���ڹ�·�ϵĳ߶�(�ӹ��߷����㵽��ײ��ĳ���)
        float t;
        //������ײ��
        vec3 p;
        //��ײλ�õĵ�λ����
        vec3 normal;
        //��ײ����Ĳ���
        Material* material;
        //��ײ�����id
        int id;
        //�����ײ���Ӧ����ĵ��u,v����
        float u = 0.f;
        float v = 0.f;
    };

}

