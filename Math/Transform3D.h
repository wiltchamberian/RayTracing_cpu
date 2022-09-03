#ifndef __TRANSFORM_H
#define __TRANSFORM_H

#include "Vector3D.h"
#include "Matrix4x4.h"
#include "Quaternion.h"

namespace Sun
{
    struct RigidTransform {
        vec3 translate;
        Quaternion qua;
    };

    //��������Ŀռ���Ϣ������������������������ھֲ�����ϵ��
    //����Transform3D����תΪ��������
    class Transform3D
    {
    public:
        Transform3D();
        Transform3D(const vec3& translation, const Quaternion& quater,
            const vec3& scale);
        ~Transform3D();
        void translate(const vec3& world_space_v);
        Transform3D getTranslate(const vec3 v);
        void scale(const vec3& scale);
        void scale(float x, float y, float z);
        //this rotate is relative the local coordinate
        void rotate(const vec3& axis, double radians);
        //����ڸ�������ϵ��ĳ������ת
        //void rotateRelatvieParent(const Vector3D& axis, double radians);
        Matrix4x4 toMatrix() const;

        vec3 z_direction() const;
        vec3 x_direction() const;
        vec3 y_direction() const;

        void setScale(const vec3& s);
        void setRotation(const Quaternion& r);
        void setRotation(float a, float b, float c, float d);
        void setTranslation(const vec3& t);
        void setMatrix(const Matrix4x4& matrix);

        // Transformation getter.
        vec3 getTranslation() const { return translation_; }
        Quaternion getRotation() const { return rotation_; }
        vec3 getScale() const { return scale_; }

        vec3 operator* (const vec3& p) const;

        inline friend Transform3D operator * (const Transform3D& t2, const Transform3D& t1);

        Transform3D getInverse();

        //lookAt�����ʾ�������������굽Ŀ��Ǽ�����ı任����
        static Transform3D getLookAt(vec3 cameraPos, vec3 target, vec3 worldUp);

        //֮���Կ���Ϊ����ԭ�������¼��㣬1������֧��set,get����ʵ���Ѿ��൱��
        //���У�2���û�ֱ���޸Ĳ����ƻ������ڲ�״̬��
    public:
        //ƽ��
        Vector3D_4 translation_;
        //��ת
        Quaternion rotation_;
        //����
        Vector3D_4 scale_;

        //Matrix4x4 _model;
    };

    //����ͨ���任������ʸ��������˷�������(t2*t1)(p) = t2(t1(p))
    //ÿ���任��������ھֲ�����ϵ���Ե�����任
    inline Transform3D operator* (const Transform3D& t2, const Transform3D& t1) {
        Transform3D result;

        result.rotation_ = t2.rotation_ * t1.rotation_;
        result.translation_ = t2.translation_ + t2.rotation_.rotate(t2.scale_ * t1.translation_);
        result.scale_ = t2.scale_ * t1.scale_;

        return result;
    }

    //����transform�ȿ��Ա�ʾһ�����̣�����״̬֮���ת����ϵ����Ҳ����
    //��ʾ��ǰ��״̬�����ײ������壻�������ⶨ����һ������������������
    using TransformState = Transform3D;
}



#endif

