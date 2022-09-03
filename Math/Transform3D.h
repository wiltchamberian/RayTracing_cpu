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

    //描述物体的空间信息，物体上任意点的坐标是相对于局部坐标系，
    //经过Transform3D可以转为世界坐标
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
        //相对于父亲坐标系的某个轴旋转
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

        //lookAt矩阵表示的是由世界坐标到目标骨架坐标的变换矩阵
        static Transform3D getLookAt(vec3 cameraPos, vec3 target, vec3 worldUp);

        //之所以开放为公有原因有以下几点，1，可以支持set,get访问实际已经相当于
        //共有，2，用户直接修改不会破会对象的内部状态；
    public:
        //平移
        Vector3D_4 translation_;
        //旋转
        Quaternion rotation_;
        //放缩
        Vector3D_4 scale_;

        //Matrix4x4 _model;
    };

    //由于通常变换作用于矢量采用左乘法，所以(t2*t1)(p) = t2(t1(p))
    //每个变换都想相对于局部坐标系而言的连轴变换
    inline Transform3D operator* (const Transform3D& t2, const Transform3D& t1) {
        Transform3D result;

        result.rotation_ = t2.rotation_ * t1.rotation_;
        result.translation_ = t2.translation_ + t2.rotation_.rotate(t2.scale_ * t1.translation_);
        result.scale_ = t2.scale_ * t1.scale_;

        return result;
    }

    //由于transform既可以表示一个过程（相邻状态之间的转化关系），也可以
    //表示当前的状态，容易产生歧义；所以特意定义了一个类型来区分这两者
    using TransformState = Transform3D;
}



#endif

