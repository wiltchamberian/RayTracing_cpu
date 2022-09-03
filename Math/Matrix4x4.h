#ifndef Matrix4x4_H
#define Matrix4x4_H

#include "MathUtils.h"
#include "Matrix3x3.h"
#include "Vector4D.h"
#include "Quaternion.h"

/**
 * @projectName   SoftRenderer
 * @brief         4x4 matrix class.
 * @author        YangWC
 * @date          2019-04-28
 */
namespace Sun
{ 
    class Matrix4x4
    {
    public:
        typedef float scalar;
        scalar entries[16];

        // constructors
        Matrix4x4(){loadIdentity();}
        Matrix4x4(scalar e0, scalar e1, scalar e2, scalar e3,
            scalar e4, scalar e5, scalar e6, scalar e7,
            scalar e8, scalar e9, scalar e10, scalar e11,
            scalar e12, scalar e13, scalar e14, scalar e15);
        Matrix4x4(const scalar* rhs);
        Matrix4x4(const Matrix4x4 &rhs);
        ~Matrix4x4() = default;

        // setter,getter
        void setEntry(int position, scalar value);
        scalar getEntry(int position) const;
        vec4 getRow(int position) const;
        vec4 getColumn(int position) const;
        void loadIdentity();
        void loadZero();
        Matrix3x3 toMatrix3x3() const;
        void setMatrix3x3(const Matrix3x3& mat);

        const float* data() const;

        // overloaded operators
        Matrix4x4 operator+(const Matrix4x4 & rhs) const;
        Matrix4x4 operator-(const Matrix4x4 & rhs) const;
        Matrix4x4 operator*(const Matrix4x4 & rhs) const;
        Matrix4x4 operator*(const scalar rhs) const;
        Matrix4x4 operator/(const scalar rhs) const;

        bool operator==(const Matrix4x4 & rhs) const;
        bool operator!=(const Matrix4x4 & rhs) const;

        void operator+=(const Matrix4x4 & rhs);
        void operator-=(const Matrix4x4 & rhs);
        void operator*=(const Matrix4x4 & rhs);
        void operator*=(const scalar rhs);
        void operator/=(const scalar rhs);

        Matrix4x4 operator-() const;
        Matrix4x4 operator+() const {return (*this);}
        vec4 operator*(const vec4& rhs) const;
        vec3 operator*(const vec3& rhs) const;

        float determinant() const;
        // inverse, transpose
        void inverted();
        Matrix4x4 getInverse() const;
        void transpose();
        Matrix4x4 getTranspose() const;
        void invertTranspose();
        Matrix4x4 getInverseTranspose() const;
        //this function has special meaning
        Matrix4x4 getInverseTransform() const;

        // operation on space
        static Matrix4x4 getTranslation(const vec3& translation);
        static Matrix4x4 getScale(const vec3& scaleFactor);
        static Matrix4x4 getRotationAxis(const double radians, const vec3& axis);
        static Matrix4x4 getRotationX(const double radians);
        static Matrix4x4 getRotationY(const double radians);
        static Matrix4x4 getRotationZ(const double radians);
        static Matrix4x4 getRotationEuler(const double radiansX, const double radiansY, const double radiansZ);
        static Matrix4x4 getPerspective(float fovy, float aspect, float near, float far);
        static Matrix4x4 getOrtho(float left, float right, float bottom, float top, float near, float far);
        static Matrix4x4 getLookAt(vec3 cameraPos, vec3 target, vec3 worldUp);
        static Matrix4x4 getViewPort(int left, int bottom, int width, int height);

        void decomposeTransRotScale(vec3& position, Quaternion& rotation,vec3& scale) const;
    
    };

    using mat4 = Matrix4x4;
}

#endif // Matrix4x4_H
