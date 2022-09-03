/**
 * @projectName   SoftRenderer
 * @brief         3D vector class.
 * @author        YangWC , revised by ACE
 * @date          2019-04-28
 */

#ifndef VECTOR3D_H
#define VECTOR3D_H

#include "MathUtils.h"

namespace Sun
{
    template<class _T>
    class Vector3D
    {
    public:
        _T x,y,z;

        // constructors
        Vector3D():x(0), y(0), z(0){}
        Vector3D(_T newX, _T newY, _T newZ):x(newX), y(newY), z(newZ){}
        Vector3D(const _T* rhs):x(*rhs), y(*(rhs+1)), z(*(rhs+2)){}
        Vector3D(const Vector3D &rhs):x(rhs.x), y(rhs.y), z(rhs.z){}
        Vector3D(_T value) :x(value), y(value), z(value) {}
        ~Vector3D() = default;

        const _T* data() const {
            return &x;
        }

        static Vector3D<_T> fromEulerianAngles(float yaw, float pitch) {
            Vector3D<_T> v;
            v.z = cos(pitch)*cos(yaw);
            v.x = cos(pitch) * sin(yaw);
            v.y = sin(pitch);
            return v;
        }

        void clear() { x = 0.f; y = 0.f; z = 0.f; }
        // setter,getter
        void set(_T newX, _T newY, _T newZ){x=newX;y=newY;z=newZ;}
        void setX(_T newX) {x = newX;}
        void setY(_T newY) {y = newY;}
        void setZ(_T newZ) {z = newZ;}
        float getX() const {return x;}
        float getY() const {return y;}
        float getZ() const {return z;}

        bool isZero() const { return Math::equal(x, 0) && Math::equal(y, 0) && Math::equal(z, 0); }
        // normalization
        void normalize();
        Vector3D getNormalized() const;
        Vector3D sign() const;
        Vector3D getAbs() const;
        Vector3D& abs();
        Vector3D getInverse() const;

        // length caculation
        float getLength() const {return static_cast<_T>(sqrt(x*x+y*y+z*z));}
        float getSquaredLength() const {return x*x+y*y+z*z;}

        // product
        float dotProduct(const Vector3D<_T>&rhs) const {return x*rhs.x + y*rhs.y + z*rhs.z;}
        Vector3D<_T> crossProduct(const Vector3D<_T>&rhs) const
        {return Vector3D<_T>(y*rhs.z - z*rhs.y, z*rhs.x - x*rhs.z, x*rhs.y - y*rhs.x);}

        // linear interpolation
        Vector3D<_T> lerp(const Vector3D<_T>&v2, float factor) const {return (*this)*(1.0f-factor) + v2*factor;}
        Vector3D<_T> QuadraticInterpolate(const Vector3D<_T>&v2, const Vector3D<_T>&v3, float factor) const
        {return (*this)*(1.0f-factor)*(1.0f-factor) + v2*2.0f*factor*(1.0f-factor) + v3*factor*factor;}

        // overloaded operators
        Vector3D<_T> operator+(const Vector3D<_T> &rhs) const {return Vector3D<_T>(x + rhs.x, y + rhs.y, z + rhs.z);}
        Vector3D<_T> operator-(const Vector3D<_T>&rhs) const {return Vector3D<_T>(x - rhs.x, y - rhs.y, z - rhs.z);}
        Vector3D<_T> operator*(const float rhs) const {return Vector3D<_T>(x*rhs, y*rhs, z*rhs);}
        Vector3D<_T> operator/(const float rhs) const {return (Math::equal(rhs,0.0f))? Vector3D<_T>(0.0f, 0.0f, 0.0f):Vector3D(x/rhs, y/rhs, z/rhs);}

        bool operator==(const Vector3D<_T>&rhs) const {return (Math::equal(x,rhs.x) && Math::equal(y,rhs.y) && Math::equal(z,rhs.z));}
        bool operator!=(const Vector3D<_T>&rhs) const {return !((*this)==rhs);}

        void operator+=(const Vector3D<_T>&rhs) {x+=rhs.x;y+=rhs.y;z+=rhs.z;}
        void operator-=(const Vector3D<_T>& rhs) {x-=rhs.x;y-=rhs.y;z-=rhs.z;}
        void operator*=(const float rhs){x*=rhs;y*=rhs;z*=rhs;}
        void operator/=(const float rhs){if(!Math::equal(rhs,0.0f)){x/=rhs; y/=rhs; z/=rhs;}}

        Vector3D<_T> operator-() const {return Vector3D<_T>(-x, -y, -z);}
        Vector3D<_T> operator+() const {return *this;}

        //user make sure the i>=0 and <3!for performance,this func do not check
        float operator[](int i) const {
            return *((&x) + i);
        }

        friend bool operator<(const Vector3D& v1, const Vector3D& v2) {
            return v1.x < v2.x 
                || (v1.x==v2.x)&&(v1.y < v2.y) 
                || (v1.x==v2.x)&&(v1.y==v2.y)&&(v1.z<v2.z);
        }

        float& operator[](int i) {
            return ((float*)this)[i];
        }

        //inline friend Vector3D<_T> operator* (const Vector3D<_T>& v1, const Vector3D<_T>& v2);
    };

    using vec3 = Vector3D<float>;
    using ivec3 = Vector3D<int>;
    using pt3 = Vector3D<float>;

    //x,y,z轴单位矢量
    extern const vec3 g_uint_vector[3];
#define UINT_VECTOR(i) g_unit_vector[i&3]
#define UINT_X g_unit_vector[0]
#define UINT_Y g_unit_vector[1]
#define UINT_Z g_unit_vector[2]

    //判断两个vector是否平行，0向量与所有向量平行
    template<class _T>
    inline bool parallel(const Vector3D<_T>& v1, const Vector3D<_T>& v2) {
        return Math::equal(v1.x * v2.y, v2.x * v1.y) && Math::equal(v1.x * v2.z, v1.z * v2.x);
    }

    template<class _T>
    Vector3D<_T> operator* (const Vector3D<_T>& v1, const Vector3D<_T>& v2) {
        return Vector3D<_T>(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
    }

    inline float includedAngle(const vec3& v1, const vec3& v2) {
        float dot = v1.dotProduct(v2);
        float l1 = v1.getLength();
        float l2 = v2.getLength();
        float cos = dot / l1 / l2;
        return acos(cos);
    }

    template<class _T>
    void Vector3D<_T>::normalize()
    {
        float length = getLength();
        if (Math::equal(length, 1.0f) || Math::equal(length, 0.0f))
            return;
        float scalefactor = 1.0f / length;
        x *= scalefactor;
        y *= scalefactor;
        z *= scalefactor;
    }

    template<class _T>
    Vector3D<_T> Vector3D<_T>::getNormalized() const
    {
        Vector3D<_T> result(*this);
        result.normalize();
        return result;
    }

    template<class _T>
    Vector3D<_T> Vector3D<_T>::sign() const {
        return Vector3D(x >= 0 ? 1 : 0, y >= 0 ? 1 : 0, z >= 0 ? 1 : 0);
    }

    template<class _T>
    Vector3D<_T> Vector3D<_T>::getAbs() const {
        return Vector3D(fabs(x), fabs(y), fabs(z));
    }

    template<class _T>
    Vector3D<_T>& Vector3D<_T>::abs() {
        x = fabs(x);
        y = fabs(y);
        z = fabs(z);
        return *this;
    }

    template<class _T>
    Vector3D<_T> Vector3D<_T>::getInverse() const {
        return Vector3D<_T>(1.0 / x, 1.0 / y, 1.0 / z);
    }

    //占用4个字节的Vector3D,主要是为了GPU Compatibility
    class Vector3D_4 : public Vector3D<float>
    {
    public:
        Vector3D_4() :Vector3D() {}
        Vector3D_4(float x, float y, float z) :Vector3D(x, y, z) {}
        Vector3D_4& operator = (const Vector3D& v) {
            x = v.x;
            y = v.y;
            z = v.z;
            return *this;
        }
        float w = 0.0;
    };

}

#endif // VECTOR3D_H
