#ifndef VECTOR4D_H
#define VECTOR4D_H

#include "MathUtils.h"
#include "Vector3D.h"

/**
 * @projectName   SoftRenderer
 * @brief         4D vector class.
 * @author        YangWC
 * @date          2019-04-28
 */
namespace Sun
{
    template<class _T>
    class Vector4D
    {
    public:
        _T x,y,z,w;

        // constructors
        Vector4D():x(0.0f), y(0.0f), z(0.0f), w(0.0f){}
        Vector4D(_T newX, _T newY, _T newZ, _T newW):x(newX), y(newY), z(newZ), w(newW){}
        Vector4D(const float * rhs):x(*rhs), y(*(rhs+1)), z(*(rhs+2)), w(*(rhs+3)){}
        Vector4D(const Vector4D<_T> &rhs):x(rhs.x), y(rhs.y), z(rhs.z), w(rhs.w){}
        Vector4D(const Vector3D<_T> & rhs):	x(rhs.x), y(rhs.y), z(rhs.z), w(1.0f){}
        Vector4D(const Vector3D<_T>& rhs, _T newW) :x(rhs.x), y(rhs.y), z(rhs.z), w(newW) {}
        ~Vector4D() = default;

        // setter,getter
        void set(_T newX, _T newY, _T newZ, _T newW){x=newX;y=newY;z=newZ;w=newW;}
        void setX(_T newX) {x = newX;}
        void setY(_T newY) {y = newY;}
        void setZ(_T newZ) {z = newZ;}
        void setW(_T newW) {w = newW;}
        _T getX() const {return x;}
        _T getY() const {return y;}
        _T getZ() const {return z;}
        _T getW() const {return w;}
        Vector3D<_T> xyz() const { return Vector3D<_T>(x, y, z); };
        Vector3D<_T> toVector3D() const { return Vector3D<_T>(x, y, z); };

        // product
        _T dotProduct(const Vector4D<_T> &rhs) const {return x*rhs.x + y*rhs.y + z*rhs.z + w*rhs.w;}

        // linear interpolation
        Vector4D<_T> lerp(const Vector4D<_T>&v2, _T factor) const {return (*this)*(1.0f-factor) + v2*factor;}
        Vector4D<_T> QuadraticInterpolate(const Vector4D<_T>&v2, const Vector4D<_T>&v3, _T factor) const
        {return (*this)*(1.0f-factor)*(1.0f-factor)+v2*2.0f*factor*(1.0f-factor)+v3*factor*factor;}

        // overloaded operators
        Vector4D<_T> operator*(const float rhs) const {return Vector4D<_T>(x*rhs, y*rhs, z*rhs, w*rhs);}
        Vector4D<_T> operator/(const float rhs) const
        {return (Math::equal(rhs,0.0f))?Vector4D(0.0f, 0.0f, 0.0f, 0.0f): Vector4D<_T>(x/rhs, y/rhs, z/rhs, w/rhs);}

        //user make sure the i>=0 and <4!for performance,this func do not check
        float operator[](int i) const {
            return *((&x) + i);
        }

        bool operator==(const Vector4D<_T>&rhs) const {return (Math::equal(x,rhs.x)&& Math::equal(y,rhs.y)&& Math::equal(z,rhs.z)&& Math::equal(w,rhs.w));}
        bool operator!=(const Vector4D<_T>&rhs) const {return !((*this)==rhs);}

        void operator+=(const Vector4D<_T>&rhs) {x+=rhs.x;y+=rhs.y;z+=rhs.z;w+=rhs.w;}
        void operator-=(const Vector4D<_T>& rhs) {x-=rhs.x;y-=rhs.y;z-=rhs.z;w-=rhs.w;}
        void operator*=(const float rhs){x*=rhs;y*=rhs;z*=rhs;w*=rhs;}
        void operator/=(const float rhs){if(!Math::equal(rhs,0.0f)){x/=rhs; y/=rhs; z/=rhs; w/=rhs;}}

        Vector4D<_T> operator-() const {return Vector4D<_T>(-x, -y, -z, -w);}
        Vector4D<_T> operator+() const {return *this;}

        float& operator[](int i) {
            return ((float*)this)[i];
        }

        friend Vector4D<_T> operator+(const Vector4D<_T>& l, const Vector4D<_T>& r)
        {
            return Vector4D<_T>(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w);
        }
        friend Vector4D<_T> operator-(const Vector4D<_T>& l, const Vector4D<_T>& r)
        {
            return Vector4D<_T>(l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w);
        }
    };

    using vec4 = Vector4D<float>;

    using rgba = vec4;

    extern const vec4 g_defaultColor;

}

#endif // VECTOR4D_H
