#ifndef VECTOR_H
#define VECTOR_H

#include "MathUtils.h"

/**
 * @projectName   SoftRenderer
 * @brief         2D vector class.
 * @author        YangWC
 * @date          2019-04-28
 */
namespace Sun
{
    template<class _T>
    class Vector2D
    {
    public:
        _T x,y;

        // constructors
        Vector2D():x(0.0f), y(0.0f) {}
        Vector2D(_T newX, _T newY):x(newX), y(newY){}
        Vector2D(const _T* rhs):x(*rhs), y((*rhs)+1) {}
        Vector2D(const Vector2D<_T> & rhs):x(rhs.x), y(rhs.y){}
        ~Vector2D() = default;

        // setter,getter
        void set(_T newX, _T newY){x=newX;y=newY;	}
        void setX(_T newX) {x = newX;}
        void setY(_T newY) {y = newY;}
        _T getX() const {return x;}
        _T getY() const {return y;}

        // normalization
        void normalize();
        Vector2D<_T> getNormalize()const;

        // length
        float getLength() const { return static_cast<float>(sqrt(x*x + y*y));}
        float getSquaredLength()const{return static_cast<float>(x*x + y*y);}

        // overloaded operators
        Vector2D<_T> operator+(const Vector2D<_T>&rhs) const {return Vector2D<_T>(x + rhs.x, y + rhs.y);}
        Vector2D<_T> operator-(const Vector2D<_T>&rhs) const {return Vector2D<_T>(x - rhs.x, y - rhs.y);}
        Vector2D<_T> operator*(const _T rhs) const {return Vector2D<_T>(x*rhs, y*rhs);}
        Vector2D<_T> operator/(const _T rhs) const {return (rhs==0) ? Vector2D<_T>(0.0f, 0.0f) : Vector2D(x / rhs, y / rhs);}

        bool operator==(const Vector2D<_T>&rhs) const {return (Math::equal(x,rhs.x) && Math::equal(y,rhs.y));}
        bool operator!=(const Vector2D<_T>&rhs) const {return !((*this)==rhs);}

        void operator+=(const Vector2D<_T>&rhs){x+=rhs.x;	y+=rhs.y;}
        void operator-=(const Vector2D<_T>&rhs){x-=rhs.x;	y-=rhs.y;}
        void operator*=(const float rhs){x*=rhs;y*=rhs;}
        void operator/=(const float rhs){if(!Math::equal(rhs, 0.0)){x/=rhs;y/=rhs;}}

        Vector2D<_T> operator-() const {return Vector2D<_T>(-x, -y);}
        Vector2D<_T> operator+() const {return *this;}

        float& operator[](int i) {
            return ((float*)this)[i];
        }

        // interpolation
        Vector2D<_T> lerp(const Vector2D<_T>&v2,const float factor)const {return (*this)*(1.0f - factor) + v2*factor;}
        Vector2D<_T> quadraticInterpolate(const Vector2D<_T>& v2, const Vector2D<_T>& v3, const float factor) const
        {return (*this)*(1.0f-factor)*(1.0f-factor) + v2*2.0f*factor*(1.0f-factor) + v3*factor*factor;}

    };

    using vec2 = Vector2D<float>;

    template<class _T>
    void Vector2D<_T>::normalize()
    {
        float length;
        float scalefactor;
        length = getLength();
        if (Math::equal(length, 1.0) || Math::equal(length, 0))
            return;
        scalefactor = 1.0f / length;
        x *= scalefactor;
        y *= scalefactor;
    }

    template<class _T>
    Vector2D<_T> Vector2D<_T>::getNormalize() const
    {
        Vector2D<_T> result(*this);
        result.normalize();
        return result;
    }
}
#endif // VECTOR_H
