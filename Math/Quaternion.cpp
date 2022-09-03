#include "Quaternion.h"
#include "Vector3D.h"
#include "MathUtils.h"
#include "Matrix4x4.h"
#include <cmath> // will be replace, TODO

namespace Sun
{
    Quaternion::Quaternion()
    {
    }

    Quaternion::Quaternion(scalar w)
        :w(w)
    {
    }

    Quaternion::Quaternion(scalar x_, scalar y_, scalar z_, scalar w_)
        : x(x_)
        , y(y_)
        , z(z_)
        , w(w_)
    {
    }

    Quaternion::Quaternion(const vec3& v)
        :w(0)
        , x(v.x)
        , y(v.y)
        , z(v.z)
    {
    }

    void Quaternion::setRotateAxisAndRadius(const vec3& v, double radius)
    {
        w = cos(radius / 2);

        vec3 tmp = v;
        tmp.normalize();
        double sinAngle = sin(radius / 2);
        x = tmp.x * sinAngle;
        y = tmp.y * sinAngle;
        z = tmp.z * sinAngle;
    }

    Quaternion Quaternion::fromRotateAxisAndRadius(const vec3& v, float radius) {
        Quaternion res;
        res.w = cos(radius / 2);
        vec3 tmp = v;
        tmp.normalize();
        double Sin = sin(radius / 2);
        res.x = tmp.x * Sin;
        res.y = tmp.y * Sin;
        res.z = tmp.z * Sin;
        return res;
    }

    double Quaternion::length() const
    {
        return sqrt(x * x + y * y + z * z + w * w);
    }

    double Quaternion::getLength() const
    {
        return length();
    }

    double Quaternion::getLengthSquare() const
    {
        return x * x + y * y + z * z + w * w;
    }

    void Quaternion::normalize()
    {
        double l = length();
        x = x / l;
        y = y / l;
        z = z / l;
        w = w / l;
    }


    Quaternion Quaternion::getNormalize() const
    {
        Quaternion q(*this);
        q.normalize();
        return q;
    }

    void Quaternion::inverse()
    {
        double l = getLengthSquare();

        x = -x / l;
        y = -y / l;
        z = -z / l;
        w = w / l;
    }

    Quaternion Quaternion::getInverse() const
    {
        Quaternion q(*this);
        q.inverse();
        return q;
    }

    void Quaternion::conjugate()
    {
        x = -x;
        y = -y;
        z = -z;
    }

    Quaternion Quaternion::getConjugate() const
    {
        return Quaternion(-x, -y, -z, w);
    }

    vec3 Quaternion::toVector3D() const
    {
        return vec3(x, y, z);
    }

    Matrix4x4 Quaternion::toMatrix() const
    {
        Matrix4x4 result(
            1.0f - 2.0f * y * y - 2.0f * z * z, 2.0f * x * y - 2.0f * z * w, 2.0f * x * z + 2.0f * y * w, 0.0f,
            2.0f * x * y + 2.0f * z * w, 1.0f - 2.0f * x * x - 2.0f * z * z, 2.0f * y * z - 2.0f * x * w, 0.0f,
            2.0f * x * z - 2.0f * y * w, 2.0f * y * z + 2.0f * x * w, 1.0f - 2.0f * x * x - 2.0f * y * y, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);

        //result.transpose();
        return result;
    }

    void Quaternion::set(float x_, float y_, float z_, float w_)
    {
        x = x_; y = y_; z = z_; w = w_;
    }

    double Quaternion::dot(const Quaternion& lhs, const Quaternion& rhs)
    {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
    }

    Quaternion Quaternion::lerp(const Quaternion& a, const Quaternion& b, double t)
    {
        return Quaternion(
            (1 - t) * a.x + t * b.x,
            (1 - t) * a.y + t * b.y,
            (1 - t) * a.z + t * b.z,
            (1 - t) * a.w + t * b.w
        );
    }

    Quaternion Quaternion::slerp(const Quaternion& a, const Quaternion& b, double t)
    {
        float cos_theta = dot(a, b);

        // if B is on opposite hemisphere from A, use -B instead  what is this?
        float sign = 1.f;
        if (cos_theta < 0.f)
        {
            cos_theta = -cos_theta;
            sign = -1.f;
        }

        float c1, c2;
 
        if (cos_theta > 1.f - 0.000001f)
        {
            // if q2 is (within precision limits) the same as q1,
            // just linear interpolate between A and B
            //c2 = t;
            //c1 = 1.f - t;
            return a;
        }
        else
        {
            float theta = acos(cos_theta);
            float sin_theta = sin(theta);
            float t_theta = t * theta;
            float inv_sin_theta = 1.f / sin_theta;
            c2 = sin(t_theta) * inv_sin_theta;
            c1 = sin(theta - t_theta) * inv_sin_theta;
        }

        c2 *= sign;
        // interpolate
        return Quaternion(
            a.x * c1 + b.x * c2,
            a.y * c1 + b.y * c2,
            a.z * c1 + b.z * c2,
            a.w * c1 + b.w * c2);
    }

    Quaternion Quaternion::createByNormalizedDirection(const vec3& d1, const vec3& d2) {
        vec3 axis = d1.crossProduct(d2);
        float radius = acos(d1.dotProduct(d2));
        Quaternion qua;
        qua.setRotateAxisAndRadius(axis, radius);
        return qua;
    }

    Quaternion Quaternion::operator*(double s)
    {
        return Quaternion(x * s, y * s, z * s, w * s);
    }

    Quaternion operator + (const Quaternion& lhs, const Quaternion& rhs)
    {
        Quaternion q;
        q.x = lhs.x + rhs.x;
        q.y = lhs.y + rhs.y;
        q.z = lhs.z + rhs.z;
        q.w = lhs.w + rhs.w;
        return q;
    }

    Quaternion operator - (const Quaternion& lhs, const Quaternion& rhs)
    {
        Quaternion q;
        q.x = lhs.x - rhs.x;
        q.y = lhs.y - rhs.y;
        q.z = lhs.z - rhs.z;
        q.w = lhs.w - rhs.w;
        return q;
    }

    //make sure (lhs*rhs).rotate(v) == lhs.rotate(rhs.rotate(v)) TODO
    Quaternion operator *(const Quaternion& lhs, const Quaternion& rhs)
    {
        double w1 = lhs.w;
        double w2 = rhs.w;
        vec3 v1(lhs.x, lhs.y, lhs.z);
        vec3 v2(rhs.x, rhs.y, rhs.z);
        double w3 = w1 * w2 - v1.dotProduct(v2);
        vec3 v3 = v1.crossProduct(v2) + v2 * w1 + v1 * w2;
        return Quaternion(v3.x, v3.y, v3.z, w3);
    }

    //旋转的具体公式是计算: qvq^(-1)
    vec3 Quaternion::rotate(const vec3& v) const
    {
        // Extract the vector part of the quaternion
        vec3 u(x, y, z);
        // qvq^(-1)
        return u * 2.0f * u.dotProduct(v)
            + v * (w * w - u.dotProduct(u))
            + u.crossProduct(v) * 2.0f * w;
    }

}
