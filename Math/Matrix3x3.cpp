#include "Matrix3x3.h"
#include <memory>

namespace Sun
{
    Matrix3x3::Matrix3x3(float e0, float e1, float e2, float e3,
        float e4, float e5, float e6, float e7,
        float e8)
    {
        entries[0] = e0;
        entries[1] = e1;
        entries[2] = e2;
        entries[3] = e3;
        entries[4] = e4;
        entries[5] = e5;
        entries[6] = e6;
        entries[7] = e7;
        entries[8] = e8;
    }

    Matrix3x3::Matrix3x3(const float* rhs)
    {
        memcpy(entries, rhs, 9 * sizeof(float));
    }

    Matrix3x3::Matrix3x3(const Matrix3x3& rhs)
    {
        memcpy(entries, rhs.entries, 9 * sizeof(float));
    }

    void Matrix3x3::setEntry(int position, float value)
    {
        if (position >= 0 && position <= 8)
            entries[position] = value;
    }

    float Matrix3x3::getEntry(int position) const
    {
        if (position >= 0 && position <= 8)
            return entries[position];
        return 0.0f;
    }

    vec3 Matrix3x3::getRow(int position) const
    {
        switch (position)
        {
        case 0:
            return vec3(entries[0], entries[1], entries[2]);
        case 1:
            return vec3(entries[3], entries[4], entries[5]);
        case 2:
            return vec3(entries[6], entries[7], entries[8]);
        }
        return vec3(0.0f, 0.0f, 0.0f);
    }

    vec3 Matrix3x3::getColumn(int position) const
    {
        switch (position)
        {
        case 0:
            return vec3(entries[0], entries[3], entries[6]);
        case 1:
            return vec3(entries[1], entries[4], entries[7]);
        case 2:
            return vec3(entries[2], entries[5], entries[8]);
        }
        return vec3(0.0f, 0.0f, 0.0f);
    }

    void Matrix3x3::loadIdentity()
    {
        memset(entries, 0, 9 * sizeof(float));
        entries[0] = 1.0f;
        entries[4] = 1.0f;
        entries[8] = 1.0f;
    }

    void Matrix3x3::loadZero()
    {
        memset(entries, 0, 9 * sizeof(float));
    }

    Matrix3x3 Matrix3x3::operator+(const Matrix3x3& rhs) const
    {
        return Matrix3x3
        (entries[0] + rhs.entries[0],
            entries[1] + rhs.entries[1],
            entries[2] + rhs.entries[2],
            entries[3] + rhs.entries[3],
            entries[4] + rhs.entries[4],
            entries[5] + rhs.entries[5],
            entries[6] + rhs.entries[6],
            entries[7] + rhs.entries[7],
            entries[8] + rhs.entries[8]);
    }

    Matrix3x3 Matrix3x3::operator-(const Matrix3x3& rhs) const
    {
        return Matrix3x3
        (entries[0] - rhs.entries[0],
            entries[1] - rhs.entries[1],
            entries[2] - rhs.entries[2],
            entries[3] - rhs.entries[3],
            entries[4] - rhs.entries[4],
            entries[5] - rhs.entries[5],
            entries[6] - rhs.entries[6],
            entries[7] - rhs.entries[7],
            entries[8] - rhs.entries[8]);
    }

    Matrix3x3 Matrix3x3::operator*(const Matrix3x3& rhs) const
    {
        Matrix3x3 m;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                int index = i * 3 + j;
                m.entries[index] = 0;
                for (int k = 0; k < 3; ++k) {
                    m.entries[index] += entries[i * 3 + k] * rhs.entries[k * 3 + j];
                }
            }
        }
        return m;
    }


    Matrix3x3 Matrix3x3::operator*(const float rhs) const
    {
        return Matrix3x3
        (entries[0] * rhs,
            entries[1] * rhs,
            entries[2] * rhs,
            entries[3] * rhs,
            entries[4] * rhs,
            entries[5] * rhs,
            entries[6] * rhs,
            entries[7] * rhs,
            entries[8] * rhs);
    }

    Matrix3x3 Matrix3x3::operator/(const float rhs) const
    {
        if (Math::equal(rhs, 0.0f) || Math::equal(rhs, 1.0f))
            return (*this);
        float temp = 1 / rhs;
        return (*this) * temp;
    }

    bool Matrix3x3::operator==(const Matrix3x3& rhs) const
    {
        for (int i = 0; i < 9; i++)
        {
            if (!Math::equal(entries[i], rhs.entries[i]))
                return false;
        }
        return true;
    }

    bool Matrix3x3::operator!=(const Matrix3x3& rhs) const
    {
        return !((*this) == rhs);
    }

    void Matrix3x3::operator+=(const Matrix3x3& rhs)
    {
        (*this) = (*this) + rhs;
    }

    void Matrix3x3::operator-=(const Matrix3x3& rhs)
    {
        (*this) = (*this) - rhs;
    }

    void Matrix3x3::operator*=(const Matrix3x3& rhs)
    {
        (*this) = (*this) * rhs;
    }

    void Matrix3x3::operator*=(const float rhs)
    {
        (*this) = (*this) * rhs;
    }

    void Matrix3x3::operator/=(const float rhs)
    {
        (*this) = (*this) / rhs;
    }

    Matrix3x3 Matrix3x3::operator-() const
    {
        Matrix3x3 result(*this);
        for (int i = 0; i < 9; i++)
            result.entries[i] = -result.entries[i];
        return result;
    }

    vec3 Matrix3x3::operator*(const vec3 rhs) const
    {
        return vec3(
            entries[0] * rhs.x
            + entries[1] * rhs.y
            + entries[2] * rhs.z,

            entries[3] * rhs.x
            + entries[4] * rhs.y
            + entries[5] * rhs.z,

            entries[6] * rhs.x
            + entries[7] * rhs.y
            + entries[8] * rhs.z);
    }

    void Matrix3x3::inverted()
    {
        *this = getInverse();
    }

    Matrix3x3 Matrix3x3::getInverseTranspose() const
    {
        Matrix3x3 result = getInverse();
        result.transpose();
        return result;
    }

    void Matrix3x3::transpose()
    {
        *this = getTranspose();
    }

    Matrix3x3 Matrix3x3::getTranspose() const
    {
        return Matrix3x3
        (entries[0], entries[3], entries[6],
            entries[1], entries[4], entries[7],
            entries[2], entries[5], entries[8]);
    }

    void Matrix3x3::invertTranspose()
    {
        *this = getInverseTranspose();
    }

    Matrix3x3 Matrix3x3::getInverse() const
    {
        Matrix3x3 m;
        float det = determinant();
        if (Math::equal(det, 0)) return m;

        m.entries[0] = entries[4] * entries[8] - entries[5] * entries[7];
        m.entries[3] = entries[5] * entries[6] - entries[3] * entries[8];
        m.entries[6] = entries[3] * entries[7] - entries[4] * entries[6];
        m.entries[1] = entries[2] * entries[7] - entries[1] * entries[8];
        m.entries[2] = entries[1] * entries[5] - entries[2] * entries[4];
        m.entries[4] = entries[0] * entries[8] - entries[2] * entries[6];
        m.entries[5] = entries[2] * entries[3] - entries[0] * entries[5];
        m.entries[7] = entries[1] * entries[6] - entries[0] * entries[7];
        m.entries[8] = entries[0] * entries[4] - entries[1] * entries[3];

        m = m / det;
        return m;
    }

    float Matrix3x3::determinant() const
    {
        float res = 0.0;
        res += entries[0] * (entries[4] * entries[8] - entries[5] * entries[7]);
        res -= entries[1] * (entries[3] * entries[8] - entries[5] * entries[6]);
        res += entries[2] * (entries[3] * entries[7] - entries[4] * entries[6]);
        return res;
    }

    const float* Matrix3x3::data() const {
        return &entries[0];
    }

    Matrix3x3 Matrix3x3::getScale(const vec3& scaleFactor)
    {
        Matrix3x3 m;
        m.entries[0] = scaleFactor.x;
        m.entries[4] = scaleFactor.y;
        m.entries[8] = scaleFactor.z;
        return m;
    }

    Matrix3x3 Matrix3x3::getRotationAxis(const double radians, const vec3& axis)
    {
        Matrix3x3 m;

        vec3 u = axis.getNormalized();

        float sinAngle = static_cast<float>(sin(radians));
        float cosAngle = static_cast<float>(cos(radians));
        float oneMinusCosAngle = 1.0f - cosAngle;

        m.entries[0] = cosAngle + oneMinusCosAngle * u.x * u.x;
        m.entries[1] = (u.x) * (u.y) * (oneMinusCosAngle)-sinAngle * u.z;
        m.entries[2] = (u.x) * (u.z) * (oneMinusCosAngle)+sinAngle * u.y;

        m.entries[3] = (u.x) * (u.y) * (oneMinusCosAngle)+sinAngle * u.z;
        m.entries[4] = cosAngle + oneMinusCosAngle * u.y * u.y;
        m.entries[5] = (u.y) * (u.z) * (oneMinusCosAngle)-sinAngle * u.x;

        m.entries[6] = (u.x) * (u.z) * (oneMinusCosAngle)-sinAngle * u.y;
        m.entries[7] = (u.y) * (u.z) * (oneMinusCosAngle)+sinAngle * u.x;
        m.entries[8] = cosAngle + oneMinusCosAngle * u.z * u.z;

        return m;
    }

    Matrix3x3 Matrix3x3::getRotationX(const double radians)
    {
        Matrix3x3 m;
        m.entries[4] = static_cast<float>(cos(radians));
        m.entries[5] = -static_cast<float>(sin(radians));
        m.entries[7] = -m.entries[5];
        m.entries[8] = m.entries[4];
        return m;
    }

    Matrix3x3 Matrix3x3::getRotationY(const double radians)
    {
        Matrix3x3 m;
        m.entries[0] = static_cast<float>(cos(radians));
        m.entries[2] = static_cast<float>(sin(radians));
        m.entries[6] = -m.entries[2];
        m.entries[8] = m.entries[0];
        return m;
    }

    Matrix3x3 Matrix3x3::getRotationZ(const double radians)
    {
        Matrix3x3 m;
        m.entries[0] = static_cast<float>(cos(radians));
        m.entries[1] = -static_cast<float>(sin(radians));
        m.entries[3] = -m.entries[1];
        m.entries[4] = m.entries[0];
        return m;
    }

}
