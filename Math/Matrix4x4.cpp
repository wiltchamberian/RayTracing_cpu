#include "Matrix4x4.h"
#include <memory>

namespace Sun
{

    Matrix4x4::Matrix4x4(scalar e0, scalar e1, scalar e2, scalar e3,
        scalar e4, scalar e5, scalar e6, scalar e7,
        scalar e8, scalar e9, scalar e10, scalar e11,
        scalar e12, scalar e13, scalar e14, scalar e15)
    {
        entries[0]  = e0;
        entries[1]  = e1;
        entries[2]  = e2;
        entries[3]  = e3;
        entries[4]  = e4;
        entries[5]  = e5;
        entries[6]  = e6;
        entries[7]  = e7;
        entries[8]  = e8;
        entries[9]  = e9;
        entries[10] = e10;
        entries[11] = e11;
        entries[12] = e12;
        entries[13] = e13;
        entries[14] = e14;
        entries[15] = e15;
    }

    Matrix4x4::Matrix4x4(const scalar* rhs)
    {
        memcpy(entries, rhs, 16*sizeof(scalar));
    }

    Matrix4x4::Matrix4x4(const Matrix4x4 &rhs)
    {
        memcpy(entries, rhs.entries, 16*sizeof(scalar));
    }

    void Matrix4x4::setEntry(int position, scalar value)
    {
        if(position>=0 && position<=15)
            entries[position]=value;
    }

    Matrix4x4::scalar Matrix4x4::getEntry(int position) const
    {
        if(position>=0 && position<=15)
            return entries[position];
        return 0.0f;
    }

    vec4 Matrix4x4::getRow(int position) const
    {
        switch(position)
        {
        case 0:
            return vec4(entries[0], entries[1], entries[2], entries[3]);
        case 1:
            return vec4(entries[4], entries[5], entries[6], entries[7]);
        case 2:
            return vec4(entries[8], entries[9], entries[10], entries[11]);
        case 3:
            return vec4(entries[12], entries[13], entries[14], entries[15]);
        }
        return vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    vec4 Matrix4x4::getColumn(int position) const
    {
        switch(position)
        {
        case 0:
            return vec4(entries[0], entries[4], entries[8], entries[12]);
        case 1:
            return vec4(entries[1], entries[5], entries[9], entries[13]);
        case 2:
            return vec4(entries[2], entries[6], entries[10], entries[14]);
        case 3:
            return vec4(entries[3], entries[7], entries[11], entries[15]);
        }
        return vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    void Matrix4x4::loadIdentity()
    {
        memset(entries, 0, 16*sizeof(scalar));
        entries[0]  = 1.0f;
        entries[5]  = 1.0f;
        entries[10] = 1.0f;
        entries[15] = 1.0f;
    }

    void Matrix4x4::loadZero()
    {
        memset(entries, 0, 16*sizeof(scalar));
    }

    Matrix3x3 Matrix4x4::toMatrix3x3() const
    {
        Matrix3x3 m;
        m.entries[0] = entries[0];
        m.entries[1] = entries[1];
        m.entries[2] = entries[2];
        m.entries[3] = entries[4];
        m.entries[4] = entries[5];
        m.entries[5] = entries[6];
        m.entries[6] = entries[8];
        m.entries[7] = entries[9];
        m.entries[8] = entries[10];
        return m;
    }

    void Matrix4x4::setMatrix3x3(const Matrix3x3& mat)
    {
        entries[0] = mat.entries[0];
        entries[1] = mat.entries[1];
        entries[2] = mat.entries[2];
        entries[4] = mat.entries[3];
        entries[5] = mat.entries[4];
        entries[6] = mat.entries[5];
        entries[8] = mat.entries[6];
        entries[9] = mat.entries[7];
        entries[10] = mat.entries[8];
    }

    const float* Matrix4x4::data() const {
        return &entries[0];
    }

    Matrix4x4 Matrix4x4::operator+(const Matrix4x4 &rhs) const
    {
        return Matrix4x4
                (entries[0]  + rhs.entries[0],
                entries[1]  + rhs.entries[1],
                entries[2]  + rhs.entries[2],
                entries[3]  + rhs.entries[3],
                entries[4]  + rhs.entries[4],
                entries[5]  + rhs.entries[5],
                entries[6]  + rhs.entries[6],
                entries[7]  + rhs.entries[7],
                entries[8]  + rhs.entries[8],
                entries[9]  + rhs.entries[9],
                entries[10] + rhs.entries[10],
                entries[11] + rhs.entries[11],
                entries[12] + rhs.entries[12],
                entries[13] + rhs.entries[13],
                entries[14] + rhs.entries[14],
                entries[15] + rhs.entries[15]);
    }

    Matrix4x4 Matrix4x4::operator-(const Matrix4x4 &rhs) const
    {
        return Matrix4x4
                (entries[0]  - rhs.entries[0],
                entries[1]  - rhs.entries[1],
                entries[2]  - rhs.entries[2],
                entries[3]  - rhs.entries[3],
                entries[4]  - rhs.entries[4],
                entries[5]  - rhs.entries[5],
                entries[6]  - rhs.entries[6],
                entries[7]  - rhs.entries[7],
                entries[8]  - rhs.entries[8],
                entries[9]  - rhs.entries[9],
                entries[10] - rhs.entries[10],
                entries[11] - rhs.entries[11],
                entries[12] - rhs.entries[12],
                entries[13] - rhs.entries[13],
                entries[14] - rhs.entries[14],
                entries[15] - rhs.entries[15]);
    }

    Matrix4x4 Matrix4x4::operator*(const Matrix4x4 &rhs) const
    {
        Matrix4x4 res;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int index = i * 4 + j;
                res.entries[index] = 0;
                for (int k = 0; k < 4; ++k) {
                    scalar t1 = entries[i * 4 + k];
                    scalar t2 = rhs.entries[k * 4 + j];
                    scalar t = t1 * t2;
                    res.entries[index] += t;
                }
            }
        }
        return res;
    }

    Matrix4x4 Matrix4x4::operator*(const scalar rhs) const
    {
        return Matrix4x4
                (entries[0]*rhs,
                entries[1]*rhs,
                entries[2]*rhs,
                entries[3]*rhs,
                entries[4]*rhs,
                entries[5]*rhs,
                entries[6]*rhs,
                entries[7]*rhs,
                entries[8]*rhs,
                entries[9]*rhs,
                entries[10]*rhs,
                entries[11]*rhs,
                entries[12]*rhs,
                entries[13]*rhs,
                entries[14]*rhs,
                entries[15]*rhs);
    }

    Matrix4x4 Matrix4x4::operator/(const scalar rhs) const
    {
        if (Math::equal(rhs,0.0f) || Math::equal(rhs,1.0f))
            return (*this);
        double temp = 1 / rhs;
        return (*this)*temp;
    }

    bool Matrix4x4::operator==(const Matrix4x4 &rhs) const
    {
        for(int i=0; i<16; i++)
        {
            if(!Math::equal(entries[i],rhs.entries[i]))
                return false;
        }
        return true;
    }

    bool Matrix4x4::operator!=(const Matrix4x4 &rhs) const
    {
        return !((*this) == rhs);
    }

    void Matrix4x4::operator+=(const Matrix4x4 &rhs)
    {
        (*this) = (*this)+rhs;
    }

    void Matrix4x4::operator-=(const Matrix4x4 &rhs)
    {
        (*this) = (*this)-rhs;
    }

    void Matrix4x4::operator*=(const Matrix4x4 &rhs)
    {
        (*this) = (*this)*rhs;
    }

    void Matrix4x4::operator*=(const scalar rhs)
    {
        (*this) = (*this)*rhs;
    }

    void Matrix4x4::operator/=(const scalar rhs)
    {
        (*this) = (*this)/rhs;
    }

    Matrix4x4 Matrix4x4::operator-() const
    {
        Matrix4x4 result(*this);
        for(int i=0; i<16; i++)
            result.entries[i] = -result.entries[i];
        return result;
    }

    vec4 Matrix4x4::operator*(const vec4& rhs) const
    {
        return vec4(
                    entries[0]*rhs.x
                +	entries[1]*rhs.y
                +	entries[2]*rhs.z
                +	entries[3]*rhs.w,

                entries[4]*rhs.x
                +	entries[5]*rhs.y
                +	entries[6]*rhs.z
                +	entries[7]*rhs.w,

                entries[8]*rhs.x
                +	entries[9]*rhs.y
                +	entries[10]*rhs.z
                +	entries[11]*rhs.w,

                entries[12]*rhs.x
                +	entries[13]*rhs.y
                +	entries[14]*rhs.z
                +	entries[15]*rhs.w);
    }

    vec3 Matrix4x4::operator*(const vec3& rhs) const
    {
        return vec3(
            entries[0] * rhs.x
            + entries[1] * rhs.y
            + entries[2] * rhs.z,

            entries[4] * rhs.x
            + entries[5] * rhs.y
            + entries[6] * rhs.z,

            entries[8] * rhs.x
            + entries[9] * rhs.y
            + entries[10] * rhs.z);
    }

    float Matrix4x4::determinant() const {

//#define ENTRIES(A,B,C,D) (entries[A]*entries[B]-entries[C]*entries[D])

        double d10_15_11_14 = entries[10]*entries[15]-entries[11]*entries[14];
        double d9_15_11_13 = entries[9] * entries[15]-entries[11]*entries[13];
        double d9_14_10_13 = entries[9]*entries[14]-entries[10]*entries[13];
        double d8_15_11_12 = entries[8]*entries[15]-entries[11]*entries[12];
        double d8_14_10_12 = entries[8]*entries[14]-entries[10]*entries[12];
        double d8_13_9_12 = entries[8]*entries[13]-entries[9]*entries[12];

        double res = 0;
        res += entries[0] * (entries[5] * d10_15_11_14 - entries[6] * d9_15_11_13 + entries[7] * d9_14_10_13);
        res -= entries[1] * (entries[4] * d10_15_11_14 - entries[6] * d8_15_11_12 + entries[7] * d8_14_10_12);
        res += entries[2] * (entries[4] * d9_15_11_13 - entries[5] * d8_15_11_12 + entries[7] * d8_13_9_12);
        res -= entries[3] * (entries[4] * d9_14_10_13 - entries[5] * d8_14_10_12 + entries[6] * d8_13_9_12);

        return res;
    }

    void Matrix4x4::inverted()
    {
        *this = getInverse();
    }

    Matrix4x4 Matrix4x4::getInverse() const
    {
        Matrix4x4 m;
//#define ENTRIES(A,B,C,D) (entries[A]*entries[B]-entries[C]*entries[D])

        scalar d10_15_11_14 = entries[10]*entries[15]-entries[11]*entries[14];
        scalar d9_15_11_13 = entries[9]*entries[15]-entries[11]*entries[13];
        scalar d9_14_10_13 = entries[9]*entries[14]-entries[10]*entries[13];
        scalar d8_15_11_12 = entries[8]*entries[15]-entries[11]*entries[12];
        scalar d8_14_10_12 = entries[8]*entries[14]-entries[10]*entries[12];
        scalar d8_13_9_12 = entries[8]*entries[13]-entries[9]*entries[12];

        scalar d2_15_3_14 = entries[2]*entries[15]-entries[3]*entries[14];
        scalar d2_11_3_10 = entries[2]*entries[11]-entries[3]*entries[10];
        scalar d6_15_7_14 = entries[6]*entries[15]-entries[7]*entries[14];
        scalar d2_7_3_6 = entries[2]*entries[7]-entries[3]*entries[6];

        scalar d6_11_7_10 = entries[6]*entries[11]-entries[7]*entries[10];
        scalar d4_11_7_8 = entries[4]*entries[11]-entries[7]*entries[8];
        scalar d4_10_6_8 = entries[4]*entries[10]-entries[6]*entries[8];

        scalar d5_15_7_13 = entries[5]*entries[15]-entries[7]*entries[13];
        scalar d4_15_7_12 = entries[4]*entries[15]-entries[7]*entries[12];
        scalar d4_13_5_12 = entries[4]*entries[13]-entries[5]*entries[12];

        scalar d5_11_7_9 = entries[5]*entries[11]-entries[7]*entries[9];
        scalar d4_9_5_8 = entries[4]*entries[9]-entries[5]*entries[8];

        scalar d5_14_6_13 = entries[5]*entries[14]-entries[6]*entries[13];
        scalar d4_14_6_12 = entries[4]*entries[14]-entries[6]*entries[12];

        scalar d5_10_6_9 = entries[5]*entries[10]-entries[6]*entries[9];

        m.entries[0] = entries[5] * d10_15_11_14 - entries[6] * d9_15_11_13 + entries[7] * d9_14_10_13;
        m.entries[4] = -(entries[4] * d10_15_11_14 - entries[6] * d8_15_11_12 + entries[7] * d8_14_10_12);
        m.entries[8] = entries[4] * d9_15_11_13 - entries[5] * d8_15_11_12 + entries[7] * d8_13_9_12;
        m.entries[12] = -(entries[4] * d9_14_10_13 - entries[5] * d8_14_10_12 + entries[6] * d8_13_9_12);
        m.entries[1] = -(entries[1] * d10_15_11_14 - entries[9] * d2_15_3_14 + entries[13] * d2_11_3_10);
        m.entries[2] = entries[1] * d6_15_7_14 - entries[5] * d2_15_3_14 + entries[13] * d2_7_3_6;
        m.entries[3] = -(entries[1] * d6_11_7_10 - entries[5] * d2_11_3_10 + entries[9] * d2_7_3_6);
        m.entries[5] = entries[0] * d10_15_11_14 - entries[2] * d8_15_11_12 + entries[3] * d8_14_10_12;
        m.entries[6] = -(entries[0] * d6_15_7_14 - entries[4] * d2_15_3_14 + entries[12] * d2_7_3_6);
        m.entries[7] = entries[0] * d6_11_7_10 - entries[2] * d4_11_7_8 + entries[3] * d4_10_6_8;
        m.entries[9] = -(entries[0] * d9_15_11_13 - entries[1] * d8_15_11_12 + entries[3] * d8_13_9_12);
        m.entries[13] = entries[0] * d9_14_10_13 - entries[1] * d8_14_10_12 + entries[2] * d8_13_9_12;
        m.entries[10] = entries[0] * d5_15_7_13 - entries[1] * d4_15_7_12 + entries[3] * d4_13_5_12;
        m.entries[11] = -(entries[0] * d5_11_7_9 - entries[1] * d4_11_7_8 + entries[3] * d4_9_5_8);
        m.entries[14] = -(entries[0] * d5_14_6_13 - entries[1] * d4_14_6_12 + entries[2] * d4_13_5_12);
        m.entries[15] = entries[0] * d5_10_6_9 - entries[1] * d4_10_6_8 + entries[2] * d4_9_5_8;

        scalar det = entries[0] * m.entries[0]
            + entries[1] * m.entries[4]
            + entries[2] * m.entries[8]
            + entries[3] * m.entries[12];

        if (Math::equal(det, 0))
            return Matrix4x4();
        
        m = m / det;

        return m;
    }

    void Matrix4x4::transpose()
    {
        *this = getTranspose();
    }

    Matrix4x4 Matrix4x4::getTranspose() const
    {
        return Matrix4x4
                (entries[ 0], entries[ 4], entries[ 8], entries[12],
                entries[ 1], entries[ 5], entries[ 9], entries[13],
                entries[ 2], entries[ 6], entries[10], entries[14],
                entries[ 3], entries[ 7], entries[11], entries[15]);
    }

    void Matrix4x4::invertTranspose()
    {
        *this = getInverseTranspose();
    }

    Matrix4x4 Matrix4x4::getInverseTranspose() const {
        Matrix4x4 mat = getInverse();
        mat.transpose();
        return mat;
    }

    Matrix4x4 Matrix4x4::getInverseTransform() const
    {
#if 0
        Matrix4x4 m2;
        m2.entries[12] = -entries[12];
        m2.entries[13] = -entries[13];
        m2.entries[14] = -entries[14];

        Matrix3x3 m1 = toMatrix3x3();
        m1 = m1.getInverse();
        vec3 v = m1 * vec3(m2.entries[12], m2.entries[13], m2.entries[14]);

        m2.setMatrix3x3(m1);
        m2.entries[12] = v.x;
        m2.entries[13] = v.y;
        m2.entries[14] = v.z;

        return m2;
#endif

        Matrix4x4 m2;
        m2.entries[3] = -entries[3];
        m2.entries[7] = -entries[7];
        m2.entries[11] = -entries[11];

        Matrix3x3 m1 = toMatrix3x3();
        m1 = m1.getInverse();
        vec3 v = m1 * vec3(m2.entries[3], m2.entries[7], m2.entries[11]);

        m2.setMatrix3x3(m1);
        m2.entries[3] = v.x;
        m2.entries[7] = v.y;
        m2.entries[11] = v.z;

        return m2;
    }

    Matrix4x4 Matrix4x4::getTranslation(const vec3& translation)
    {
        Matrix4x4 m;
        m.entries[3] = translation.x;
        m.entries[7] = translation.y;
        m.entries[11] = translation.z;
        return m;
    }

    Matrix4x4 Matrix4x4::getScale(const vec3& scaleFactor)
    {
        Matrix4x4 m;
        m.entries[0] = scaleFactor.x;
        m.entries[5] = scaleFactor.y;
        m.entries[10] = scaleFactor.z;
        return m;
    }

    Matrix4x4 Matrix4x4::getRotationAxis(const double radians, const vec3& axis)
    {
        Matrix4x4 m;

        vec3 u = axis.getNormalized();

        float sinAngle = static_cast<float>(sin(radians));
        float cosAngle = static_cast<float>(cos(radians));
        float oneMinusCosAngle = 1.0f - cosAngle;

        m.entries[0] = cosAngle + oneMinusCosAngle * (1 - u.x * u.x);
        m.entries[1] = (u.x) * (u.y) * (oneMinusCosAngle)-sinAngle * u.z;
        m.entries[2] = (u.x) * (u.z) * (oneMinusCosAngle)+sinAngle * u.y;

        m.entries[4] = (u.x) * (u.y) * (oneMinusCosAngle)+sinAngle * u.z;
        m.entries[5] = cosAngle + oneMinusCosAngle * u.y * u.y;
        m.entries[6] = (u.y) * (u.z) * (oneMinusCosAngle)-sinAngle * u.x;

        m.entries[8] = (u.x) * (u.z) * (oneMinusCosAngle)-sinAngle * u.y;
        m.entries[9] = (u.y) * (u.z) * (oneMinusCosAngle)+sinAngle * u.x;
        m.entries[10] = cosAngle + oneMinusCosAngle * u.z * u.z;

        return m;
    }

    Matrix4x4 Matrix4x4::getRotationX(const double radians)
    {
        Matrix4x4 m;
        m.entries[5]  =  static_cast<float>(cos(radians));
        m.entries[6]  =  -static_cast<float>(sin(radians));
        m.entries[9]  = -m.entries[6];
        m.entries[10] =  m.entries[5];
        return m;
    }

    Matrix4x4 Matrix4x4::getRotationY(const double radians)
    {
        Matrix4x4 m;
        m.entries[0]  =  static_cast<float>(cos(radians));
        m.entries[2]  = static_cast<float>(sin(radians));
        m.entries[8]  = -m.entries[2];
        m.entries[10] =  m.entries[0];
        return m;
    }

    Matrix4x4 Matrix4x4::getRotationZ(const double radians)
    {
        Matrix4x4 m;
        m.entries[0]  =  static_cast<float>(cos(radians));
        m.entries[1]  =  -static_cast<float>(sin(radians));
        m.entries[4]  = -m.entries[1];
        m.entries[5]  = m.entries[0];
        return m;
    }

    Matrix4x4 Matrix4x4::getRotationEuler(const double radiansX, const double radiansY, const double radiansZ)
    {
        Matrix4x4 m;
        // pitch, raw, roll
        float cr   = cos(radiansX);
        float sr   = sin(radiansX);
        float cp   = cos(radiansY);
        float sp   = sin(radiansY);
        float cy   = cos(radiansZ);
        float sy   = sin(radiansZ);

        m.entries[0]  = static_cast<float>(cp*cy);
        m.entries[1]  = static_cast<float>(cp*sy);
        m.entries[2]  = static_cast<float>(-sp);

        double srsp = sr*sp;
        double crsp = cr*sp;

        m.entries[4]  = static_cast<float>(srsp*cy-cr*sy);
        m.entries[5]  = static_cast<float>(srsp*sy+cr*cy);
        m.entries[6]  = static_cast<float>(sr*cp);
        m.entries[8]  = static_cast<float>(crsp*cy+sr*sy);
        m.entries[9]  = static_cast<float>(crsp*sy-sr*cy);
        m.entries[10] = static_cast<float>(cr*cp);

        return m;
    }

    Matrix4x4 Matrix4x4::getPerspective(float fovy, float aspect, float near, float far)
    {
        Matrix4x4 mat;
        mat.loadZero();
        // convert fov from degrees to radians
        float rFovy = fovy * A_PI / 180;
        const float tanHalfFovy = tanf(static_cast<float>(rFovy * 0.5f));
        mat.entries[0] = 1.0f / (aspect * tanHalfFovy);
        mat.entries[5] = 1.0f / (tanHalfFovy);
        mat.entries[10] = -(far + near) / (far - near);
        //this two should be swapped,but if done ,the gltf model can not show
        //normal ,why? FIXME!
        mat.entries[11] = (-2.0f * near * far) / (far - near);
        mat.entries[14] = -1.0f;

        return mat;
    }


    Matrix4x4 Matrix4x4::getOrtho(float left, float right, float bottom, float top, float near, float far)
    {
        Matrix4x4 mat;
        mat.entries[0] = 2.0f / (right - left);
        mat.entries[5] = 2.0f / (top - bottom);
        mat.entries[10] = -2.0f / (far - near);
        mat.entries[3] = -(right + left) / (right - left);
        mat.entries[7] = -(top + bottom) / (top - bottom);
        mat.entries[11] = -(far + near) / (far - near);

        return mat;
    }

    Matrix4x4 Matrix4x4::getLookAt(vec3 cameraPos, vec3 target, vec3 worldUp)
    {
        Matrix4x4 mat;
        vec3 zAxis = cameraPos - target;
        zAxis.normalize();
        vec3 xAxis = worldUp.crossProduct(zAxis);
        xAxis.normalize();
        vec3 yAxis = zAxis.crossProduct(xAxis);
        yAxis.normalize();

        mat.entries[0] = xAxis.x;
        mat.entries[1] = xAxis.y;
        mat.entries[2] = xAxis.z;

        mat.entries[4] = yAxis.x;
        mat.entries[5] = yAxis.y;
        mat.entries[6] = yAxis.z;

        mat.entries[8] = zAxis.x;
        mat.entries[9] = zAxis.y;
        mat.entries[10] = zAxis.z;

        mat.entries[3] = -(xAxis.dotProduct(cameraPos));
        mat.entries[7] = -(yAxis.dotProduct(cameraPos));
        mat.entries[11] = -(zAxis.dotProduct(cameraPos));

        return mat;
    }

    Matrix4x4 Matrix4x4::getViewPort(int left, int top, int width, int height)
    {
        Matrix4x4 mat;
        mat.entries[0] = static_cast<scalar>(width) / 2.0f;
        mat.entries[5] = -static_cast<scalar>(height) / 2.0f;
        mat.entries[3] = static_cast<scalar>(left) + static_cast<scalar>(width) / 2.0f;
        mat.entries[7] = static_cast<scalar>(top) + static_cast<scalar>(height) / 2.0f;

        return mat;
    }

    void Matrix4x4::decomposeTransRotScale(vec3& position, Quaternion& rotation, vec3& scale) const
    {
        position.x = entries[3];
        position.y = entries[7];
        position.z = entries[11];

        scale.x = 1.0;
        scale.y = 1.0;
        scale.z = 1.0;
        float x = entries[0] * entries[0] + entries[4] * entries[4] + entries[8] * entries[8];
        float y = entries[1] * entries[1] + entries[5] * entries[5] + entries[9] * entries[9];
        float z = entries[2] * entries[2] + entries[6] * entries[6] + entries[10] * entries[10];
        if (x > SMALL_NUMBER)
            scale.x = sqrt(x);
        else
            scale.x = 0;
        if (y > SMALL_NUMBER)
            scale.y = sqrt(y);
        else
            scale.y = 0;
        if (z > SMALL_NUMBER)
            scale.z = sqrt(z);
        else
            scale.z = 0;

        if (Math::equal(scale.x, 0) || Math::equal(scale.y, 0) || Math::equal(scale.z, 0)) {
            rotation = Quaternion();
            return;
        }

        float mat[9] = { entries[0] / scale.x, entries[1] / scale.y, entries[2] / scale.z,
            entries[4] / scale.x, entries[5] / scale.y, entries[6] / scale.z,
            entries[8] / scale.x, entries[9] / scale.y, entries[10] / scale.z };

        //计算四元数 from UE4
        if ((fabs(mat[0] <= KINDA_SMALL_NUMBER
            && fabs(mat[1]) <= KINDA_SMALL_NUMBER
            && fabs(mat[2]) <= KINDA_SMALL_NUMBER) || (
                fabs(mat[3]) <= KINDA_SMALL_NUMBER &&
                fabs(mat[4]) <= KINDA_SMALL_NUMBER &&
                fabs(mat[5]) <= KINDA_SMALL_NUMBER)) || (
                    fabs(mat[6] <= KINDA_SMALL_NUMBER) &&
                    fabs(mat[7] <= KINDA_SMALL_NUMBER) &&
                    fabs(mat[8] <= KINDA_SMALL_NUMBER)))
        {
            rotation = Quaternion(0, 0, 0, 1);
            return;
        }
        float s = 0.f;
        float tr = mat[0] + mat[4] + mat[8];
        if (tr > 0.0) {
            float invS = Math::invSqrt(tr + 1.0f);
            rotation.w = 0.5f * (tr + 1.0f) * invS;
            float s = 0.5f * invS;
#if 0 
            rotation.x = (mat[5] - mat[7]) * s;
            rotation.y = (mat[6] - mat[2]) * s;
            rotation.z = (mat[1] - mat[3]) * s;
#endif 
            rotation.x = (mat[7] - mat[5]) * s;
            rotation.y = (mat[2] - mat[6]) * s;
            rotation.z = (mat[3] - mat[1]) * s;
        }
        else {
            int i = 0;
            if (mat[4] > mat[0])
                i = 1;
            if (mat[8] > entries[i * 3 + i])
                i = 2;
            static const int nxt[3] = { 1,2,0 };
            const int j = nxt[i];
            const int k = nxt[j];

            s = mat[i * 3 + i] - mat[j * 3 + j] - mat[k * 3 + k] + 1.0f;
            float invS = Math::invSqrt(s);
            float qt[4];
            qt[i] = 0.5 * (1.0 / invS);
            s = 0.5 * invS;

            qt[3] = (mat[j * 3 + k] - mat[k * 3 + j]) * s;
            qt[j] = (mat[i * 3 + j] + mat[j * 3 + i]) * s;
            qt[k] = (mat[i * 3 + k] + mat[k * 3 + i]) * s;
#if 0
            rotation.x = qt[0];
            rotation.y = qt[1];
            rotation.z = qt[2];
#endif
            rotation.x = -qt[0];
            rotation.y = -qt[1];
            rotation.z = -qt[2];
        }
    }

}
