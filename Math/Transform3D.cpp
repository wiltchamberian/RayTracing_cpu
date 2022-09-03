#include "Transform3D.h"
#include "MathUtils.h"
#include "Quaternion.h"

namespace Sun
{

    Transform3D::Transform3D()
        :scale_(1.0,1.0,1.0)
    {
    }

    Transform3D::Transform3D(const vec3& translation, const Quaternion& quater,
        const vec3& scale) {
        translation_ = translation;
        rotation_ = quater;
        scale_ = scale;
    }

    Transform3D::~Transform3D()
    {

    }

    void Transform3D::translate(const vec3& world_space_v) {
        translation_ += world_space_v;
    }

    Transform3D Transform3D::getTranslate(const vec3 v) {
        Transform3D res = *this;
        res.translate(v);
        return res;
    }

    void Transform3D::scale(const vec3& scale) {
        scale_.x *= scale.x;
        scale_.y *= scale.y;
        scale_.z *= scale.z;
    }

    void Transform3D::scale(float x, float y, float z)
    {
        scale_.x *= x;
        scale_.y *= y;
        scale_.z *= z;
    }

    void Transform3D::rotate(const vec3& axis, double radians)
    {
        Quaternion q;
        q.setRotateAxisAndRadius(axis, radians);
        rotation_ = q * rotation_;
    }

    Matrix4x4 Transform3D::toMatrix() const
    {
        Matrix4x4 res;
        Matrix4x4 rotation = rotation_.toMatrix();
        Matrix4x4 trans, scals;
        scals = Matrix4x4::getScale(scale_);
        trans = Matrix4x4::getTranslation(translation_);
        //res = trans * scals * rotation; //FIXME make sure
        res = trans * rotation * scals;
        return res;

    }

    vec3 Transform3D::z_direction() const
    {
        return rotation_.rotate(vec3(0, 0, 1));
    }

    vec3 Transform3D::x_direction() const
    {
        return rotation_.rotate(vec3(1, 0, 0));
    }

    vec3 Transform3D::y_direction() const
    {
        return rotation_.rotate(vec3(0, 1, 0));
    }

    void Transform3D::setScale(const vec3& s) {
        scale_ = s;
    }

    void Transform3D::setRotation(const Quaternion& r) {
        rotation_ = r;
    }

    void Transform3D::setRotation(float a, float b, float c, float d) {
        rotation_ = Quaternion(a, b, c, d);
    }

    void Transform3D::setTranslation(const vec3& t) {
        translation_ = t;
    }

    void Transform3D::setMatrix(const Matrix4x4& matrix)
    {
        //model = matrix;
        matrix.decomposeTransRotScale(translation_, rotation_, scale_);
    }
     
    //将p从模型空间映射到世界空间(T*R*S)*p
    vec3 Transform3D::operator* (const vec3& p) const {
        vec3 res = this->scale_ * p;
        res = this->rotation_.rotate( res);
        res = this->translation_ + res;
        return res;
    }

    //this should be checked! TODO
    // T represent the translation, S represent the scale, q represent the quaterion
    // p'= T+q*(S*p)*q^(-1) => 
    // p =  S^(-1)*q^(-1)*(p'-T)*q =>
    // p = S^(-1)*q^(-1)*(-T)*q + S^(-1)*q^(-1)*p'*q  =>
    // p = S^(-1)*q^(-1)*(-T)*q + q^(-1)*[q*S^(-1)*q^(-1)]*p'*q
    // set T' = S^(-1)*q^(-1)*(-T)*q , q'=q^(-1) ,S'= q*S^(-1)*q^(-1), we have
    // p = T' + q'*(S'*p')* q'^(-1) ,this is what we want!
    Transform3D Transform3D::getInverse()
    {
        Transform3D res;
        vec3 invScale = vec3(1.0 / scale_.x, 1.0 / scale_.y, 1.0 / scale_.z);
        Quaternion invRot = rotation_.getInverse();
        res.translation_ = invScale * invRot.rotate(-translation_);
        res.rotation_ = invRot;
        res.scale_ = rotation_.rotate(invScale);
        return res;
    }

    Transform3D Transform3D::getLookAt(vec3 cameraPos, vec3 target, vec3 worldUp) {
        Transform3D tf;
        tf.scale_ = Vector3D_4(1.0, 1.0, 1.0);

        vec3 axisZ = (cameraPos - target).getNormalized();
        vec3 axisX = worldUp.crossProduct(axisZ).getNormalized();
        vec3 targetAxisZ(0, 0, 1);
        vec3 targetAxisX(1, 0, 0);

        Quaternion qua1;
        qua1 = Quaternion::createByNormalizedDirection(axisZ, targetAxisZ);
        vec3 axisX2 = qua1.rotate(axisX);

        Quaternion qua2;
        qua2 = Quaternion::createByNormalizedDirection(axisX2, targetAxisX);

        tf.rotation_ = qua2 * qua1;

        tf.translation_ = - tf.rotation_ .rotate(cameraPos);

        return tf;
#if 0
        //将z轴旋转到和目标z轴相同
        Quaternion qua1;
        Vector3D axisZ = cameraPos - target;
        axisZ.normalize();
        Vector3D rotateAxis = Vector3D(0, 0, 1).crossProduct(axisZ);
        float radius = acos(Vector3D(0, 0, 1).dotProduct(axisZ));
        qua1.setRotateAxisAndRadius(rotateAxis,radius);

        //计算此时的x轴指向
        Vector3D axisX = qua1.rotate(Vector3D(1, 0, 0));

        //目标X轴指向
        Vector3D targetAxisX = worldUp.crossProduct(axisZ);
        Vector3D rotate2 = axisX.crossProduct(targetAxisX);
        float radius2 = acos(axisX.dotProduct(targetAxisX));
        Quaternion qua2;
        qua2.setRotateAxisAndRadius(rotate2, radius2);

        tf.rotation_ = qua2 * qua1;

        return tf;
#endif 

    }

}