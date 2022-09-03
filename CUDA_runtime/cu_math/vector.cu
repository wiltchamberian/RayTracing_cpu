#include "cu_vector.h"

__host__ __device__ cvec3 vec3_gen(float x, float y, float z) {
    cvec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

__host__ __device__ cvec3 vec3_add(const cvec3& v1, const cvec3& v2) {
    cvec3 out;
    out.x = v1.x + v2.x;
    out.y = v1.y + v2.y;
    out.z = v1.z + v2.z;
    return out;
}

__host__ __device__ cvec3 vec3_sub(const cvec3& v1, const cvec3& v2) {
    cvec3 out;
    out.x = v1.x - v2.x;
    out.y = v1.y - v2.y;
    out.z = v1.z - v2.z;
    return out;
}

__host__ __device__ float vec3_dot(const cvec3& v1, const cvec3& v2) {
    float out;
    out = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    return out;
}

__host__ __device__ cvec3 vec3_mul(const cvec3& v1, const cvec3& v2) {
    cvec3 res;
    res.x = v1.x * v2.x;
    res.y = v1.y * v2.y;
    res.z = v1.z * v2.z;
    return res;
}

__host__ __device__ cvec3 vec3_nmul(const cvec3& v1, float k) {
    cvec3 out;
    out.x = v1.x * k;
    out.y = v1.y * k;
    out.z = v1.z * k;
    return out;
}

__host__ __device__ cvec3 vec3_ndiv(const cvec3& v1, float k) {
    cvec3 out;
    out.x = v1.x / k;
    out.y = v1.y / k;
    out.z = v1.z / k;
    return out;
}

__host__ __device__ cvec3 vec3_reflect(const cvec3& v, const cvec3& normal) {
    return vec3_sub(vec3_nmul(normal,-vec3_dot(v, normal)*2),v);
}

__host__ __device__ cvec3 vec3_normalize(cvec3& vec) {
    cvec3 res;
    float l = rsqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    res.x = vec.x * l;
    res.y = vec.y * l;
    res.z = vec.z * l;
    return res;
}

__host__ __device__ cvec3 vec3_cross(const cvec3& v1, const cvec3& v2) {
    cvec3 res;
    res.x = v1.y * v2.z - v1.z * v2.y;
    res.y = v1.z * v2.x - v1.x * v2.z;
    res.z = v1.x * v2.y - v1.y * v2.z;
    return res;
}

