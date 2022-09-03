#pragma once

#include <cmath>
#include "Math/Vector3D.h"
#include "Math/Random.h"

namespace Sun {

    class texture
    {
    public:
        virtual vec3 value(float u, float v, const vec3& p) const = 0;
    };

    class constant_texture :public texture {
    public:
        constant_texture() {}
        constant_texture(const vec3& c) :color_(c) {}
        virtual vec3 value(float u, float v, const vec3& p)const {
            return color_;
        }
        vec3 color_;
    };

    class checker_texture : public texture {
    public:
        checker_texture() {}
        checker_texture(texture* t0, texture* t1) :even_(t0), odd_(t1) {
        }
        virtual vec3 value(float u, float v, const vec3& p) const {
            float sines = std::sin(10 * p.x) * sin(10 * p.y) * sin(10 * p.z);
            if (sines < 0) {
                return odd_->value(u, v, p);
            }
            else {
                return even_->value(u, v, p);
            }
        }
    public:
        texture* odd_;
        texture* even_;
    };

    class image_texture : public texture {
    public:
        image_texture() {}
        image_texture(unsigned char* pixels, int w, int h) :data_(pixels), w_(w), h_(h)
        {
        }
        virtual vec3 value(float u, float v, const vec3& p) const;
        unsigned char* data_;
        int w_, h_;
    };

    inline float trilinear_interp(float c[2][2][2], float u, float v, float w) {
        float accum = 0;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    accum += (i * u + (1 - i) * (1 - u)) *
                        (j * v + (1 - j) * (1 - v)) *
                        (k * w + (1 - k) * (1 - w)) * c[i][j][k];
                }
            }
        }
        return accum;
    }

    inline float perlin_interp(vec3 c[2][2][2], float u, float v, float w) {
        float uu = u * u * (3 - 2 * u);
        float vv = v * v * (3 - 2 * v);
        float ww = w * w * (3 - 2 * w);
        float accum = 0;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    vec3 weight_v(u - i, v - j, w - k);
                    accum += (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) *
                        (k * ww + (1 - k) * (1 - ww)) * c[i][j][k].dotProduct(weight_v);
                }
            }
        }
        return accum;
    }

    //°ØÁÖÔëÉù
    class perlin {
    public:
        float noise(const vec3& p) const {
            float u = p.x - floor(p.x);
            float v = p.y - floor(p.y);
            float w = p.z - floor(p.z);
            /*u = u * u * (3 - 2 * u);
            v = v * v * (3 - 2 * v);
            w = w * w * (3 - 2 * w);*/
            /*int i = int(4 * p.x) & 255;
            int j = int(4 * p.y) & 255;
            int k = int(4 * p.z) & 255;*/
            int i = floor(p.x);
            int j = floor(p.y);
            int k = floor(p.z);
            vec3 c[2][2][2];
            for (int di = 0; di < 2; di++) {
                for (int dj = 0; dj < 2; dj++) {
                    for (int dk = 0; dk < 2; dk++) {
                        c[di][dj][dk] = ranvec[perm_x[(i + di) & 255] ^ perm_y[(j + dj) ^ 255] ^ perm_z[(k + dk) & 255]];
                    }
                }
            }
            return perlin_interp(c, u, v, w);
            //return trilinear_interp(c, u, v, w);
            //return ranfloat[perm_x[i] ^ perm_y[i] ^ perm_z[k]];


        }
        float turb(const vec3& p, int depth = 7)const {
            float accum = 0;
            vec3 tmp_p = p;
            float weight = 1.0;
            for (int i = 0; i < depth; ++i) {
                accum += weight * noise(tmp_p);
                weight *= 0.5;
                tmp_p *= 2;
            }
            return accum > 0 ? accum : (-accum);
        }
        static float* ranfloat;
        static vec3* ranvec;
        static int* perm_x;
        static int* perm_y;
        static int* perm_z;
    };

    static float* perlin_generate() {
        float* p = new float[256];
        for (int i = 0; i < 256; ++i) {
            p[i] = rand48();
        }
        return p;
    }

    static vec3* perlin_generateVec() {
        vec3* p = new vec3[256];
        for (int i = 0; i < 256; ++i) {
            p[i] = vec3(-1 + 2 * rand48(), -1 + 2 * rand48(), -1 + 2 * rand48()).getNormalized();
        }
        return p;
    }

    static void permute(int* p, int n) {
        for (int i = n - 1; i > 0; --i) {
            int target = int(rand48() * (i + 1));
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
        return;
    }

    static int* perlin_generate_perm() {
        int* p = new int[256];
        for (int i = 0; i < 256; ++i) {
            p[i] = i;
        }
        permute(p, 256);
        return p;
    }


    class noise_texture :public texture {
    public:
        noise_texture() {}
        noise_texture(float scale) :scale_(scale) {}
        virtual vec3 value(float u, float v, const vec3& p)const {
            //return Vector3D(1, 1, 1) * noise_.noise(p);
            return vec3(1, 1, 1) * 0.5 * (1 + sin(scale_ * p.z + 10 * noise_.turb(p)));

        }
        perlin noise_;
        float scale_ = 1.;
    };

}
