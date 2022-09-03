#include "texture.h"

namespace Sun {

    float* perlin::ranfloat = perlin_generate();
    vec3* perlin::ranvec = perlin_generateVec();
    int* perlin::perm_x = perlin_generate_perm();
    int* perlin::perm_y = perlin_generate_perm();
    int* perlin::perm_z = perlin_generate_perm();

    vec3 image_texture::value(float u, float v, const vec3& p) const {
        int i = (u)*w_;
        int j = (1 - v) * h_; //-0.001?
        if (i < 0)i = 0;
        if (j < 0) j = 0;
        if (i > w_ - 1)i = w_ - 1;
        if (j > h_ - 1) j = h_ - 1;
        vec3 res;
        res.x = int(data_[3 * i + 3 * w_ * j]) / 255.;
        res.y = int(data_[3 * i + 3 * w_ * j + 1]) / 255.;
        res.z = int(data_[3 * i + 3 * w_ * j + 2]) / 255.;
        return res;
    }

}
