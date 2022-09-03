#ifndef __CUDA_TEXTURE_H
#define __CUDA_TEXTURE_H

#include <cstdint>

//img:width*height*4
struct CTexture {
    uint32_t width;
    uint32_t height;
    unsigned char* data;
};




#endif