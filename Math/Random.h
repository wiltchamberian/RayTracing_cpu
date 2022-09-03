/*****************************************************************************
* @brief : Random
* 一些基础的随机数生成，跨平台
* @author : acedtang
* @date : 2021/5/9
* @version : ver 1.0
* @inparam :
* @outparam :
*****************************************************************************/

#ifndef __RANDOM_H
#define __RANDOM_H

#include <random>
#include <ctime>
#include "Vector2D.h"
#include "Vector3D.h"
#include "Vector4D.h"

namespace Sun {

    //
    struct GpuRandom {
    public:
        inline uint whash(uint seed)
        {
            seed = (seed ^ uint(61)) ^ (seed >> uint(16));
            seed *= uint(9);
            seed = seed ^ (seed >> uint(4));
            seed *= uint(0x27d4eb2d);
            seed = seed ^ (seed >> uint(15));
            return seed;
        }
        //

        float randcore4()
        {
            wseed = whash(wseed);

            return float(wseed) * (1.0 / 4294967296.0);
        }
    public:

        uint wseed = 0;
    };
    


    //[0,1]之间的随机数
    inline float rand48() {
        return (float)rand() / RAND_MAX;
    }

    //生成单位球内的随机向量
    vec3 randomPointInUnitSphere();

    inline void randomSeed() {
        srand((int)time(0));
    }

    inline float randomNumber(float lowerBound, float upperBound) {
        //srand((int)time(0));  // 产生随机种子  把0换成NULL也行
        int n = rand();
        float res = (float)n / RAND_MAX * (upperBound - lowerBound) + lowerBound;

        return res;
    }

    inline int randomIntNumber(int lowerBound, int upperBound) {
        int n = rand();
        float res = (float)n / RAND_MAX * (upperBound - lowerBound) + lowerBound;

        return (int)res;
    }

    //随机生成n个数字
    inline std::vector<float> randomNumbers(float lowerBound, float upperBound, int n) {
        std::vector<float> res(n, 0);
        for (int i = 0; i < res.size(); ++i) {
            res[i] = randomNumber(lowerBound, upperBound);
        }
        return std::move(res);
    }

    //随机生成1个2d点,均匀分布
    inline vec2 randomVector2D(float x_min, float x_max, float y_min, float y_max) {
        vec2 res;
        res.x = randomNumber(x_min, x_max);
        res.y = randomNumber(y_min, y_max);

        return res;
    }

    //随机生成n个2d点
    inline std::vector<vec2> randomVector2Ds(float x_min, float x_max, float y_min, float y_max, int n) {
        std::vector<vec2> res(n);
        for (int i = 0; i < n; ++i) {
            res[i] = randomVector2D(x_min, x_max, y_min, y_max);
        }
        return std::move(res);
    }

    //随机生成1个3d点
    inline vec3 randomVector3D(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
        vec3 res;
        res.x = randomNumber(x_min, x_max);
        res.y = randomNumber(y_min, y_max);
        res.z = randomNumber(z_min, z_max);
        return res;
    }

    inline vec3 randomIntVector3D(int x_min, int x_max, int y_min, int y_max, int z_min, int z_max) {
        vec3 res;
        res.x = randomIntNumber(x_min, x_max);
        res.y = randomIntNumber(y_min, y_max);
        res.z = randomIntNumber(z_min, z_max);
        return res;
    }

    //随机生成n个3d点
    inline std::vector<vec3> randomVector3Ds(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, int n) {
        std::vector<vec3> res(n);
        for (int i = 0; i < n; ++i) {
            res[i] = randomVector3D(x_min, x_max, y_min, y_max, z_min, z_max);
        }
        return std::move(res);
    }

    //整数值输出
    std::vector<vec3> randomIntVector3Ds(int x_min, int x_max, int y_min, int y_max, int z_min, int z_max, int n);


    //随机生成rgb颜色
    inline vec4 randomColorRGB() {
        vec4 color;
        color.x = randomNumber(0, 1);
        color.y = randomNumber(0, 1);
        color.z = randomNumber(0, 1);
        color.w = 1.0;
        return color;
    }

  
    

    //在起始点为(x0,y0,z0),X,Y,Z方向长度分别(x,y,z)的正六面体区域内
    //生成n个散点集，其中每个散点集在宽度为2*r的正方体内，该正方体的中心是一个位于该正六面体内均匀分布的随机数
    std::vector<std::vector<vec3>> buildScatterPoints(const vec3& start,
        const vec3& box, float r, int n);

}

#endif