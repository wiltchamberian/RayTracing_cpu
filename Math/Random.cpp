#include "Random.h"

namespace Sun {

    vec3 randomPointInUnitSphere() {
        vec3 p;
        do {
            p = vec3(rand48(), rand48(), rand48()) * 2 - vec3(1, 1, 1);
        } while (p.dotProduct(p) >= 1.0);
        return p;
    }

    std::vector<vec3> randomIntVector3Ds(int x_min, int x_max, int y_min, int y_max, int z_min, int z_max, int n) {
        std::vector<vec3> res(n);
        for (int i = 0; i < n; ++i) {
            res[i] = randomIntVector3D(x_min, x_max, y_min, y_max, z_min, z_max);
        }
        return std::move(res);
    }

    std::vector<std::vector<vec3>> buildScatterPoints(const vec3& start,
        const vec3& box, float r, int n)
    {
        std::vector<std::vector<vec3>> result;
        randomSeed();
        for (int i = 0; i < n; ++i) {
            vec3 center = randomVector3D(start.x, start.x + box.x, start.y, start.y + box.y,
                start.z, start.z + box.z);
            std::vector<vec3> vec = randomVector3Ds(center.x - r / 2, center.x + r / 2, center.y - r / 2, center.y + r / 2, center.z - r / 2, center.z + r / 2, n);
            //把超出范围的平移到范围内部
            for (int j = 0; j < vec.size(); ++j) {
                if (vec[j].x < start.x)vec[j].x = start.x;
                if (vec[j].y < start.y)vec[j].y = start.y;
                if (vec[j].z < start.z)vec[j].z = start.z;
                if (vec[j].x > start.x + box.x) vec[j].x = start.x + box.x;
                if (vec[j].y > start.y + box.y)vec[j].y = start.y + box.y;
                if (vec[j].z > start.z + box.z)vec[j].z = start.z + box.z;
            }
            result.push_back(vec);
        }
        return result;
    }


}