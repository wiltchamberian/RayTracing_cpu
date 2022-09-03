#pragma once

#include "HitableList.h"
#include "SphereObj.h"
#include "texture.h"
#include "Material.h"
using namespace Sun;

Hitable* final() {
    int nb = 20;
    std::vector<Hitable*> list;
    std::vector<Hitable*> boxList;
    std::vector<Hitable*> boxList2;
    Material* white = new Lambertian(new constant_texture({ 0.73,0.73,0.73 }));
    Material* ground = new Lambertian(new constant_texture({ 0.48,0.83,0.53 }));
    int b = 0;
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < nb; ++j) {
            float w = 100;
            float x0 = -1000 + i * w;
            float z0 = -1000 + j * w;
            float y0 = 0;
            float x1 = x0 + w;
            float y1 = 100 * (rand48() + 0.01);
            float z1 = z0 + w;
            //boxList[b++] = new Box({ x0,y0,z0 }, { x1,y1,z1 }, ground);
        }
    }
    return nullptr;
}

