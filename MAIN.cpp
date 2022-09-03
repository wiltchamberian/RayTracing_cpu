// RayTracing.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include <random>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Math/Algorithm.h"
#include "RayTracing.h"
#include "HitableList.h"
#include "SphereObj.h"
#include "Camera.h"
#include "Material.h"
#include "texture.h"
#include "BVH.h"
#include "GpuRayTracing.h"
#include "Math/Matrix4x4.h"
#include "SurfaceBuilder.h"
#include "3x_add_1.h"
using namespace Sun;

//构建场景中不同材质的物体，这里主要是球体
HitableList* renderScene() {
    int n = 500;
    std::vector<Hitable*> list(n + 1, nullptr);
    texture* checker = new checker_texture(new constant_texture({ 0.2,0.3,0.1 }),new constant_texture(
    { 0.9, 0.9, 0.9 }));
    list[0] = new SphereObj({ 0,-1000,0 }, 1000, new Lambertian(checker/*new constant_texture({ 0.5,0.5,0.5 })*/));
    int i = 1;
    //[-11,10]区域内构建球体
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; ++b) {
            float choose_mat = rand48();
            //随机中心位置,但是高度始终为0.2
            vec3 center(a + 0.9 * rand48(), 0.2, b + 0.9 * rand48());
            if ((center - vec3(4, 0.2, 0)).getLength() > 0.9) {
                if (choose_mat < 0.8) { //diffuse                
                    list[i++] = new SphereObj(center, 0.2, new Lambertian(new constant_texture(vec3(rand48() * rand48(), rand48() * rand48(), rand48() * rand48()))));

                }
                else if (choose_mat < 0.95) { //metal
                    list[i++] = new SphereObj(center,0.2,new Metal(vec3(0.5 * (1 + rand48()), 0.5 * (1 + rand48()), 0.5 * (1 + rand48())), 0.5 * rand48()));
                }
                else { //glass
                    list[i++] = new SphereObj(center, 0.2, new Dielectrics(1.5));
                }
            }
        }
    }
    list[i++] = new SphereObj({ 0,1,0 }, 1.0, new Dielectrics(1.5));
    list[i++] = new SphereObj({ -4,1,0 }, 1.0, new Lambertian(new constant_texture({ 0.4,0.2,0.1 })));
    list[i++] = new SphereObj({ 4,1,0 }, 1.0, new Metal({ 0.7,0.6,0.5 }, 0.0));
    list.resize(i);
    return new HitableList(list);
}

Hitable* simple_light() {
    texture* pertext = new noise_texture(4);
    Hitable* tmp;
    std::vector<Hitable*> vec;
    vec.push_back(new SphereObj(vec3(0, -1000, 0), 1000, new Lambertian(pertext)));
    vec.push_back(new SphereObj({ 0,2,0 }, 2, new Lambertian(pertext)));
    vec.push_back(new SphereObj({ 0,7,0 }, 2, new diffuse_light(new constant_texture({ 4,4,4 }))));
    vec.push_back(new xy_rect(3, 5, 1, 3, -2, new diffuse_light(new constant_texture({ 4,4,4 }))));
    HitableList* list = new HitableList(vec);
    return list;
}

Hitable* twoShpereScene() {
    texture* checker = new checker_texture(new constant_texture({ 0.2,0.3,0.1 }),
        new constant_texture({ 0.9,0.9,0.9 }));
    int n = 50;
    std::vector<Hitable*> vec;
    vec.push_back(new SphereObj({ 0,-10,0 }, 10, new Metal({ 0.7,0.6,0.5 },0.0)/*new Lambertian(checker)*/));
    vec.push_back(new SphereObj({ 0,10,0 }, 10, new Dielectrics(1.5)/*new Lambertian(checker)*/));

    return new HitableList(vec);
}

Hitable* twoPerlinSpheres() {
    texture* pertext = new noise_texture();
    std::vector<Hitable*> vec;
    vec.push_back(new SphereObj({ 0,-1000,0 }, 1000, new Lambertian(pertext)));
    vec.push_back(new SphereObj({ 0,2,0 }, 2, new Lambertian(pertext)));

    return new HitableList(vec);
}

Hitable* cornell_box() {
    std::vector<Hitable*> vec(7);
    int i = 0;
    Material* red = new Lambertian(new constant_texture({ 0.65,0.05,0.05 }));
    Material* white = new Lambertian(new constant_texture({ 0.73,0.73,0.73 }));
    Material* green = new Lambertian(new constant_texture({ 0.12,0.45, 0.15 }));
    Material* light = new diffuse_light(new constant_texture({ 15,15,15 }));
    vec[i++] = new yz_rect(0, 555, 0, 555, 555, green);
    vec[i++] = new yz_rect(0, 555, 0, 555, 0, red);
    vec[i++] = new xz_rect(213, 343, 227, 332, 554, light);
    vec[i++] = new xz_rect(0, 555, 0, 555, 0, white);
    vec[i++] = new xy_rect(0, 555, 0, 555, 555, white);
    vec[i++] = new box({ 130,0,65 }, { 295,165,230 }, white);
    vec[i++] = new box({ 265,0,295 }, { 430,330,460 }, white);
    return new HitableList(vec);
}

Camera buildCamera(int nx, int ny) {
    //相机位置
    vec3 lookfrom(3, 3, 2);
    //相机朝向目标
    vec3 lookat(0, 0, -1);
    float dist_to_focus = (lookfrom - lookat).getLength();
    float aperture = 2.0;
    float time0 = 0;
    float time1 = 0;
    //Camera camera({ -2,2,1 }, { 0,0,-1 }, { 0,1,0 }, 90, float(nx) / float(ny), 0, 1, time0, time1);
    Camera camera({ 0,0,2 }, lookat, { 0,1,0 }, 50, float(nx) / float(ny), 0, 1, time0, time1);

    return camera;
}

Camera buildCameraTwoSphere(int nx, int ny) {
    vec3 lookfrom(13, 2, 3);
    vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.0;

    Camera cam(lookfrom, lookat, { 0,1,0 }, 20, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);
    return cam;
}

Camera buildCameraCornellBox(int nx ,int ny) {
    vec3 lookfrom(278, 278, -800);
    vec3 lookat(278, 278, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.0;
    float vfov = 40.;

    Camera cam(lookfrom, lookat, { 0,1,0 }, vfov, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);
    return cam;
}

void generateNoiseData() {
    srand((unsigned)time(NULL));
    std::fstream fs("Noise.ppm", std::ios::out);
    fs << "P3\n" << 100 << " " << 100 << "\n255\n";
    for (int i = 0; i < 10000; ++i) {
        vec3 v = randomPointInUnitSphere();
        int ir = (int)((v.x+1)*0.5 * 256);
        int ig = (int)((v.y + 1) * 0.5 * 256);
        int ib = (int)((v.z + 1) * 0.5 * 256);
        fs << ir << " " << ig << " " << ib << "\n";
    }
    fs.close();
}

const std::string g_dir = "C:\\Users\\Administrator\\Pictures\\";

//构建莫比乌斯带(main radius, sqrt(b),sqrt(a))
Polynomial3D buildMobius(float a, float b) {
    Polynomial3D x = Polynomial3D({ {3,0,0,1} });
    Polynomial3D y = Polynomial3D({ {0,1,0,1} });
    Polynomial3D z = Polynomial3D({ {0,0,1,1} });
    Polynomial3D one({ { 0,0,0,1 } });
    Polynomial3D left = (x*x + y*y + one)*(a*x*x+b*y*y)+z*z*(b*x*x+a*y*y)-2*(a+b)*x*y*z-a*b*(x*x+y*y);
    Polynomial3D mid = a * x * x + b * y * y - x * y * z * (a + b);
    Polynomial3D right = 4 * (x * x + y * y) * mid * mid;
    Polynomial3D last = left * left - right;
    return last;
}

void gpuRayTrace() {

    //test
   




    GpuRayTracing tracing;
    int nx = 1024;
    int ny = 768;
    Camera camera = buildCamera(nx,ny);
    GpuRayTracing::Camera& cam = tracing.camera;
    cam.horizontal = camera.horizontal_;
    cam.lower_left_corner = camera.lower_left_corner_;
    cam.origin = camera.origin_;
    cam.vertical = camera.vertical_;

    tracing.camera.origin = vec3(0, 0, 2);
    tracing.camera.horizontal = vec3(2, 0, 0);
    tracing.camera.vertical = vec3(0, 2, 0);
    tracing.camera.lower_left_corner = vec3(-1, -1, 1);

    GpuRayTracing::Ray ray;
    ray.origin = cam.origin;
    float u = 0.4;
    float v = 0.5;
    ray.direction = cam.lower_left_corner + cam.horizontal * u + cam.vertical * v - cam.origin;

    //添加球体
    /*
        struct Sphere {
            vec3 center;
            vec3 color;
            float radius;
            vec2 padding; t
    */
    //生成一些球体
    std::vector<GpuRayTracing::Sphere> spheres(3);
    spheres[0].center = { 0,0,-1 }; spheres[0].radius = 0.5;
    spheres[1].center = { 0,-100.5,-1 }; spheres[1].radius = 100;
    spheres[2].center = { 1,0,-1 }; spheres[2].radius = 0.5;
    spheres[0].color = { 0.1,0.2,0.5 };
    spheres[1].color = { 0.8,0.8,0 };
    spheres[2].color = { 0.8, 0.6, 0.2 };
    //tracing.spheres = spheres;

    //生成曲面
#if 0
    //(x2 + y2 + z2 + 1 − a)2 = 4(x2 + y2).
    float a = 0.09;
    Polynomial3D p1 = { {2,0,0,1} };
    Polynomial3D p2 = { {0,2,0,1} };
    Polynomial3D p3 = { {0,0,2,1} };
    Polynomial3D con = { {0,0,0,1 - a} };
    Polynomial3D poly = p1+ p2+ p3 + con;
    Polynomial3D poly2 = (p1 + p2);
    poly2 *= 4;
    Polynomial3D last = poly * poly;
    last = last - poly2;
    tracing.polynomials = { last };
    tracing.partialXs = { last.getPartialX() };
    tracing.partialYs = { last.getPartialY() };
    tracing.partialZs = { last.getPartialZ() };
#endif
    
#if 0
    Polynomial3D mobius = buildMobius(0.09, 0.04);
    tracing.polynomials.push_back(mobius);
    tracing.partialXs.push_back(mobius.getPartialX());
    tracing.partialYs.push_back(mobius.getPartialY());
    tracing.partialZs.push_back(mobius.getPartialZ());
#endif

#if 1
    //implicit surface
    //Expression expression("sqrt(x^2+y^2+z^2)-1");
    //Expression expression("(x^2 + y^2 + z^2 + 1 − 0.09)^2 - 4*(x^2 + y^2)");
    //Expression expression = buildTorus(1,0.3); //torus
    //Expression expression = buildBox({ 0.5,0.4,0.3 });
    //Expression expression = buildSdLink(1, 0.3,0.2);
    AExpression expression = buildMobius(1, 0.3, 0.1, 3);
    Transform3D trans;
    trans.translate({ 1,0,0 });
    trans.rotate({ 1,0,0 },A_PI/2);
    Surface surf;
    std::vector<Token> tokens = expression.getSymValuedDAG();
    assert(tokens.size() <= 50);
    for (int i = 0; i < tokens.size(); ++i) {
        surf.tokens[i] = tokens[i];
    }
    surf.num = tokens.size();
    tracing.sdfs.push_back(surf);
    Matrix4x4 mat = trans.toMatrix();
    tracing.sdf_transforms.push_back(mat);
    tracing.sdf_invTransforms.push_back(mat.getInverse());
    Transform3D b = trans * trans.getInverse();
#endif

    std::fstream fs("RayTraceGPU.ppm", std::ios::out);
    //图像分辨率
    nx = 800;
    ny = 600;
    int l = 2;
    int ns = l * l;
    fs << "P3\n" << nx << " " << ny << "\n255\n";

    vec3 col;
    float invW = 1.0 / float(nx);
    float invH = 1.0 / float(ny);
    for (int j = ny-1; j >=0; --j) {
        for (int i = 0; i < nx; ++i) {
            col.clear();
            //单个像素内随机采样多条光线计算颜色 并 取平均值
            for (int s = 0; s < ns; ++s) {
                float r1 = rand48();
                float r2 = rand48();
                float u = float(i + r1) / float(nx);
                float v = float(j + r2) / float(ny);

                //int dy = s / l;
                //int dx = s - l * dy;
                //float u = (float(i) + (float(dx)+0.5) / l) / nx;
                //float v = (float(j) + (float(dy)+0.5) / l) / ny;
                ray.origin = cam.origin;
                ray.direction = cam.lower_left_corner + cam.horizontal * u + cam.vertical * v - cam.origin;
                col += tracing.rayTracing(ray);
            }
            col = col / (float)ns;

            //gamma correction
            //col.x = Sun::sqrt(col.x);
            //col.y = Sun::sqrt(col.y);
            //col.z = Sun::sqrt(col.z);

            int ir = int(255.99 * col.x);
            int ig = int(255.99 * col.y);
            int ib = int(255.99 * col.z);

            fs << ir << " " << ig << " " << ib << "\n";
        }
    }
    fs.close();
    //tracing.rayTracing(ray);
}

void randTest() {
    GpuRandom gpuRand;
    gpuRand.wseed = 10234;
    const int n = 10000;
    int l = 100;
    std::vector<int> vec(l);
    for (int i = 0; i < n; ++i) {
        float f = gpuRand.randcore4();
        assert(f >= 0 && f <= 1.0);
        int t = int(f * l);
        vec[t] += 1;
    }

}



int main()
{
    //three_add_one();


    //testExpression();

    //gpuRayTrace();
    
  
    //generateNoiseData();
    
    int nx, ny, nn; unsigned char* tex_data = stbi_load((g_dir+"earth.jpg").c_str(), &nx, &ny, &nn, 0);
    Material* mat = new Lambertian(new image_texture(tex_data, nx, ny));
    Hitable* hitt = new SphereObj({ 0,0,-1 }, 0.5, mat);

    float a = 0;
    float b = 10 / a;
    float g = 10 + b;
    float c = b * 0.;

    srand((unsigned)time(NULL));

    std::fstream fs("RayTrace.ppm",std::ios::out);

    //图像分辨率
    nx = 1024;
    ny = 768;
    int l=1;
    int ns = l*l;
    fs<< "P3\n" << nx << " " << ny << "\n255\n";

    Camera camera = buildCamera(nx, ny);
    //Camera camera = buildCameraTwoSphere(nx, ny);
    //Camera camera = buildCameraCornellBox(nx, ny);
#if 1
    std::vector<Hitable*> vec;
    vec.push_back(new SphereObj(vec3(0, 0, -1), 0.5f, new Lambertian(new constant_texture({ 0.1,0.2,0.5 /*0.8,0.3,0.3 */}))));
    vec.push_back(new SphereObj({ 0,-100.5,-1 }, 100, new Lambertian(new constant_texture({ 0.8,0.8,0 }))));
    vec.push_back(new SphereObj({ 1,0,-1 }, 0.5, new Metal({0.8, 0.6, 0.2} ,0.3)));
    //vec.push_back(new SphereObj({ -1,0,-1 }, 0.5, new Metal({0.8, 0.8, 0.8} ,0.1)));
    //vec.push_back(new SphereObj({ -1,0,-1 }, 0.5, new Dielectrics(1.5)));
    //负半径构建中空球体
    vec.push_back(new SphereObj({ -1,0,-1 }, -0.45, new Dielectrics(1.5)));

    HitableList lis(vec);
#endif 
    //Hitable* lis = renderScene();
    //Hitable* lis = twoShpereScene();
    //Hitable* lis({ hitt });
    //Hitable* lis = twoPerlinSpheres();
    //Hitable* lis = simple_light();
    //Hitable* lis = cornell_box();

    vec3 col;
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; ++i) {
            col.clear();
            //单个像素内随机采样多条光线计算颜色 并 取平均值
            for (int s = 0; s < ns; ++s) {
                float r1 = rand48();
                float r2 = rand48();
                /*int dy = s / l;
                int dx = s - dy * l;
                float r1 = (float(dx)+0.5) / 6.f;
                float r2 = (float(dy) + 0.5) / 6.f;
                */
                float u = float(i +r1 ) / float(nx);
                float v = float(j +r2) / float(ny);
                Ray ray = camera.buildRay(u, v);
                col += rayTracing(ray, &lis, 0);
            }
            col = col / (float)ns;
            
            //gamma correction
            col.x = Math::sqrt(col.x);
            col.y = Math::sqrt(col.y);
            col.z = Math::sqrt(col.z);
            
            int ir = int(255.99 * col.x);
            int ig = int(255.99 * col.y);
            int ib = int(255.99 * col.z);

            fs << ir << " " << ig << " " << ib << "\n";
        }
    }

    fs.close();
}

