#pragma once

#include "common/GL/glut.h"
#include "cuda_gl_interop.h"
#include "cu_raytracing.h"
#include "Hitable.h"
#include "Math/Ray.h"
#include "Math/Sphere.h"
#include "../HitRecord.h"
#include "kernal.h"
#include "func.h"
#include "common/cpu_bitmap.h"
//#include "common/book.h"
#include "common/cpu_anim.h"
#include <stdio.h>
#include <string.h>
#include <vector>
#include <chrono>

int main1()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
#if 0
int main() {
    cudaDeviceProp prop;

    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));

    for (int i = 0; i < count; i++) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        printf(" --- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Device copy overlap: ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Kernel execition timeout : ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf(" --- Memory Information for device %d ---\n", i);
        printf("Total global mem: %ld\n", prop.totalGlobalMem);
        printf("Total constant Mem: %ld\n", prop.totalConstMem);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("Texture Alignment: %ld\n", prop.textureAlignment);
        printf(" --- MP Information for device %d ---\n", i);
        printf("Multiprocessor count: %d\n",
            prop.multiProcessorCount);
        printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n",
            prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n",
            prop.maxThreadsDim[0], prop.maxThreadsDim[1],
            prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n",
            prop.maxGridSize[0], prop.maxGridSize[1],
            prop.maxGridSize[2]);
        printf("\n");

    }

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;
    int dev = -1;
    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
    printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
    HANDLE_ERROR(cudaSetDevice(dev));

    
    constexpr int n = 100;
    int a[n];
    int b[n];
    for (int i = 0; i < n; ++i) {
        a[i] = -i;
        b[i] = i * i;
    }
    int c[n];//for output
    parrel_add(n, a, b, c);


    return 0;
}
#endif 

#ifndef DIM
#define DIM 1024
#endif

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2(void) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(unsigned char* ptr) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    int juliaValue = julia(x, y);
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char* dev_bitmap;
    CPUAnimBitmap* bitmap;
};

#if 0
int main(void) {
    DataBlock   data;
    CPUBitmap bitmap(DIM, DIM, &data);
    unsigned char* dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
    data.dev_bitmap = dev_bitmap;

    dim3 grid(DIM, DIM);
    kernel << <grid, 1 >> > (dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

    cudaFree(dev_bitmap);

    bitmap.display_and_exit();
}
#endif

//波纹动画
void cleanup(DataBlock* d) {
    cudaFree(d->dev_bitmap);
}
__global__ void kernal_anim(unsigned char* ptr, int ticks) {

    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    // now calculate the value at that position
    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrtf(fx * fx + fy * fy);

    unsigned char grey = (unsigned char)(128.0f + 127.0f *
        cos(d / 10.0f - ticks / 7.0f) /
        (d / 10.0f + 1.0f));
    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;
}
void generate_frame(DataBlock* d, int ticks) {
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernal_anim << <blocks, threads >> > (d->dev_bitmap, ticks);
    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(),
        d->dev_bitmap,
        d->bitmap->image_size(),
        cudaMemcpyDeviceToHost));
}

#if 0
int main(void) {
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap,
        bitmap.image_size()));

    bitmap.anim_and_exit((void (*)(void*, int))generate_frame,
        (void (*)(void*))cleanup);
}
#endif

////////////////////ray tracing//////////////////////////
#define INF 2e10f

#define DIM_ 512

#define rnd( x ) (x * rand() / RAND_MAX) 
#define SPHERES 20


struct TSphere {
    float r, b, g;
    float radius;
    float x, y, z;
    __device__ float hit(float ox, float oy, float* n) {
        //计算到球心的水平偏移量
        float dx = ox - x;
        float dy = oy - y;
        //偏移量若小于半径，则相交
        if (dx * dx + dy * dy < radius * radius) {
            //计算dz
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            //计算正弦角度(越大越垂直)
            *n = dz / sqrtf(radius * radius);
            //返回距离(正射投影)
            return dz + z;
        }
        return -INF;
    }
};

__global__ void kernal_test(const Sun::Ray& ray, std::vector<Sun::Hitable*>& vec) {

}

__constant__ TSphere s[SPHERES];

//单条光线的颜色采点
__global__ void kernel_ray(TSphere* local, unsigned char* ptr) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = local[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = local[i].r * fscale;
            g = local[i].g * fscale;
            b = local[i].b * fscale;
            maxz = t;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}


#if 1
int main(void) {
    // capture the start time
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    CPUBitmap bitmap(DIM, DIM);
    unsigned char* dev_bitmap;
    // allocate memory on the GPU for the output bitmap
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap,
        bitmap.image_size()));
    // allocate temp memory, initialize it, copy to constant
    // memory on the GPU, and then free our temp memory
    TSphere* temp_s = (TSphere*)malloc(sizeof(TSphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }
    HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s,
        sizeof(TSphere) * SPHERES));

    TSphere* local = NULL;
    HANDLE_ERROR(cudaMalloc(&local, sizeof(TSphere) * SPHERES));
    HANDLE_ERROR(cudaMemcpy(local, temp_s,
        sizeof(TSphere) * SPHERES,cudaMemcpyHostToDevice));

    free(temp_s);
    // generate a bitmap from our sphere data
    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel_ray << <grids, threads >> > (local,dev_bitmap);
    // copy our bitmap back from the GPU for display
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
        bitmap.image_size(),
        cudaMemcpyDeviceToHost));
    // get stop time, and display the timing results
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
        start, stop));
    printf("Time to generate: %3.1f ms\n", elapsedTime);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    // display
    bitmap.display_and_exit();
    // free our memory
    cudaFree(dev_bitmap);
}
#endif

//创建opengl纹理
GLuint createTexture2D(int width,int height) {
    
    GLuint texID;
    

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    return texID;
}

cudaSurfaceObject_t mapArrayFromOpengl(GLuint id) {
    cudaSurfaceObject_t obj;

    cudaGraphicsResource_t cudaResource;
    cudaArray* devArray;

    cudaError_t er = cudaGraphicsGLRegisterImage(&cudaResource,id,GL_TEXTURE_2D,cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&devArray, cudaResource, 0, 0);

    cudaResourceDesc desc;
    memset(&desc, 0, sizeof(cudaResourceDesc));
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = devArray;

    cudaCreateSurfaceObject(&obj ,&desc);

    return obj;
}

#if 0
int main() {
#if 0
    cudaError_t er;

    int num = 10;
    HitObject* objs = (HitObject*)malloc(sizeof(HitObject) * num);
    int useNum = 2;

    // 1
    objs[0].type = OT_SPHERE;
    CSphere* sphere = (CSphere*)malloc(sizeof(CSphere));
    objs[0].data = sphere;
    sphere->center = vec3_gen(0, 0, -1);
    sphere->r = 0.5f;
    objs[0].material.type = EMatType::MT_Metal;
    objs[0].material.dataSiz = sizeof(CMetal);
    CMetal* metal = (CMetal*)malloc(sizeof(CMetal));
    objs[0].material.data = metal;
    metal->albedo = vec3_gen(0.8, 0.6, 0.2);

    // 2
    objs[1].type = OT_SPHERE;
    CSphere* sphere2 = (CSphere*)malloc(sizeof(CSphere));
    objs[1].data = sphere;
    sphere2->center = vec3_gen(0, -100.5, -1);
    sphere2->r = 100;
    objs[1].material.type = EMatType::MT_Metal;
    objs[1].material.dataSiz = sizeof(CMetal);
    CMetal* metal2 = (CMetal*)malloc(sizeof(CMetal));
    objs[1].material.data = metal2;
    metal->albedo = vec3_gen(0.4, 0.5, 0.3);

    HitObject* devObjs = NULL;
    er = cudaMalloc((void**)&devObjs, num * sizeof(HitObject));
    cudaMemcpy(devObjs, objs, num * sizeof(HitObject), cudaMemcpyHostToDevice);

    er = cudaMalloc(&devObjs[0].data, sizeof(CSphere));
    cudaMemcpy(devObjs[0].data, objs[0].data, sizeof(CSphere), cudaMemcpyHostToDevice);
    er = cudaMalloc(&devObjs[0].material.data, objs[0].material.dataSiz);
    cudaMemcpy(devObjs[0].material.data, objs[0].material.data, objs[0].material.dataSiz, cudaMemcpyHostToDevice);
    
    er = cudaMalloc(&devObjs[1].data, sizeof(CSphere));
    cudaMemcpy(devObjs[1].data, objs[1].data, sizeof(CSphere), cudaMemcpyHostToDevice);
    er = cudaMalloc(&devObjs[1].material.data, objs[1].material.dataSiz);
    cudaMemcpy(devObjs[1].material.data, objs[1].material.data, objs[1].material.dataSiz, cudaMemcpyHostToDevice);
#endif

    

    cudaError_t er;

    int num = 10;
    HitObject* objs;
    er = cudaMallocManaged(&objs, sizeof(HitObject) * num);
    int useNum = 2;

    // 1
    objs[0].type = OT_SPHERE;
    CSphere* sphere = NULL;
    er = cudaMallocManaged(&sphere, sizeof(CSphere));
    objs[0].data = sphere;
    sphere->center = vec3_gen(0, 0, -1);
    sphere->r = 0.5f;
    objs[0].id = 0;
    objs[0].material.type = EMatType::MT_Metal;
    objs[0].material.dataSiz = sizeof(CMetal);
    CMetal* metal = NULL;
    er = cudaMallocManaged(&metal,sizeof(CMetal));
    objs[0].material.data = metal;
    metal->albedo = vec3_gen(0.8, 0.6, 0.2);

    // 2
    objs[1].type = OT_SPHERE;
    objs[1].id = 1;
    CSphere* sphere2 = NULL;
    er = cudaMallocManaged(&sphere2, sizeof(CSphere));
    objs[1].data = sphere2;
    sphere2->center = vec3_gen(0, -100.5, -1);
    sphere2->r = 100;
    objs[1].material.type = EMatType::MT_Metal;
    objs[1].material.dataSiz = sizeof(CMetal);
    CMetal* metal2 = NULL;
    er = cudaMallocManaged(&metal2, sizeof(CMetal));
    objs[1].material.data = metal2;
    metal2->albedo = vec3_gen(0.8, 0.6, 0.2);

    int nx, ny, nn;
    nx = 1024;
    ny = 1024;
    int l = 1;
    int ns = l * l;

    dim3 blocks(nx, ny);
    dim3 threads(l, l);

    //texture<float, 2> tex;

    GLuint texId = createTexture2D(nx,ny);
    cudaSurfaceObject_t surface = mapArrayFromOpengl(texId);

    CTexture texture;
    texture.width = 1024;
    texture.height = 1024;
    er = cudaMallocManaged((void**)&texture.data, texture.width * texture.height * 4);

    cvec3 lookfrom = vec3_gen(0, 0, 2);
    cvec3 lookat = vec3_gen(0, 0, -1);
    float time0 = 0.0;
    float time1 = 0.0;
    CCamera camera({ 0,0,2 }, lookat, { 0,1,0 }, 50, float(nx) / float(ny), 0, 1, time0, time1);

    // capture the start time
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    auto st = std::chrono::system_clock::now();

    cu_raytracing << <blocks, threads >> > (surface,texture,camera, objs, useNum);
    //for (int i = 0; i < 1024; ++i) {
    //    for (int j = 0; j < 1024; ++j) {
    //        host_raytracing(texture, i, j, camera, objs, useNum);
    //    }
    //}

    auto ed = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st);
    printf("cpu time:%d\n", dur.count());

    // get stop time, and display the timing results
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
        start, stop));
    printf("Time to generate: %3.1f ms\n", elapsedTime);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));


    CPUBitmap bitmap(texture.width, texture.height);
    int bitmapSiz = bitmap.image_size();
    er = cudaMemcpy(bitmap.get_ptr(), texture.data,
        bitmapSiz,
        cudaMemcpyDeviceToHost);

    

    // display
    bitmap.display_and_exit();
    cudaFree(texture.data);

    return 0;
}
#endif

void draw() {
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    //glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.0);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.0);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.0);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.0);
    glEnd();

    glFlush();
}

void opengl_init(int w,int h) {
    int c = 1;
    char* dummy = "";
    glutInit(&c, &dummy);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
    glutInitWindowSize(w, h);
    glutCreateWindow("bitmap");

    //glutKeyboardFunc(Key);
    glutDisplayFunc(draw);
}

void opengl_run() {
    glutMainLoop();
}

#if 0
int main() {
    cudaError_t er;

    int num = 10;
    HitObject* objs;
    er = cudaMallocManaged(&objs, sizeof(HitObject) * num);
    int useNum = 2;

    // 1
    objs[0].type = OT_SPHERE;
    CSphere* sphere = NULL;
    er = cudaMallocManaged(&sphere, sizeof(CSphere));
    objs[0].data = sphere;
    sphere->center = vec3_gen(0, 0, -1);
    sphere->r = 0.5f;
    objs[0].id = 0;
    objs[0].material.type = EMatType::MT_Metal;
    objs[0].material.dataSiz = sizeof(CMetal);
    CMetal* metal = NULL;
    er = cudaMallocManaged(&metal, sizeof(CMetal));
    objs[0].material.data = metal;
    metal->albedo = vec3_gen(0.8, 0.6, 0.2);

    // 2
    objs[1].type = OT_SPHERE;
    objs[1].id = 1;
    CSphere* sphere2 = NULL;
    er = cudaMallocManaged(&sphere2, sizeof(CSphere));
    objs[1].data = sphere2;
    sphere2->center = vec3_gen(0, -100.5, -1);
    sphere2->r = 100;
    objs[1].material.type = EMatType::MT_Metal;
    objs[1].material.dataSiz = sizeof(CMetal);
    CMetal* metal2 = NULL;
    er = cudaMallocManaged(&metal2, sizeof(CMetal));
    objs[1].material.data = metal2;
    metal2->albedo = vec3_gen(0.8, 0.6, 0.2);

    int dim = 16;
    int nx, ny, nn;
    nx = 1024;
    ny = 1024;
    int l = 1;
    int ns = l * l;

    dim3 blocks(nx/dim, ny/dim);
    dim3 threads(dim, dim);

    //texture<float, 2> tex;
    opengl_init(nx,ny);
    GLuint texId = createTexture2D(nx, ny);
    cudaSurfaceObject_t surface;
    cudaGraphicsResource_t cudaResource;
    cudaArray* devArray;
    er = cudaGraphicsGLRegisterImage(&cudaResource, texId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&devArray, cudaResource, 0, 0);
    cudaResourceDesc desc;
    memset(&desc, 0, sizeof(cudaResourceDesc));
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = devArray;
    cudaCreateSurfaceObject(&surface, &desc);

    cvec3 lookfrom = vec3_gen(0, 0, 2);
    cvec3 lookat = vec3_gen(0, 0, -1);
    float time0 = 0.0;
    float time1 = 0.0;
    CCamera camera({ 0,0,2 }, lookat, { 0,1,0 }, 50, float(nx) / float(ny), 0, 1, time0, time1);

    // capture the start time
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    auto st = std::chrono::system_clock::now();

    CTexture texture;
    cu_raytracing << <blocks, threads >> > (surface, camera, objs, useNum);

    // get stop time, and display the timing results
    er = cudaEventRecord(stop, 0);
    er = cudaEventSynchronize(stop);
    float elapsedTime;
    er = cudaEventElapsedTime(&elapsedTime,
        start, stop);
    printf("Time to generate: %3.1f ms\n", elapsedTime);
    er = cudaEventDestroy(start);
    er = cudaEventDestroy(stop);

    er = cudaDestroySurfaceObject(surface);
    er = cudaGraphicsUnmapResources(1, &cudaResource, 0);
    er = cudaStreamSynchronize(0);

    auto ed = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st);
    printf("cpu time:%d\n", dur.count());

    opengl_run();

    return 0;
}
#endif
