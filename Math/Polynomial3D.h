#ifndef __POLYNOMIAL_3D_H
#define __POLYNOMIAL_3D_H

#include <vector>
#include <initializer_list>
#include "Vector3D.h"
#include "Polynomial.h"

namespace Sun {

    using scalar = float;

//how to make sure the value?
#define EPS  0.01

    struct Item3D {
        int x;
        int y;
        int z;
        float c;
        Item3D()
        :x(0),y(0),z(0),c(0.f)
        {

        }
        Item3D(int X, int Y, int Z, float C) 
        :x(X),y(Y),z(Z),c(C)
        {

        }
        friend bool operator < (const Item3D& a, const Item3D& b);
        friend bool operator > (const Item3D& a, const Item3D& b);
        friend bool operator == (const Item3D& a, const Item3D& b);
        friend bool powerEqual(const Item3D& a,const Item3D& b);
    };

    inline bool powerEqual(const Item3D& a, const Item3D& b) {
        return (a.x == b.x && a.y == b.y && a.z == b.z);
    }

    inline bool operator == (const Item3D& a, const Item3D& b) {
        return (a.x == b.x && a.y == b.y && a.z == b.z);
    }

    inline bool operator < (const Item3D& a, const Item3D& b) {
        if (a.x < b.x) return true;
        else if (a.x == b.x && a.y < b.y) return true;
        else if (a.x == b.x && a.y == b.y && a.z < b.z) return true;
        return false;
    }

    inline bool operator > (const Item3D& a, const Item3D& b) {
        return b < a;
    }

//#define X3 Polynomial3D({{3,0,0,1}});
//#define X2 Polynomial3D({{2,0,0,1}});
//#define X Polynomial3D({{1,0,0,1}}); 
//#define Y3 Polynomial3D({{0,3,0,1}});
//#define Y2 Polynomial3D({{0,2,0,1}});
//#define Y Polynomial3D({{0,1,0,1}}); 
//#define Z3 Polynomial3D({{0,0,3,1}});
//#define Z2 Polynomial3D({{0,0,2,1}});
//#define Z Polynomial3D({{0,0,1,1}}); 

    class Polynomial3D {
    public:
        Polynomial3D();
        Polynomial3D(std::initializer_list<Item3D> lis);
        void addItem(const Item3D& item);
        Polynomial toPolynomial(const Polynomial& p1,const Polynomial& p2, const Polynomial& p3) const;
        static Polynomial3D zero();
        float value(float x, float y, float z) const;
        float value(const vec3& v) const;
        void merge();
        vec3 gradient(float x,float y,float z) const;
        vec3 gradient(const vec3& v) const;       
        //形式微分 df/dx;
        Polynomial3D getPartialX() const;
        Polynomial3D getPartialY() const;
        Polynomial3D getPartialZ() const;
        void partialX();
        void partialY();
        void partialZ();
        //数值微分
        float partialX(float x, float y, float z) const;
        float partialY(float x, float y, float z) const;
        float partialZ(float x, float y, float z) const;
        float partialX(const vec3& p) const;
        float partialY(const vec3& p) const;
        float partialZ(const vec3& p) const;
        void sort();
        float root(float t, const vec3& origin, const vec3& dir, int maxIterTimes = 10, float tol=0.001);
        friend Polynomial3D operator*(const Polynomial3D& p1, const Polynomial3D& p2);
        friend Polynomial3D operator+(const Polynomial3D& p1, const Polynomial3D& p2);
        friend Polynomial3D operator-(const Polynomial3D& p1, const Polynomial3D& p2);
        Polynomial3D& operator *= (float c);
        Polynomial3D operator* (float c) const;
    protected:
        std::vector<Item3D> vec_;
    };

    Polynomial3D operator*(const Polynomial3D& p1, const Polynomial3D& p2);
    Polynomial3D operator+(const Polynomial3D& p1, const Polynomial3D& p2);
    Polynomial3D operator-(const Polynomial3D& p1, const Polynomial3D& p2);
    Polynomial3D operator*(float f, const Polynomial3D& p);

    float polyRoot(const Polynomial3D& poly, const Polynomial3D& partialX ,const Polynomial3D& partialY, const Polynomial3D& partialZ
        ,float t, const vec3& origin, const vec3& dir, int maxIterTimes = 10, float tol = 0.001);
}

#endif