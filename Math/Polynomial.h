#ifndef __POLYNOMIAL_H
#define __POLYNOMIAL_H

#include <vector>
#include <initializer_list>
#include <complex>

namespace Sun {

    struct Item {
        int p;
        float c;
        Item(int P, float C) :p(P), c(C) {}
        Item() :p(0), c(0) {}
    };

    class Polynomial {
    public:
        Polynomial();
        Polynomial(std::initializer_list<Item> vec);
        void addItem(const Item& item);
        int deg() const;
        float value(float t) const;
        Polynomial pow(int k) const;
        void sort();
        Polynomial& operator*= (float c);
        Polynomial operator* (float c);
        static Polynomial zero();
        static Polynomial identity();
        void partial();
        Polynomial getPartial();
        //当多项式次数<=1时，直接用求根公式求解;否则采用NewTon迭代法求解
        std::complex<float> oneRoot(float p0,int maxIterTimes = 10,float tollerance =0.001);
        void merge();
        std::vector<Item> vec_;
        friend Polynomial operator + (const Polynomial& p1, const Polynomial& p2);
    };
    
    Polynomial operator+(const Polynomial& p1, const Polynomial& p2);
    Polynomial operator*(const Polynomial& p1, const Polynomial& p2);
}


#endif