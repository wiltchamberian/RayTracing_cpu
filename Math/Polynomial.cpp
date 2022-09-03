#include "Polynomial.h"
#include <float.h>
#include <algorithm>
#include <cassert>

namespace Sun {
#define EPS 0.001
    Polynomial::Polynomial() {

    }

    Polynomial::Polynomial(std::initializer_list<Item> vec)
        :vec_(vec)
    {
        merge();
    }

    Polynomial Polynomial::zero() {
        return Polynomial();
    }

    Polynomial Polynomial::identity() {
        Polynomial res;
        res.addItem(Item(0, 1));
        return res;
    }

    void Polynomial::partial() {
        for (auto& it : vec_) {
            it.c *= it.p;
            it.p -= 1;
        }
        for (auto it = vec_.begin(); it != vec_.end();) {
            if (it->p < 0) {
                it = vec_.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    Polynomial Polynomial::getPartial() {
        Polynomial res = *this;
        res.partial();
        return res;
    }

    std::complex<float> Polynomial::oneRoot(float p0,int maxIterTimes,float tollerance){
        std::complex<float> r;
        merge();
        int d = deg();
        if (d <= 0) return std::complex<float>();
        if (d == 1) {
            if (vec_[0].p == 1) {
                return { 0,0 };
            }
            else {
                return { -vec_[0].c / vec_[1].c,0.f };
            }
        }
        //else if (d == 2) {
        //    //ax^2+bx+c=0
        //    float a = 1; 
        //    float b = 0;
        //    float c = 0;
        //    for (auto& it : vec_) {
        //        if (it.p == 0) {
        //            c == it.c;
        //        }
        //        else if (it.p == 1) {
        //            b = it.c;
        //        }
        //        else {
        //            a = it.c;
        //        }
        //    }
        //    return (-b + sqrt(b * b - 4 * a * c)) * (1 / (2 * a));
        //}

        //use newton method
        if (fabs(value(p0)) < EPS)
            return p0;
        Polynomial par = getPartial();
        float p = p0;
        for (int i = 0; i < maxIterTimes; ++i) {
            float k = par.value(p0);
            p = p0 - value(p0) / k;
            if (fabs(p - p0) < tollerance) {
                return p;
            }
            p0 = p;
        }
        return std::complex<float>(0, 1);
    }

    void Polynomial::merge() {
        sort();
        std::vector<Item> vec;
        int l = vec_.size();
        for (int i = 0; i < l; ++i) {
            Item item = vec_[i];
            while ((i + 1) < l && vec_[i + 1].p == item.p) {
                item.c += vec_[i + 1].c;
                ++i;
            }
            if (!(fabs(item.c)<DBL_EPSILON)) {
                vec.push_back(item);
            }     
        }
        std::swap(vec_, vec);
        return;
    }

    void Polynomial::addItem(const Item& item) {
        //for (auto& it : vec_) {
        //    if (it.p == item.p) {
        //        it.c += item.c;
        //        break;
        //    }
        //}
        vec_.push_back(item);
    }

    void Polynomial::sort() {
        std::sort(vec_.begin(), vec_.end(), [](const Item& t1, const Item& t2)->bool {
            return t1.p < t2.p;
            });
    }

    Polynomial& Polynomial::operator*= (float c) {
        if (fabs(c)<DBL_EPSILON) {
            vec_.clear();
        }
        for (auto& it : vec_) {
            it.c *= c;
        }
        return *this;
    }

    Polynomial Polynomial::operator* (float c) {
        Polynomial res = *this;
        res *= c;
        return res;
    }

    int Polynomial::deg() const {
        if (vec_.empty()) return -1;
        int max = 0;
        for (auto& it : vec_) {
            max = std::max<int>(it.p, max);
        }
        return max;
    }

    float Polynomial::value(float t) const {
        float res = 0.f;
        for (auto& item : vec_) {
            res += item.c * std::pow(t, item.p);
        }
        return res;
    }

    Polynomial Polynomial::pow(int k) const {
        if (k < 0) { assert(false); return {}; }
        if (k == 0) return Polynomial::identity();
        Polynomial res = *this;
        for (int i = 0; i < k - 1; ++i) {
            res = res * (*this);
        }
        return res;
    }

    Polynomial operator+(const Polynomial& p1, const Polynomial& p2) {
        Polynomial t1 = p1;
        Polynomial t2 = p2;
        t1.sort();
        t2.sort();
        int d1 = t1.deg();
        int d2 = t2.deg();
        //类似归并排序的方式合并
        int i = 0;
        int j = 0;
        Polynomial res;
        while (i < t1.vec_.size() && j < t2.vec_.size()) {
            if (t1.vec_[i].p == t2.vec_[j].p) {
                res.addItem(Item(t1.vec_[i].p, t1.vec_[i].c + t2.vec_[j].c));
                i++;
                j++;
                continue;
            }
            if (t1.vec_[i].p < t2.vec_[j].p) {
                res.addItem(t1.vec_[i]);
                i++;
                continue;
            }
            if (t1.vec_[i].p > t2.vec_[j].p) {
                res.addItem(t2.vec_[j]);
                j++;
                continue;
            }
        }
        if (i < t1.vec_.size()) {
            for (int k = i; k < t1.vec_.size(); ++k) {
                res.addItem(t1.vec_[k]);
            }
        }
        if (j < t2.vec_.size()) {
            for (int k = j; k < t2.vec_.size(); ++k) {
                res.addItem(t2.vec_[k]);
            }
        }
        return res;
    }

    Polynomial operator*(const Polynomial& p1, const Polynomial& p2) {
        Polynomial res;
        for (auto& it : p1.vec_) {
            for (auto& it2 : p2.vec_) {
                Item tmp;
                tmp.p = it.p + it2.p;
                tmp.c = it.c * it2.c;
                res.vec_.push_back(tmp);
            }
        }
        res.merge();
        return res;
    }
}