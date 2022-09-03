#include "Polynomial3D.h"
#include <algorithm>
#include <cmath>


namespace Sun {

    Polynomial3D::Polynomial3D() {

    }

    Polynomial3D::Polynomial3D(std::initializer_list<Item3D> lis)
        :vec_(lis)
    {
        merge();
    }

    void Polynomial3D::addItem(const Item3D& item) {
        //can opt to binary search if sorted TODO!
        for (int i = 0; i < vec_.size(); ++i) {
            if (powerEqual(vec_[i], item)) {
                vec_[i].c += item.c;
                return;
            }
        }
        vec_.push_back(item);
    }

    Polynomial Polynomial3D::toPolynomial(const Polynomial& p1, const Polynomial& p2, const Polynomial& p3) const {
        Polynomial poly = Polynomial::zero();
        for (auto& it : vec_) {
            Polynomial tmp = p1.pow(it.x)*p2.pow(it.y)*p3.pow(it.z)*it.c;
            poly = poly + tmp;
        }
        return poly;
    }

    Polynomial3D Polynomial3D::zero() {
        return Polynomial3D();
    }

    float Polynomial3D::value(float x, float y, float z) const{
        float res = 0.f;
        for (const Item3D& item: vec_) {
            res += item.c * (std::pow(x, item.x) * std::pow(y, item.y) * std::pow(z, item.z));
        }
        return res;
    }

    float Polynomial3D::value(const vec3& v) const {
        return value(v.x, v.y, v.z);
    }

    void Polynomial3D::merge() {
        sort();
        std::vector<Item3D> vec2;
        int l = vec_.size();
        for (int i = 0; i < l; ++i) {
            Item3D item = vec_[i];
            while ((i + 1) < l && powerEqual(vec_[i], vec_[i + 1])) {
                item.c += vec_[i + 1].c;
                i += 1;
            }
            vec2.push_back(item);
        }
        std::swap(vec_, vec2);
    }

    vec3 Polynomial3D::gradient(float x, float y, float z) const {
        vec3 res;
        Polynomial3D partialX = this->getPartialX();
        Polynomial3D partialY = this->getPartialY();
        Polynomial3D partialZ = this->getPartialZ();
        res = { partialX.value(x,y,z),partialY.value(x,y,z), partialZ.value(x,y,z) };
        return res;
    }

    vec3 Polynomial3D::gradient(const vec3& v) const {
        return gradient(v.x, v.y, v.z);
    }

    Polynomial3D Polynomial3D::getPartialX() const{
        Polynomial3D res = *this;
        res.partialX();
        return res;
    }

    Polynomial3D Polynomial3D::getPartialY() const{
        Polynomial3D res = *this;
        res.partialY();
        return res;
    }

    Polynomial3D Polynomial3D::getPartialZ() const{
        Polynomial3D res = *this;
        res.partialZ();
        return res;
    }

    void Polynomial3D::partialX() {
        for (auto& it : vec_) {
            it.c *= it.x;
            it.x -= 1;
        }
        for (auto it = vec_.begin(); it != vec_.end(); ) {
            if (it->x < 0) {
                it = vec_.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    void Polynomial3D::partialY() {
        for (auto& it : vec_) {
            it.c *= it.y;
            it.y -= 1;
        }
        for (auto it = vec_.begin(); it != vec_.end(); ) {
            if (it->y < 0) {
                it = vec_.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    void Polynomial3D::partialZ() {
        for (auto& it : vec_) {
            it.c *= it.z;
            it.z -= 1;
        }
        for (auto it = vec_.begin(); it != vec_.end(); ) {
            if (it->z < 0) {
                it = vec_.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    float Polynomial3D::partialX(float x, float y, float z) const {
        Polynomial3D tmp = this->getPartialX();
        return tmp.value(x, y, z);
    }

    float Polynomial3D::partialY(float x, float y, float z) const {
        Polynomial3D tmp = this->getPartialY();
        return tmp.value(x, y, z);
    }

    float Polynomial3D::partialZ(float x, float y, float z) const {
        Polynomial3D tmp = this->getPartialZ();
        return tmp.value(x, y, z);
    }

    float Polynomial3D::partialX(const vec3& p) const {
        return partialX(p.x, p.y, p.z);
    }

    float Polynomial3D::partialY(const vec3& p) const {
        return partialY(p.x, p.y, p.z);
    }

    float Polynomial3D::partialZ(const vec3& p) const {
        return partialZ(p.x, p.y, p.z);
    }

    void Polynomial3D::sort() {
        std::sort(vec_.begin(), vec_.end(), [](const Item3D& a,const Item3D& b)->bool {
            if (a.x < b.x) return true;
            else if (a.x==b.x && a.y < b.y) return true;
            else if (a.x==b.x && a.y==b.y && a.z < b.z) return true;
            return false;
            });
    }

    float Polynomial3D::root(float t, const vec3& origin, const vec3& dir, int maxIterTimes, float tol) {
        //use fix point f(p)=0 => g(t)=t-f(t) has fix point g(p) = p
        vec3 pt;
        for (int i = 0; i < maxIterTimes; ++i) {
            pt = origin + dir * t;
            float g = t - value(pt);
            if (fabs(g - t) < tol) {
                return g;
            }
            t = g;
        }
        return NAN;
    }

    Polynomial3D& Polynomial3D::operator *= (float c) {
        if (Math::equal(c, 0.f)) {
            vec_.clear();
        }
        for (auto& it : vec_)
        {
            it.c *= c;
        }
        return *this;
    }

    Polynomial3D Polynomial3D::operator* (float c) const {
        if (Math::equal(c, 0.f))return Polynomial3D::zero();
        Polynomial3D res = *this;
        for (auto& it:res.vec_) {
            it.c *= c;
        } 
        return res;
    }

    Polynomial3D operator*(const Polynomial3D& p1, const Polynomial3D& p2) {
        Polynomial3D poly;
        std::vector<Item3D> vec;
        for (auto& it : p1.vec_) {
            for (auto& it2 : p2.vec_) {
                Item3D item;
                item.x = it.x + it2.x;
                item.y = it.y + it2.y;
                item.z = it.z + it2.z;
                item.c = it.c * it2.c;
                vec.push_back(item);
            }
        }
        poly.vec_ = std::move(vec);
        poly.merge();
        return poly;
    }

    Polynomial3D operator+(const Polynomial3D& p1, const Polynomial3D& p2) {
        Polynomial3D t1 = p1;
        Polynomial3D t2 = p2;
        t1.sort();
        t2.sort();
        int i = 0;
        int j = 0;
        Polynomial3D res;
        while (i < t1.vec_.size() && j < t2.vec_.size()) {
            if (t1.vec_[i] == t2.vec_[j]) {
                Item3D tmp = t1.vec_[i];
                tmp.c += t2.vec_[j].c;
                res.addItem(tmp);
                i++;
                j++;
                continue;
            }
            if (t1.vec_[i] < t2.vec_[j]) {
                res.addItem(t1.vec_[i]);
                i++;
                continue;
            }
            if (t1.vec_[i] > t2.vec_[j]) {
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

    Polynomial3D operator-(const Polynomial3D& p1, const Polynomial3D& p2) {
        Polynomial3D poly = p2;
        for (auto& it : poly.vec_) {
            it.c = -it.c;
        }
        poly = poly + p1;
        return poly;
    }

    Polynomial3D operator*(float f, const Polynomial3D& p) {
        return p * f;
    }

    float polyRoot(const Polynomial3D& poly, const Polynomial3D& partialX, const Polynomial3D& partialY, const Polynomial3D& partialZ,
        float t, const vec3& origin, const vec3& dir, int maxIterTimes, float tol) {
        vec3 p = origin + dir * t;
        float r = poly.value(p);
        if (Math::equal(r, 0.0)) return t;

        for (int i = 0; i < maxIterTimes; ++i) {
            p = origin + dir * t;
            //float k = partial.value(p);
            float k = partialX.value(p) * dir.x + partialY.value(p) * dir.y + partialZ.value(p) * dir.z;
            float p2 = t - poly.value(p) / k;
            float d = p2 - t;
            if (fabs(d) < tol) {
                return p2;
            }
            t = p2;
        }
        return NAN;
    }
}