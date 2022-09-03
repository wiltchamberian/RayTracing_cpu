#ifndef __FUNCTIONS_H
#define __FUNCTIONS_H

#include "Math/Vector2D.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include <algorithm>
#include <cassert>
#include "Math/Sphere.h"
#include "Math/Box.h"
#include "Math/Plane.h"
#include "Math/Segment3D.h"
#include "Math/Ray.h"
#include "Math/Triangle.h"

namespace Sun {

	inline float dot(const vec3& a, const vec3& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	inline float dot2(const vec3& v) {
		return dot(v, v);
	}

	inline vec3 cross(const vec3& a, const vec3& b) {
		return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.z);
	}

	inline float scalarTripleProduct(const vec3& a, const vec3& b, const vec3& c) {
		return dot(a, cross(b, c));
	}
	
	//normal should be normalized , v is away from surface
	inline vec3 reflect(const vec3& v, const vec3& normal) {
		return normal* dot(v, normal) * 2 - v;
	}

	inline float sign(float a) {
		return a > 0 ? 1 : (a < 0 ? -1 : 0);
	}

	inline float length(const vec3& v) {
		return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	}

	//点向平面的投影点 (normal should normalized)
	inline vec3 proj(const vec3& pt, const vec3& normal, float dirDis) {
		float dis = dot(pt, normal) - dirDis;
		vec3 pro = pt - normal * dis;
		return pro;
	}

	inline vec3 clamp(const vec3& v, float l, float h) {
		vec3 res;
		res.x = Math::clamp(v.x, l, h);
		res.y = Math::clamp(v.y, l, h);
		res.z = Math::clamp(v.z, l, h);
		return res;
	}

	inline vec4 clamp(const vec4& v, float l, float h) {
		vec4 res;
		res.x = Math::clamp(v.x, l, h);
		res.y = Math::clamp(v.y, l, h);
		res.z = Math::clamp(v.z, l, h);
		res.w = Math::clamp(v.w, l, h);
		return res;
	}

	//三角形插值  ,pt是三角形p0p1p2所在平面上一点，t1,t2,t3对应p0,p1,p2处的值，插值pt处的值
	template<class _T>
	inline _T triangle_interpolate(const vec3& pt, const vec3& p0, const vec3& p1,const vec3& p2,const _T& t0,const _T& t1,const _T& t2) {
		vec3 v = pt - p0;
		vec3 v1 = p1 - p0;
		vec3 v2 = p2 - p0;
		//we want to get that v=a*v1+b*v2 with a,b scalar, this can trans to matrix form
		// A*(a,b)^T = v , A=(v1,v2)  ,but A is not square matrix ,so we try to get least-square solution
		// let x = (a,b)^T  , we def f(x) = (Ax-v)^T*(Ax-v),to get the minimum ,solve df/dx=0
		//f(x)= x^T(A^T*A)x - 2v^T*A*x  +v^T*v ,let A^T*A =B
		//df/dx = 2*Bx - 2A^T*v =0 => Bx=A^T*v ,this can be solved.
		//so we try to cal B= A^T*A first
		float b00 = dot(v1, v1);
		float b01 = dot(v1, v2);
		float b10 = dot(v2, v1);
		float b11 = dot(v2, v2);
		//cal A^T*v
		vec2 item;
		item.x = dot(v, v1) ;
		item.y = dot(v, v2) ;
		//B^(-1)
		float det = b00 * b11 - b01 * b10;
		float invb00 = b11/det;
		float invb01 = -b01/det;
		float invb10 = -b10/det;
		float invb11 = b00 / det;

		float a = invb00 * item.x + invb01 * item.y;
		float b = invb10 * item.x + invb11 * item.y;

		//so v = a*v1+b*v2;we can use this to interpolate by
		_T t_m = t0*(1-a) + t1*a;
		_T t_n = t0*(1-b) + t2*b;
		_T t_mid = (t_m + t_n);

		_T t = t_mid - t0;

		return t;
	}

	//计算线段上到某个固定点距离为r的点对应的百分比
	inline bool solve_segment_point(const vec3& start, const vec3& end, float r, const vec3& p,float& frac) {
		vec3 d = end - start;
		//At^2+Bt+C= 0;
		float A = d.x * d.x + d.y * d.y + d.z * d.z;
		float B = 2 * d.x * (start.x - p.x) + 2 * d.y * (start.y - p.y) + 2 * d.z * (start.z - p.z);
		float C = (start.x - p.x) * (start.x - p.x) + (start.y - p.y) * (start.y - p.y) + (start.z - p.z) * (start.z - p.z) - r * r;

		float delta = B * B - 4 * A * C;
		if (delta < 0) {
			return false;
		}
		float sqr = sqrt(delta);
		float t1 = (-B + sqr) / (2 * A);
		float t2 = (-B - sqr) / (2 * A);
		if (t1 > t2) std::swap(t1, t2);

		if (t1 > 1) return false;
		if (t2 < 0) return false;
		frac = t1 >= 0 ? t1 : t2;
		return true;
	}

	inline AABB transformSphereToAABB(const Sphere& sphere) {
		AABB box;
		box.mins_ = sphere.center + vec3(-sphere.r, -sphere.r, -sphere.r);
		box.maxs_ = sphere.center + vec3(sphere.r, sphere.r, sphere.r);
		return box;
	}

	//计算直线的普吕克坐标
	inline void plucker(const vec3& a, const vec3& b, float coor[]) {
		coor[0] = a[0] * b[1] - a[1] * b[0];
		coor[1] = a[0] * b[2] - a[2] * b[0];
		coor[3] = a[0] - b[0];
		coor[4] = a[1] * b[2] - a[2] * b[1];
		coor[5] = a[2] - b[2];
		coor[6] = b[1] - a[1];
	}

	//普吕克置换内积
	inline float permutedInnerProduct(float l1[6], float l2[6]) {
		return l1[0] * l2[4] + l1[4] * l2[0] + l1[1] * l2[5] + l1[5] * l2[1] + l1[2] * l2[3] + l1[3] * l2[2];
	}
	
	//给定直线l1(a,b),l2(c,d),以及直线l1的运动矢量dir;求t
	//使得l1+dir*t 和l2相交，如果不存在满足条件的t，返回false,并令t=1(令t为无穷?); 计算方法，普吕克坐标
	//注意，返回的t可能>1.0
	inline bool fracTranslateLineThrougLine
	(const vec3& va, const vec3& vb, const vec3& vc, const vec3& vd, const vec3& dir, float* t) {
		assert(t != nullptr);
		vec3 ld = va - vb;
		
		float tmp_t = 0.0f;

		//计算l1的普吕克坐标和l2的普吕克坐标
		float l1[6];
		float l2[6];
		plucker(va, vb, l1);
		plucker(vc, vd, l2);

		float cross[3];
		cross[0] = ld[0] * dir[1] - ld[1] * dir[0];
		cross[1] = ld[0] * dir[2] - ld[2] * dir[0];
		cross[2] = ld[1] * dir[2] - ld[2] * dir[1];

		float d = l2[4] * cross[0] + l2[5] * cross[1] + l2[2] * cross[2];
		if (Math::equal(d, 0.0f)) {
			*t = 1.0;
			return false;
		}
		*t = -permutedInnerProduct(l1, l2);
		*t = (*t) / d;
		
		return true;
	}
	
}

#endif