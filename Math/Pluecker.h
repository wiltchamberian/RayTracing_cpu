/*
===========================================================================



===========================================================================
*/

#ifndef __MATH_PLUECKER_H__
#define __MATH_PLUECKER_H__

/*
===============================================================================

	Pluecker coordinate

===============================================================================
*/

#include "Vector3D.h"
#include "Plane.h"
#include "MathUtils.h"
#include <memory>
#include <cassert>

#define ID_INLINE inline

namespace Sun {
	class Pluecker {
	public:
		Pluecker(void);
		explicit Pluecker(const float* a);
		explicit Pluecker(const vec3& start, const vec3& end);
		explicit Pluecker(const float a1, const float a2, const float a3, const float a4, const float a5, const float a6);

		float			operator[](const int index) const;
		float& operator[](const int index);
		Pluecker		operator-() const;											// flips the direction
		Pluecker		operator*(const float a) const;
		Pluecker		operator/(const float a) const;
		float			operator*(const Pluecker& a) const;						// permuted inner product
		Pluecker		operator-(const Pluecker& a) const;
		Pluecker		operator+(const Pluecker& a) const;
		Pluecker& operator*=(const float a);
		Pluecker& operator/=(const float a);
		Pluecker& operator+=(const Pluecker& a);
		Pluecker& operator-=(const Pluecker& a);

		bool			Compare(const Pluecker& a) const;						// exact compare, no epsilon
		bool			Compare(const Pluecker& a, const float epsilon) const;	// compare with epsilon
		bool			operator==(const Pluecker& a) const;					// exact compare, no epsilon
		bool			operator!=(const Pluecker& a) const;					// exact compare, no epsilon

		void 			Set(const float a1, const float a2, const float a3, const float a4, const float a5, const float a6);
		void			Zero(void);

		void			FromLine(const vec3& start, const vec3& end);			// pluecker from line
		void			FromRay(const vec3& start, const vec3& dir);			// pluecker from ray
		bool			FromPlanes(const Plane& p1, const Plane& p2);			// pluecker from intersection of planes
		bool			ToLine(vec3& start, vec3& end) const;					// pluecker to line
		bool			ToRay(vec3& start, vec3& dir) const;					// pluecker to ray
		void			ToDir(vec3& dir) const;									// pluecker to direction
		float			PermutedInnerProduct(const Pluecker& a) const;			// pluecker permuted inner product
		float			Distance3DSqr(const Pluecker& a) const;					// pluecker line distance

		float			Length(void) const;										// pluecker length
		float			LengthSqr(void) const;									// pluecker squared length
		Pluecker		Normalize(void) const;									// pluecker normalize
		float			NormalizeSelf(void);										// pluecker normalize

		int				GetDimension(void) const;

		const float* ToFloatPtr(void) const;
		float* ToFloatPtr(void);
		const char* ToString(int precision = 2) const;

	private:
		float			p[6];
	};

	extern Pluecker pluecker_origin;
#define pluecker_zero pluecker_origin

	ID_INLINE Pluecker::Pluecker(void) {
	}

	ID_INLINE Pluecker::Pluecker(const float* a) {
		memcpy(p, a, 6 * sizeof(float));
	}

	ID_INLINE Pluecker::Pluecker(const vec3& start, const vec3& end) {
		FromLine(start, end);
	}

	ID_INLINE Pluecker::Pluecker(const float a1, const float a2, const float a3, const float a4, const float a5, const float a6) {
		p[0] = a1;
		p[1] = a2;
		p[2] = a3;
		p[3] = a4;
		p[4] = a5;
		p[5] = a6;
	}

	ID_INLINE Pluecker Pluecker::operator-() const {
		return Pluecker(-p[0], -p[1], -p[2], -p[3], -p[4], -p[5]);
	}

	ID_INLINE float Pluecker::operator[](const int index) const {
		return p[index];
	}

	ID_INLINE float& Pluecker::operator[](const int index) {
		return p[index];
	}

	ID_INLINE Pluecker Pluecker::operator*(const float a) const {
		return Pluecker(p[0] * a, p[1] * a, p[2] * a, p[3] * a, p[4] * a, p[5] * a);
	}

	ID_INLINE float Pluecker::operator*(const Pluecker& a) const {
		return p[0] * a.p[4] + p[1] * a.p[5] + p[2] * a.p[3] + p[4] * a.p[0] + p[5] * a.p[1] + p[3] * a.p[2];
	}

	ID_INLINE Pluecker Pluecker::operator/(const float a) const {
		float inva;

		assert(a != 0.0f);
		inva = 1.0f / a;
		return Pluecker(p[0] * inva, p[1] * inva, p[2] * inva, p[3] * inva, p[4] * inva, p[5] * inva);
	}

	ID_INLINE Pluecker Pluecker::operator+(const Pluecker& a) const {
		return Pluecker(p[0] + a[0], p[1] + a[1], p[2] + a[2], p[3] + a[3], p[4] + a[4], p[5] + a[5]);
	}

	ID_INLINE Pluecker Pluecker::operator-(const Pluecker& a) const {
		return Pluecker(p[0] - a[0], p[1] - a[1], p[2] - a[2], p[3] - a[3], p[4] - a[4], p[5] - a[5]);
	}

	ID_INLINE Pluecker& Pluecker::operator*=(const float a) {
		p[0] *= a;
		p[1] *= a;
		p[2] *= a;
		p[3] *= a;
		p[4] *= a;
		p[5] *= a;
		return *this;
	}

	ID_INLINE Pluecker& Pluecker::operator/=(const float a) {
		float inva;

		assert(a != 0.0f);
		inva = 1.0f / a;
		p[0] *= inva;
		p[1] *= inva;
		p[2] *= inva;
		p[3] *= inva;
		p[4] *= inva;
		p[5] *= inva;
		return *this;
	}

	ID_INLINE Pluecker& Pluecker::operator+=(const Pluecker& a) {
		p[0] += a[0];
		p[1] += a[1];
		p[2] += a[2];
		p[3] += a[3];
		p[4] += a[4];
		p[5] += a[5];
		return *this;
	}

	ID_INLINE Pluecker& Pluecker::operator-=(const Pluecker& a) {
		p[0] -= a[0];
		p[1] -= a[1];
		p[2] -= a[2];
		p[3] -= a[3];
		p[4] -= a[4];
		p[5] -= a[5];
		return *this;
	}

	ID_INLINE bool Pluecker::Compare(const Pluecker& a) const {
		return ((p[0] == a[0]) && (p[1] == a[1]) && (p[2] == a[2]) &&
			(p[3] == a[3]) && (p[4] == a[4]) && (p[5] == a[5]));
	}

	ID_INLINE bool Pluecker::Compare(const Pluecker& a, const float epsilon) const {
		if (Math::fabs(p[0] - a[0]) > epsilon) {
			return false;
		}

		if (Math::fabs(p[1] - a[1]) > epsilon) {
			return false;
		}

		if (Math::fabs(p[2] - a[2]) > epsilon) {
			return false;
		}

		if (Math::fabs(p[3] - a[3]) > epsilon) {
			return false;
		}

		if (Math::fabs(p[4] - a[4]) > epsilon) {
			return false;
		}

		if (Math::fabs(p[5] - a[5]) > epsilon) {
			return false;
		}

		return true;
	}

	ID_INLINE bool Pluecker::operator==(const Pluecker& a) const {
		return Compare(a);
	}

	ID_INLINE bool Pluecker::operator!=(const Pluecker& a) const {
		return !Compare(a);
	}

	ID_INLINE void Pluecker::Set(const float a1, const float a2, const float a3, const float a4, const float a5, const float a6) {
		p[0] = a1;
		p[1] = a2;
		p[2] = a3;
		p[3] = a4;
		p[4] = a5;
		p[5] = a6;
	}

	ID_INLINE void Pluecker::Zero(void) {
		p[0] = p[1] = p[2] = p[3] = p[4] = p[5] = 0.0f;
	}

	/*
	 start[0],start[1],start[2],1
	   end[0],  end[1],  end[2],1
	*/
	ID_INLINE void Pluecker::FromLine(const vec3& start, const vec3& end) {
		p[0] = start[0] * end[1] - end[0] * start[1];
		p[1] = start[0] * end[2] - end[0] * start[2];
		p[2] = start[0] - end[0];
		p[3] = start[1] * end[2] - end[1] * start[2];
		p[4] = start[2] - end[2];
		p[5] = end[1] - start[1];
	}

	//µÈ¼ÛÓÚ Pluecker::FromLine(start,start+dir)
	ID_INLINE void Pluecker::FromRay(const vec3& start, const vec3& dir) {
		p[0] = start[0] * dir[1] - dir[0] * start[1];
		p[1] = start[0] * dir[2] - dir[0] * start[2];
		p[2] = -dir[0];
		p[3] = start[1] * dir[2] - dir[1] * start[2];
		p[4] = -dir[2];
		p[5] = dir[1];
	}

	ID_INLINE bool Pluecker::ToLine(vec3& start, vec3& end) const {
		vec3 dir1, dir2;
		float d;

		dir1[0] = p[3];
		dir1[1] = -p[1];
		dir1[2] = p[0];

		dir2[0] = -p[2];
		dir2[1] = p[5];
		dir2[2] = -p[4];

		d = dir2.dotProduct(dir2);
		if (d == 0.0f) {
			return false; // pluecker coordinate does not represent a line
		}

		start = dir2.crossProduct(dir1) * (1.0f / d);
		end = start + dir2;
		return true;
	}

	ID_INLINE bool Pluecker::ToRay(vec3& start, vec3& dir) const {
		vec3 dir1;
		float d;

		dir1[0] = p[3];
		dir1[1] = -p[1];
		dir1[2] = p[0];

		dir[0] = -p[2];
		dir[1] = p[5];
		dir[2] = -p[4];

		d = dir.dotProduct(dir);
		if (d == 0.0f) {
			return false; // pluecker coordinate does not represent a line
		}

		start = dir.crossProduct(dir1) * (1.0f / d);
		return true;
	}

	ID_INLINE void Pluecker::ToDir(vec3& dir) const {
		dir[0] = -p[2];
		dir[1] = p[5];
		dir[2] = -p[4];
	}

	//(0,4)+(4,0)+(1,5)+(5,1)+(2,3)+(3,2)
	ID_INLINE float Pluecker::PermutedInnerProduct(const Pluecker& a) const {
		return p[0] * a.p[4] + p[1] * a.p[5] + p[2] * a.p[3] + p[4] * a.p[0] + p[5] * a.p[1] + p[3] * a.p[2];
	}

	ID_INLINE float Pluecker::Length(void) const {
		return (float)Math::sqrt(p[5] * p[5] + p[4] * p[4] + p[2] * p[2]);
	}

	ID_INLINE float Pluecker::LengthSqr(void) const {
		return (p[5] * p[5] + p[4] * p[4] + p[2] * p[2]);
	}

	ID_INLINE float Pluecker::NormalizeSelf(void) {
		float l, d;

		l = LengthSqr();
		if (l == 0.0f) {
			return l; // pluecker coordinate does not represent a line
		}
		d = Math::invSqrt(l);
		p[0] *= d;
		p[1] *= d;
		p[2] *= d;
		p[3] *= d;
		p[4] *= d;
		p[5] *= d;
		return d * l;
	}

	ID_INLINE Pluecker Pluecker::Normalize(void) const {
		float d;

		d = LengthSqr();
		if (d == 0.0f) {
			return *this; // pluecker coordinate does not represent a line
		}
		d = Math::invSqrt(d);
		return Pluecker(p[0] * d, p[1] * d, p[2] * d, p[3] * d, p[4] * d, p[5] * d);
	}

	ID_INLINE int Pluecker::GetDimension(void) const {
		return 6;
	}

	ID_INLINE const float* Pluecker::ToFloatPtr(void) const {
		return p;
	}

	ID_INLINE float* Pluecker::ToFloatPtr(void) {
		return p;
	}

}

#endif /* !__MATH_PLUECKER_H__ */
