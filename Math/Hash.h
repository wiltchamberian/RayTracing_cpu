#include "Vector2D.h"
#include "Vector3D.h"
#include "Vector4D.h"

#ifndef __SUN_MATH_HASH_H
#define __SUN_MATH_HASH_H

namespace Sun {
	namespace math{
		inline void hash_combine(size_t& seed, size_t hash)
		{
			hash += 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= hash;
		}
	}
}

namespace std {

	template<typename _T>
	struct hash<Sun::Vector2D<_T>>
	{
		size_t operator()(Sun::Vector2D<_T> const& v) const {
			size_t seed = 0;
			hash<_T> hasher;
			Sun::math::hash_combine(seed, hasher(v.x));
			Sun::math::hash_combine(seed, hasher(v.y));
			return seed;
		}
	};

	template<typename _T>
	struct hash<Sun::Vector3D<_T>>
	{
		size_t operator()(Sun::Vector3D<_T> const& v) const {
			size_t seed = 0;
			hash<_T> hasher;
			Sun::math::hash_combine(seed, hasher(v.x));
			Sun::math::hash_combine(seed, hasher(v.y));
			Sun::math::hash_combine(seed, hasher(v.z));
			return seed;
		}
	};

	template<typename _T>
	struct hash<Sun::Vector4D<_T>>
	{
		size_t operator()(Sun::Vector4D<_T> const& v) const {
			size_t seed = 0;
			hash<_T> hasher;
			Sun::math::hash_combine(seed, hasher(v.x));
			Sun::math::hash_combine(seed, hasher(v.y));
			Sun::math::hash_combine(seed, hasher(v.z));
			Sun::math::hash_combine(seed, hasher(v.w));
			return seed;
		}
	};


}


#endif 