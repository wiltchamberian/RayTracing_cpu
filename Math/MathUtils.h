#ifndef MATHUTILS_H
#define MATHUTILS_H

#include "NumericLimits.h"
#include "MathConstants.h"
#include <cmath>
#include <cstdint>
#include <type_traits>

/**
 * @projectName   SoftRenderer
 * @brief         math auxiliary functions.
 * @author        YangWC
 * @date          2019-04-28
 */
namespace Sun
{
typedef unsigned int uint;

#define PI_ANGLE 180
#define SMALL_NUMBER		(1.e-8f)
#define KINDA_SMALL_NUMBER	(1.e-4f)

	namespace Math {
		inline bool equal(double a, double b)
		{
			double d = a - b;
			d = d < 0 ? (-d) : d;
			return d < EPSILON_dbl;
		}

		inline double radians(double angle)
		{
			return angle * A_PI / PI_ANGLE;
		}

		inline double angles(double radians)
		{
			return radians * PI_ANGLE / A_PI;
		}

		inline float fabs(float number) {
			//return number > 0 ? number : (-number);
			int tmp = *reinterpret_cast<int*>(&number);
			tmp &= 0x7FFFFFFF;
			return *reinterpret_cast<float*>(&tmp);
		}

		inline float invSqrt(float number) {
			return 1 / ::sqrt(number);
		}

		//逆平方根算法(from quake3)
		inline float quickInvSqrt(float number) {
			long i;
			float x2, y;
			const float threehalfs = 1.5F;

			x2 = number * 0.5F;
			y = number;
			i = *(long*)&y;						// evil floating point bit level hacking
			i = 0x5f3759df - (i >> 1);               // what the fuck?
			y = *(float*)&i;
			y = y * (threehalfs - (x2 * y * y));   // 1st iteration

			return y;
		}

		inline float sqrt(float x) {
			//return x * invSqrt(x);
			return ::sqrt(x);
		}

		////////////////////////////////////////////////////////////
		//求上界
		template <typename T>
		T upperBound(const T& t) {
			return t;
		}

		template <typename T, typename ... Args>
		T upperBound(const T& t, Args ... args) {
			T c = upperBound(args...);
			return c > t ? c : t;
		}
		//求下界
		template <typename T>
		T lowerBound(const T& t) {
			return t;
		}

		template <typename T, typename ... Args>
		inline T lowerBound(const T& t, Args ... args) {
			T c = lowerBound(args...);
			return c < t ? c : t;
		}

		//线性插值
		template<typename _Key, typename _Iter>
		typename _Iter::value_type interpolate_linear(_Iter start, _Iter end, const _Key& key) {
			for (_Iter it = start; it != end; ++it) {
				if (key < it->key()) {
					if (it == start) {
						return *it;
					}
					else {
						_Iter pre = it;
						--pre;
						double frac = (key - pre->key()) / (it->key() - pre->key());
						typename _Iter::value_type res;
						res.value() = pre->value() * (1 - frac) + it->value() * frac;
						res.key() = key;
						return res;
					}
				}
				else if (key > it->key()) {

				}
				else {
					return *it;
				}
			}
			if (end != start) {
				return *(--end);
			}
			return _Iter::value_type();
		}

		inline void mat2qua(float(*m)[3], float* qua)
		{
			float q1 = sqrt(m[0][0] + m[1][1] + m[2][2] + 1) / 2;
			float q2, q3, q4, tr, s;
			if (q1 != 0.0) {
				q2 = (m[2][1] - m[1][2]) / 4 / q1;
				q3 = (m[0][2] - m[2][0]) / 4 / q1;
				q4 = (m[1][0] - m[0][1]) / 4 / q1;
			}
			else {
				tr = m[0][0] + m[1][1] + m[2][2];
				if (tr > 0) {
					s = sqrt(tr + 1.0) * 2;
					q1 = 0.25 * s;
					q2 = (m[2][1] - m[1][2]) / s;
					q3 = (m[0][2] - m[2][0]) / s;
					q4 = (m[1][0] - m[0][1]) / s;
				}
				else if ((m[0][0] > m[1][1]) && (m[0][0] > m[2][2])) {
					s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2;
					q1 = (m[2][1] - m[1][2]) / s;
					q2 = 0.25 * s;
					q3 = (m[0][1] + m[1][0]) / s;
					q4 = (m[0][2] + m[2][0]) / s;
				}
				else if (m[1][1] > m[2][2])
				{
					s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2;
					q1 = (m[0][2] - m[2][0]) / s;
					q2 = (m[0][1] + m[1][0]) / s;
					q3 = 0.25 * s;
					q4 = (m[1][2] + m[2][1]) / s;
				}
				else {
					s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2;
					q1 = (m[1][0] - m[0][1]) / s;
					q2 = (m[0][2] + m[2][0]) / s;
					q3 = (m[1][2] + m[2][1]) / s;
					q4 = 0.25 * s;
				}
			}
			qua[0] = q1;
			qua[1] = q2;
			qua[2] = q3;
			qua[3] = q4;
		}

		inline float clamp(float v, float l, float h) {
			return v < l ? l : (v > h ? h : v);
		}

		//判断一个数字是否是2的整数次幂
		template<class _T>
		inline bool isPowerOfTwo(_T d) {
			return !(d & (d - 1));
		}

		//aligns a value to the nearest higher multiple of 'alignment', which must be a power of two.
		template <typename T>
		inline constexpr T align(T Val, std::uint64_t alignment)
		{
			static_assert(std::is_integral<T>::value || std::is_pointer<T>::value, "align expects an integer or pointer type");

			//找到>=Val的最小的为alignment整数倍的整数
			return (T)(((std::uint64_t)Val + alignment - 1) & ~(alignment - 1));
		}

		//aligns a value to the nearest lower multiple of 'alignment', which must be a power of two
		template <typename T>
		inline constexpr T alignDown(T Val, std::uint64_t alignment)
		{
			static_assert(std::is_integral<T>::value || std::is_pointer<T>::value, "alignDown expects an integer or pointer type");
			//设Alignment =2^x
			//这段代码的意思是将Val的后x位置为0，从而达到向下取整到Alignment整数倍的目的
			return (T)(((std::uint64_t)Val) & ~(alignment - 1));
		}

		//find neareset  higher or equal  number which is a power of 2 
		//思路：对于非2的整数次幂：如 1001101b, 将其变为1111111b,最后+1;
		//对于2的整数次幂如 10000b,将其变为1111b,最后+1
		inline constexpr int align_power2(int val) {
			int n = val - 1;
			n = n | (n >> 1);
			n = n | (n >> 2);
			n = n | (n >> 4);
			n = n | (n >> 8);
			n = n | (n >> 16);
			return n + 1;
		}

		//转置方阵
		template<class _Iterator>
		void transpose(_Iterator input, int rowCount) {
			for (int i = 0; i < rowCount; ++i) {
				for (int j = i + 1; j < rowCount; ++j) {
					std::swap(input[i][j], input[j][i]);
				}
			}
		}

	}
	
}

#endif // MATHUTILS_H
