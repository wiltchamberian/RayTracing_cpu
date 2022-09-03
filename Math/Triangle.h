#ifndef __TRIANGLE_H
#define __TRIANGLE_H

#include "Vector3D.h"

namespace Sun {

	class Triangle {
	public:
		vec3 v0;
		vec3 v1;
		vec3 v2;

		inline const vec3& operator[](int i) const {
			return *((vec3*)(this) + i);
		}

		inline vec3& operator[](int i) {
			return *((vec3*)(this) + i);
		}
	};

}


#endif