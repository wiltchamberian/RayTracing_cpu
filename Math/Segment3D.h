#ifndef __SEGMENT_3D_H
#define __SEGMENT_3D_H

#include "Vector3D.h"

namespace Sun {

	class Segment3D {
	public:
		Segment3D() {}
		Segment3D(const pt3& p1, const pt3& p2) :start(p1), end(p2) {}
		pt3 start;
		pt3 end;

		float length() const {
			return (end - start).getLength();
		}

		float lengthSquare() const {
			return (end - start).getSquaredLength();
		}

		vec3 direction() const {
			return end - start;
		}

		vec3 directionNormalized() const {
			return (end - start).getNormalized();
		}
	};
}

#endif