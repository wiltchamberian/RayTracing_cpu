/*****************************************************************************
* @brief : Capsule
* Capsule 胶囊体，用于物理碰撞检测
* @author : acedtang
* @date : 2021/8/4
* @version : ver 1.0
* @inparam :
* @outparam :
*****************************************************************************/

#ifndef __CAPSULE_H
#define __CAPSULE_H

#include "Math/Vector3D.h"

namespace Sun {

	struct Capsule {
		vec3 p;
		vec3 q;
		float r;
	};
}


#endif