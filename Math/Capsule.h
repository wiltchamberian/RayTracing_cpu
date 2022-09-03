/*****************************************************************************
* @brief : Capsule
* Capsule �����壬����������ײ���
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