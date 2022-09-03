/*****************************************************************************
* @brief : Cylinder
* Ô²ÖùÌå
* @author : acedtang
* @date : 2021/8/5
* @version : ver 1.0
* @inparam :
* @outparam :
*****************************************************************************/

#ifndef __CYLINDER_H
#define __CYLINDER_H

#include "Math/Vector3D.h"

namespace Sun {

	struct Cylinder {
		pt3 p;//center of bottom
		pt3 q;//center of top
		float r;//radius
	};
}



#endif