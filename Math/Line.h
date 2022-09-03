/*****************************************************************************
* @brief : Line
* @author : acedtang
* Ö±Ïß
* @date : 2021/8/5
* @version : ver 1.0
* @inparam :
* @outparam :
*****************************************************************************/

#ifndef __LINE_H
#define __LINE_H

#include "Math/Vector3D.h"

namespace Sun {

	struct Line {
		vec3 pt;
		vec3 dir; //must be normalized
	};

}

#endif