/*****************************************************************************
* @brief : Brush
* @author : acedtang
* brush表示3d空间中的一个多面体，其表示方法是用该多面体所包含的面来表示的，
* 每个面用一个平面表示，这种表示方法便于做碰撞检测相关处理
* @date : 2021/8/5
* @version : ver 1.0
* @inparam :
* @outparam :
*****************************************************************************/

#ifndef __BRUSH_H
#define __BRUSH_H

#include <vector>
#include "Math/Plane.h"

namespace Sun {

	//将brush根据面数来分类
	struct Brush6 {	
		Plane plane[6];
	};

	struct Brush {
		std::vector<Plane> planes;
	};
}

#endif