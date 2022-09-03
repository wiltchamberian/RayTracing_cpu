/*****************************************************************************
* @brief : Brush
* @author : acedtang
* brush��ʾ3d�ռ��е�һ�������壬���ʾ�������øö�������������������ʾ�ģ�
* ÿ������һ��ƽ���ʾ�����ֱ�ʾ������������ײ�����ش���
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

	//��brush��������������
	struct Brush6 {	
		Plane plane[6];
	};

	struct Brush {
		std::vector<Plane> planes;
	};
}

#endif