#ifndef __SURFACE_BUILDER_H
#define __SURFACE_BUILDER_H

#include "Math/Expression.h"
#include "Math/Vector3D.h"
#include "Math/Vector2D.h"

namespace Sun {

	void testExpression();

	inline float clamp(float x, float l, float h);

	float calSphere(const vec3& v,float r);
	//以上下大部分对象都默认中心位于原点
	AExpression buildSphere(float r);

	//v是满足x>0,y>0,z>0的顶点
	float calBox(vec3, vec3);
	AExpression buildBox(const vec3& v);

	float sdTorus(vec3 p, vec2 t);
	AExpression buildTorus(float t, float h);

	//操场形状:le表示中心到内圆心距离，r1是内圆心到腔中心线距离,r2腔半径
	//操作平铺在x-y平面
	float sdLink(vec3 p, float le, float r1, float r2);
	AExpression buildSdLink(float le, float r1, float r2);

	//infinite cylinder 无限长圆柱,垂直于x-z平面，(a,b)表示圆柱中心线与x-z平面交点，c为半径
	AExpression buildSdCyLinder(float a, float b, float c);

	//static Expression buildCone(float a, float b, float h);

	//n是平面法向，h是原点到平面距离,n指向远离原点的方向
	AExpression buildSdPlane(const vec3& n, float h);

	//Capsule/Line 胶囊体，与x-z垂直,坐在x-z平面上方，高度为h，半径为r
	//其中底半球部分嵌入x-z平面
	float sdVerticalCapsule(vec3 v, float h, float r);
	AExpression buildVerticalCapsule(float h, float r);

	//无向三角形 a,b,c是三个顶点坐标
	AExpression buildUdTriangle(const vec3& a, const vec3& b, const vec3& c);

	//moebius  k:半圈数
	float sdMobius(vec3 v, float d, float l, float r, int k);
	AExpression buildMobius(float d, float l,float r, int k);
}

#endif


