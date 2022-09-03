#ifndef __SURFACE_BUILDER_H
#define __SURFACE_BUILDER_H

#include "Math/Expression.h"
#include "Math/Vector3D.h"
#include "Math/Vector2D.h"

namespace Sun {

	void testExpression();

	inline float clamp(float x, float l, float h);

	float calSphere(const vec3& v,float r);
	//�����´󲿷ֶ���Ĭ������λ��ԭ��
	AExpression buildSphere(float r);

	//v������x>0,y>0,z>0�Ķ���
	float calBox(vec3, vec3);
	AExpression buildBox(const vec3& v);

	float sdTorus(vec3 p, vec2 t);
	AExpression buildTorus(float t, float h);

	//�ٳ���״:le��ʾ���ĵ���Բ�ľ��룬r1����Բ�ĵ�ǻ�����߾���,r2ǻ�뾶
	//����ƽ����x-yƽ��
	float sdLink(vec3 p, float le, float r1, float r2);
	AExpression buildSdLink(float le, float r1, float r2);

	//infinite cylinder ���޳�Բ��,��ֱ��x-zƽ�棬(a,b)��ʾԲ����������x-zƽ�潻�㣬cΪ�뾶
	AExpression buildSdCyLinder(float a, float b, float c);

	//static Expression buildCone(float a, float b, float h);

	//n��ƽ�淨��h��ԭ�㵽ƽ�����,nָ��Զ��ԭ��ķ���
	AExpression buildSdPlane(const vec3& n, float h);

	//Capsule/Line �����壬��x-z��ֱ,����x-zƽ���Ϸ����߶�Ϊh���뾶Ϊr
	//���еװ��򲿷�Ƕ��x-zƽ��
	float sdVerticalCapsule(vec3 v, float h, float r);
	AExpression buildVerticalCapsule(float h, float r);

	//���������� a,b,c��������������
	AExpression buildUdTriangle(const vec3& a, const vec3& b, const vec3& c);

	//moebius  k:��Ȧ��
	float sdMobius(vec3 v, float d, float l, float r, int k);
	AExpression buildMobius(float d, float l,float r, int k);
}

#endif


