#include "SurfaceBuilder.h"
#include <cassert>

namespace Sun {

	void testExpression() {
		//self def
		{
			float a = 0.3;
			float b = 0.4;
			float c = 0.5;
			//Expression epr("sqrt(max(abs(x)-a,0)^2+max(abs(y)-b,0)^2+max(abs(z)-c,0)^2)");
			//Expression epr("min(max(abs(x) - a, max(abs(y) - b, abs(z) - c)), 0.0)");
			AExpression epr("(abs(y)-b)max(abs(z)-c)");
			epr.setSymValue("a", a);
			epr.setSymValue("b", b);
			epr.setSymValue("c", c);
			float x = 0.1;
			float y = 0.2;
			float z = 0.3;
			float k = epr.value(x,y,z);
			float i1 = std::pow(std::max<float>(fabs(x) - a, 0), 2);
			float i2 = std::pow(std::max<float>(fabs(y) - b, 0), 2);
			float i3 = std::pow(std::max<float>(fabs(z) - c, 0), 2);
			float k2 = sqrt(i1 + i2 + i3);
			//k2 = std::min<float>(std::max<float>(fabs(x) - a, std::max<float>(fabs(y) - b, fabs(z) - c)), 0.0);
			k2 = std::max<float>(fabs(y) - b, fabs(z) - c);
			assert(fabs(k-k2)<0.0001);

			AExpression epr1("min(max(abs(x) - 0.1, max(abs(y) - 0.2, abs(z) - 0.3)), 0.0)");
			k = epr1.value(x, y, z);
			k2 = std::min<float>(std::max<float>(fabs(x) - 0.1, std::max<float>(fabs(y) - 0.2, fabs(z) - 0.3)), 0.0);
			assert(fabs(k - k2) < 0.0001);
		}

		//sphere
		{
			AExpression epr = buildSphere(0.3);
			float k =epr.value(0.1, 0.2, 0.3);
			float k2 = calSphere({ 0.1,0.2,0.3 }, 0.3);
			assert(Math::equal(k, k2));
		}
		
		//box
		{
			AExpression epr = buildBox({ 0.3,0.4,0.5 });
			float k = epr.value(0.1, 0.2, 0.3);
			float k2 = calBox({ 0.1,0.2,0.3 }, { 0.3,0.4,0.5 });
			assert(Math::equal(k, k2));
		}
		//torus
		{
			AExpression epr = buildTorus(1, 0.4);
			float k = epr.value(3, 4, 5);
			float k2 = sdTorus({ 3,4,5 },{ 1,0.4 });
			assert(Math::equal(k, k2));
		}

		//sdLink
		{
			AExpression epr = buildSdLink(1, 0.3, 0.2);
			float k = epr.value(3, 4, 5);
			float k2 = sdLink({ 3,4,5 }, 1, 0.3, 0.2);
			assert(Math::equal(k, k2));
		}

		//SdCyLinder

		//SdPlane

		//sdVerticalCapsule
		{
			AExpression epr = buildVerticalCapsule(1, 0.3);
			float k = epr.value(0.1, 0.2, 0.3);
			float k2 = sdVerticalCapsule({ 0.1,0.2,0.3 }, 1, 0.3);
			assert(Math::equal(k, k2));

			k = epr.value(5, 4, 3);
			k2 = sdVerticalCapsule({ 5,4,3 }, 1, 0.3);
			assert(Math::equal(k, k2));
		}

		//for mobius
		{
			AExpression expr = buildMobius(1, 0.5, 0.2, 2);
			float k = sdMobius({ 0,0,0 }, 1, 0.5, 0.2, 2);
			float k2 = expr.value(0, 0, 0);
			assert(k == k2);
		}
	}

	float calSphere(const vec3& v,float r) {
		return v.getLength() - r;
	}

	AExpression buildSphere(float r) {
		AExpression expression("sqrt(x^2+y^2+z^2)-r");
		expression.setSymValue("r", r);
		return expression;
	}

	/*
	* float sdBox( vec3 p, vec3 b )
	{
	  vec3 q = abs(p) - b;
	  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
	}
	*/
	float calBox(vec3 p, vec3 v) {
		vec3 q = p.abs() - v;
		float t1 = std::max<float>(q.x, 0.0);
		float t2 = std::max<float>(q.y, 0.0);
		float t3 = std::max<float>(q.z, 0.0);
		return sqrt(t1 * t1 + t2 * t2 * t3 * t3) + std::min<float>(
			std::max<float>(q.x, std::max<float>(q.y, q.z)), 0.0);
	}

	AExpression buildBox(const vec3& v) {
		AExpression expression("sqrt(((abs(x)-a)max0)^2+((abs(y)-b)max0)^2+((abs(z)-c)max0)^2)\
			+(((abs(x)-a)max((abs(y)-b)max(abs(z)-c)))min 0.0)");
		expression.setSymValue("a", v.x);
		expression.setSymValue("b", v.y);
		expression.setSymValue("c", v.z);
		return expression; 
	}
	
	float sdTorus( vec3 p, vec2 t )
	{
		vec2 xy = { p.x,p.y };
		vec2 q = vec2(xy.getLength() - t.x, p.z);
		return q.getLength() - t.y;
	}

	AExpression buildTorus(float t, float h) {
		AExpression expression("sqrt((sqrt(x^2+y^2)-t)^2+z^2)-h");
		expression.setSymValue("t", t);
		expression.setSymValue("h", h);
		return expression;
	}

	
	float sdLink( vec3 p, float le, float r1, float r2 )
	{
	   vec3 q = vec3( p.x, std::max<float>(fabs(p.y)-le,0.0), p.z );
	   vec2 xy = { q.x,q.y };
	   return vec2(xy.getLength()-r1,q.z).getLength() - r2;
	}

	AExpression buildSdLink(float le, float r1, float r2) {
		AExpression expression("sqrt((sqrt(x^2+((abs(y)-le)max0)^2)-r)^2+ z^2)-s");
		expression.setSymValue("le", le);
		expression.setSymValue("r", r1);
		expression.setSymValue("s", r2);
		return expression;
	}

	AExpression buildSdCyLinder(float a, float b, float c) {
		return AExpression("sqrt((x-a)^2+(z-b)^2)-c");
	}

	//n must be normalized
	AExpression buildSdPlane(const vec3& n, float h) {
		return AExpression("x*a+y*b+z*c-h");
	}
	
	float sdVerticalCapsule(vec3 v, float h, float r) {
		v.y -= Math::clamp(v.y, 0.0, h);
		return v.getLength() - r;
	}

	AExpression buildVerticalCapsule(float h, float r) {
		AExpression ep("sqrt(x^2+(y-((y clad 0) clau h))^2+z^2)-r");
		ep.setSymValue("h", h);
		ep.setSymValue("r", r);
		return ep;
	}

	/*
	* float udTriangle( vec3 p, vec3 a, vec3 b, vec3 c )
	{
	  vec3 ba = b - a; vec3 pa = p - a;
	  vec3 cb = c - b; vec3 pb = p - b;
	  vec3 ac = a - c; vec3 pc = p - c;
	  vec3 nor = cross( ba, ac );

	  //三目运算符第一个分量前计算p在abc平面的投影点是否位于三角形内部（如果不成立，则位于内部)
	  //第二个分量计算点到边（线段)的距离
	  //第三个分量计算点到平面距离
	  return sqrt(
		(sign(dot(cross(ba,nor),pa)) +
		 sign(dot(cross(cb,nor),pb)) +
		 sign(dot(cross(ac,nor),pc))<2.0)
		 ?
		 min( min(
		 dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
		 dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
		 dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc) )
		 :
		 dot(nor,pa)*dot(nor,pa)/dot2(nor) );
	}
	*/

	AExpression buildUdTriangle(const vec3& a, const vec3& b, const vec3& c) {
		return AExpression();
	}

	float sdMobius(vec3 v, float d, float l, float r,int k) {
		float phi = atan2(v.z, v.x);
		float theta = phi * k / 2;
		float t = sqrt(v.x * v.x + v.z * v.z);
		float h = v.y;
		float t2 = (t-d) * cos(theta) + h * sin(theta);
		float h2 = (t-d) * (-sin(theta)) + h * cos(theta);
		t2 = fabs(t2);
		h2 = fabs(h2);
		t2 = t2 < l ? l : t2;
		float m = (t2 - l) * (t2 - l);
		float mn = sqrt((t2 - l) * (t2 - l) + h2 * h2);
		float dis = sqrt((t2 - l) * (t2 - l) + h2 * h2) - r;
		return dis;
	}

	AExpression buildMobius(float d, float l, float r, int k) {
		std::string str_t2 = "abs((sqrt(x*x+z*z)-d)*cos((z atan x)*k*0.5)+y*sin((z atan x)*k*0.5)) clad l";
		std::string str_h2 = "abs((d-sqrt(x*x+z*z))*sin((z atan x)*k*0.5)+y*cos((z atan x)*k*0.5))";

		std::string str = "sqrt((" + str_t2 + "-l)*(" + str_t2 + "-l)+" + str_h2 + "*" + str_h2+")" +"-r";
		AExpression expr(str.c_str());
		expr.setSymValue("d", d);
		expr.setSymValue("l", l);
		expr.setSymValue("r", r);
		expr.setSymValue("k", k);
		
		return expr;
	}

}