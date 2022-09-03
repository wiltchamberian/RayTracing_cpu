#ifndef __SUN_RAND_H
#define __SUN_RAND_H

namespace Sun {

	template<class _Trait>
	class Seed {

	};

	template<class _Trait>
	class Distribution : public _Trait{
	public:
		void operator() (const Seed& seed) {
			
		}
	};



}


#endif