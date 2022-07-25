/*
urand.c

uniform random number (0-1)
*/

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

/*
Park and Miller's random number
Numerical Recipe, p.206
seed : I/O : seed (>0)
return : 0-1
*/
static double ran0_nr(int64_t *seed)
{
	const int64_t a = 16807;
	const int64_t m = 2147483647;  // = 2^32 - 1
	const int64_t q = 127773;
	const int64_t r = 2836;
	const double minv = 1.0 / m;

	const int64_t k = (*seed) / q;
	*seed = a * (*seed - (k * q)) - r * k;
	if (*seed < 0) {
		*seed += m;
	}

	return minv * (*seed);
}
/*
generate a random number (0-1) (period = 1664501)

M.Mori p.43 1988
*/
static double urand_1664501(int64_t *r)
{
	const int64_t m = 1664501;
	const int64_t a = 1229;
	const int64_t c = 351750;
	const double minv = 1.0 / m;

	*r = (a * (*r) + c) % m;
	return (*r) * minv;
}


//#define MBIG 1000000000
//#define MSEED 161803398
//#define MZ 0
//#define FAC (1.0/MBIG)
static double ran3_nr(int64_t *idum)
{
	static int inext,inextp;
	static int64_t ma[56];
	static int iff=0;
	const int64_t MBIG = 1000000000;
	const int64_t MSEED = 161803398;
	const int64_t MZ = 0;
	const double FAC = (1.0/MBIG);
	//int64_t mj,mk;
	//int i,ii,k;

	if (*idum < 0 || iff == 0) {
		iff=1;
		int64_t mj=MSEED-(*idum < 0 ? -*idum : *idum);
		mj %= MBIG;
		ma[55]=mj;
		int64_t mk=1;
		for (int i=1;i<=54;i++) {
			int ii=(21*i) % 55;
			ma[ii]=mk;
			mk=mj-mk;
			if (mk < MZ) mk += MBIG;
			mj=ma[ii];
		}
		for (int k=1;k<=4;k++)
			for (int i=1;i<=55;i++) {
				ma[i] -= ma[1+(i+30) % 55];
				if (ma[i] < MZ) ma[i] += MBIG;
			}
		inext=0;
		inextp=31;
		*idum=1;
	}
	if (++inext == 56) inext=1;
	if (++inextp == 56) inextp=1;
	int64_t mj=ma[inext]-ma[inextp];
	if (mj < MZ) mj += MBIG;
	ma[inext]=mj;
	return mj*FAC;
}
//#undef MBIG
//#undef MSEED
//#undef MZ
//#undef FAC
/* (C) Copr. 1986-92 Numerical Recipes Software 1+5-5i. */


// uniform random number (0-1)
double urand(int type, int64_t *seed)
{
	double ret = 0;

	assert((type >= 1) && (type <= 4));
	if      (type == 1) {
		ret = rand() / (RAND_MAX + 1.0);
	}
	else if (type == 2) {
		ret = ran0_nr(seed);
	}
	else if (type == 3) {
		ret = urand_1664501(seed);
	}
	else if (type == 4) {
		ret = ran3_nr(seed);
	}
	assert((ret >= 0) && (ret < 1));

	return ret;
}
