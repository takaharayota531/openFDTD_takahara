/*
murHx.c (OpenMP)

Mur for Hx
*/

#include "ofd.h"

void murHx(void)
{
	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
#ifdef __NEC__
#pragma _NEC ivdep
#endif
	for (n = 0; n < numMurHx; n++) {
		const int i  = fMurHx[n].i;
		const int j  = fMurHx[n].j;
		const int k  = fMurHx[n].k;
		const int i1 = fMurHx[n].i1;
		const int j1 = fMurHx[n].j1;
		const int k1 = fMurHx[n].k1;
		HX(i, j, k) = fMurHx[n].f
		            + fMurHx[n].g * (HX(i1, j1, k1) - HX(i, j, k));
		fMurHx[n].f = HX(i1, j1, k1);
	}
}
