/*
murHy.c (OpenMP)

Mur for Hy
*/

#include "ofd.h"

void murHy(void)
{
	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
#ifdef __NEC__
#pragma _NEC ivdep
#endif
	for (n = 0; n < numMurHy; n++) {
		const int i  = fMurHy[n].i;
		const int j  = fMurHy[n].j;
		const int k  = fMurHy[n].k;
		const int i1 = fMurHy[n].i1;
		const int j1 = fMurHy[n].j1;
		const int k1 = fMurHy[n].k1;
		HY(i, j, k) = fMurHy[n].f
		            + fMurHy[n].g * (HY(i1, j1, k1) - HY(i, j, k));
		fMurHy[n].f = HY(i1, j1, k1);
	}
}
