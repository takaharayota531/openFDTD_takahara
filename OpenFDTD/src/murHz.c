/*
murHz.c (OpenMP)

Mur for Hz
*/

#include "ofd.h"

void murHz(void)
{
	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
#ifdef __NEC__
#pragma _NEC ivdep
#endif
	for (n = 0; n < numMurHz; n++) {
		const int i  = fMurHz[n].i;
		const int j  = fMurHz[n].j;
		const int k  = fMurHz[n].k;
		const int i1 = fMurHz[n].i1;
		const int j1 = fMurHz[n].j1;
		const int k1 = fMurHz[n].k1;
		HZ(i, j, k) = fMurHz[n].f
		            + fMurHz[n].g * (HZ(i1, j1, k1) - HZ(i, j, k));
		fMurHz[n].f = HZ(i1, j1, k1);
	}
}
