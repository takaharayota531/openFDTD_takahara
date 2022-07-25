/*
pmlHz.c (OpenMP)

PML for Hz
*/

#include "ofd.h"

void pmlHz(void)
{
	const int lx = cPML.l;
	const int ly = cPML.l;

	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (n = 0; n < numPmlHz; n++) {
		const int  i = fPmlHz[n].i;
		const int  j = fPmlHz[n].j;
		const int  k = fPmlHz[n].k;
		const id_t m = fPmlHz[n].m;

		const real_t dey = EY(i + 1, j, k) - EY(i, j, k);
		const real_t rx = RXc[MIN(MAX(i, 0), Nx - 1)] * rPmlH[m];
		Hzx[n] = (Hzx[n] - (rx * dey)) * gPmlXc[i + lx];

		const real_t dex = EX(i, j + 1, k) - EX(i, j, k);
		const real_t ry = RYc[MIN(MAX(j, 0), Ny - 1)] * rPmlH[m];
		Hzy[n] = (Hzy[n] + (ry * dex)) * gPmlYc[j + ly];

		HZ(i, j, k) = Hzx[n] + Hzy[n];
	}
}
