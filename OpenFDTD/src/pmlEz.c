/*
pmlEz.c (OpenMP)

PML for Ez
*/

#include "ofd.h"

void pmlEz(void)
{
	const int lx = cPML.l;
	const int ly = cPML.l;

	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (n = 0; n < numPmlEz; n++) {
		const int  i = fPmlEz[n].i;
		const int  j = fPmlEz[n].j;
		const int  k = fPmlEz[n].k;
		const id_t m = fPmlEz[n].m;

		const real_t dhy = HY(i, j, k) - HY(i - 1, j, k);
		const real_t rx = RXn[MIN(MAX(i, 0), Nx    )] * rPmlE[m];
		Ezx[n] = (Ezx[n] + (rx * dhy)) * gPmlXn[i + lx];

		const real_t dhx = HX(i, j, k) - HX(i, j - 1, k);
		const real_t ry = RYn[MIN(MAX(j, 0), Ny    )] * rPmlE[m];
		Ezy[n] = (Ezy[n] - (ry * dhx)) * gPmlYn[j + ly];

		EZ(i, j, k) = Ezx[n] + Ezy[n];
	}
}
