/*
pmlEy.c (OpenMP)

PML for Ey
*/

#include "ofd.h"

void pmlEy(void)
{
	const int lz = cPML.l;
	const int lx = cPML.l;

	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (n = 0; n < numPmlEy; n++) {
		const int  i = fPmlEy[n].i;
		const int  j = fPmlEy[n].j;
		const int  k = fPmlEy[n].k;
		const id_t m = fPmlEy[n].m;

		const real_t dhx = HX(i, j, k) - HX(i, j, k - 1);
		const real_t rz = RZn[MIN(MAX(k, 0), Nz    )] * rPmlE[m];
		Eyz[n] = (Eyz[n] + (rz * dhx)) * gPmlZn[k + lz];

		const real_t dhz = HZ(i, j, k) - HZ(i - 1, j, k);
		const real_t rx = RXn[MIN(MAX(i, 0), Nx    )] * rPmlE[m];
		Eyx[n] = (Eyx[n] - (rx * dhz)) * gPmlXn[i + lx];

		EY(i, j, k) = Eyz[n] + Eyx[n];
	}
}
