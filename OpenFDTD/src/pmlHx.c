/*
pmlHx.c (OpenMP)

PML for Hx
*/

#include "ofd.h"

void pmlHx(void)
{
	const int ly = cPML.l;
	const int lz = cPML.l;

	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (n = 0; n < numPmlHx; n++) {
		const int  i = fPmlHx[n].i;
		const int  j = fPmlHx[n].j;
		const int  k = fPmlHx[n].k;
		const id_t m = fPmlHx[n].m;

		const real_t dez = EZ(i, j + 1, k) - EZ(i, j, k);
		const real_t ry = RYc[MIN(MAX(j, 0), Ny - 1)] * rPmlH[m];
		Hxy[n] = (Hxy[n] - (ry * dez)) * gPmlYc[j + ly];

		const real_t dey = EY(i, j, k + 1) - EY(i, j, k);
		const real_t rz = RZc[MIN(MAX(k, 0), Nz - 1)] * rPmlH[m];
		Hxz[n] = (Hxz[n] + (rz * dey)) * gPmlZc[k + lz];

		HX(i, j, k) = Hxy[n] + Hxz[n];
	}
}
