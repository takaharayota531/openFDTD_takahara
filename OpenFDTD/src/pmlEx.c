/*
pmlEx.c (OpenMP)

PML for Ex
*/

#include "ofd.h"

void pmlEx(void)
{
	const int ly = cPML.l;
	const int lz = cPML.l;

	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (n = 0; n < numPmlEx; n++) {
		const int  i = fPmlEx[n].i;
		const int  j = fPmlEx[n].j;
		const int  k = fPmlEx[n].k;
		const id_t m = fPmlEx[n].m;

		const real_t dhz = HZ(i, j, k) - HZ(i, j - 1, k);
		const real_t ry = RYn[MIN(MAX(j, 0), Ny    )] * rPmlE[m];
		Exy[n] = (Exy[n] + (ry * dhz)) * gPmlYn[j + ly];

		const real_t dhy = HY(i, j, k) - HY(i, j, k - 1);
		const real_t rz = RZn[MIN(MAX(k, 0), Nz    )] * rPmlE[m];
		Exz[n] = (Exz[n] - (rz * dhy)) * gPmlZn[k + lz];

		EX(i, j, k) = Exy[n] + Exz[n];
	}
}
