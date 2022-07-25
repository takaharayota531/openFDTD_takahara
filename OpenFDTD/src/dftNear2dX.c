/*
dftNear2dX.c

DFT of near field, X plane
*/

#include "ofd.h"
#include "nodeE_r.h"
#include "nodeH_r.h"
#include "dftField.h"

void dftNear2dX(
	int i, int64_t adr1, int64_t adr2,
	d_complex_t *cex, d_complex_t *cey, d_complex_t *cez,
	d_complex_t *chx, d_complex_t *chy, d_complex_t *chz)
{
	int    j;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    j = jMin; j <= jMax; j++) {
	for (int k = kMin; k <= kMax; k++) {
		if (!iProc[i]) continue;  // MPI

		real_t ex, ey, ez, hx, hy, hz;
		nodeE_r(i, j, k, &ex, &ey, &ez);
		nodeH_r(i, j, k, &hx, &hy, &hz);

		const int64_t adr = adr1 + ((j - jMin) * (Nz + 1)) + k;
		dftField(
			&cex[adr], &cey[adr], &cez[adr], &chx[adr], &chy[adr], &chz[adr],
			ex, ey, ez, hx, hy, hz,
			cEdft[adr2], cHdft[adr2]);
	}
	}
}
