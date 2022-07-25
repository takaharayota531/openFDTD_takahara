/*
dftNear2dZ.c

DFT of near field, Z plane
*/

#include "ofd.h"
#include "nodeE_r.h"
#include "nodeH_r.h"
#include "dftField.h"

void dftNear2dZ(
	int k, int64_t adr1, int64_t adr2,
	d_complex_t *cex, d_complex_t *cey, d_complex_t *cez,
	d_complex_t *chx, d_complex_t *chy, d_complex_t *chz)
{
	int    i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		if (!iProc[i]) continue;  // MPI

		real_t ex, ey, ez, hx, hy, hz;
		nodeE_r(i, j, k, &ex, &ey, &ez);
		nodeH_r(i, j, k, &hx, &hy, &hz);

		const int64_t adr = adr1 + ((i - iMin) * (Ny + 1)) + j;
		dftField(
			&cex[adr], &cey[adr], &cez[adr], &chx[adr], &chy[adr], &chz[adr],
			ex, ey, ez, hx, hy, hz,
			cEdft[adr2], cHdft[adr2]);
	}
	}
}
