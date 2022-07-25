/*
setupPmlHx.c

setup PML for Hx
*/

#include "ofd.h"

void setupPmlHx(int mode)
{
	int lx = cPML.l;
	int ly = cPML.l;
	int lz = cPML.l;

	int64_t num = 0;
	for (int i = iMin - lx + 1; i < iMax + lx; i++) {
	for (int j = jMin - ly + 0; j < jMax + ly; j++) {
	for (int k = kMin - lz + 0; k < kMax + lz; k++) {
		if ((i < 0) || (Nx <  i) ||
		    (j < 0) || (Ny <= j) ||
		    (k < 0) || (Nz <= k)) {
			if      (mode == 1) {
				fPmlHx[num].i = i;
				fPmlHx[num].j = j;
				fPmlHx[num].k = k;
				id_t m = 0;
				if      (i  <  0) m = IHX(0,      j,      k     );
				else if (Nx <  i) m = IHX(Nx,     j,      k     );
				else if (j  <  0) m = IHX(i,      0,      k     );
				else if (Ny <= j) m = IHX(i,      Ny - 1, k     );
				else if (k  <  0) m = IHX(i,      j,      0     );
				else if (Nz <= k) m = IHX(i,      j,      Nz - 1);
				fPmlHx[num].m = m;
			}
			num++;
		}
	}
	}
	}

	// array size
	if (mode == 0) {
		numPmlHx = num;
	}
}
