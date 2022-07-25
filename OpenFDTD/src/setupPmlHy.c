/*
setupPmlHy.c

setup PML for Hy
*/

#include "ofd.h"

void setupPmlHy(int mode)
{
	int lx = cPML.l;
	int ly = cPML.l;
	int lz = cPML.l;

	int64_t num = 0;
	for (int i = iMin - lx + 0; i < iMax + lx; i++) {
	for (int j = jMin - ly + 1; j < jMax + ly; j++) {
	for (int k = kMin - lz + 0; k < kMax + lz; k++) {
		if ((i < 0) || (Nx <= i) ||
		    (j < 0) || (Ny <  j) ||
		    (k < 0) || (Nz <= k)) {
			if      (mode == 1) {
				fPmlHy[num].i = i;
				fPmlHy[num].j = j;
				fPmlHy[num].k = k;
				id_t m = 0;
				if      (j  <  0) m = IHY(i,      0,      k     );
				else if (Ny <  j) m = IHY(i,      Ny,     k     );
				else if (k  <  0) m = IHY(i,      j,      0     );
				else if (Nz <= k) m = IHY(i,      j,      Nz - 1);
				else if (i  <  0) m = IHY(0,      j,      k     );
				else if (Nx <= i) m = IHY(Nx - 1, j,      k     );
				fPmlHy[num].m = m;
			}
			num++;
		}
	}
	}
	}

	// array size
	if (mode == 0) {
		numPmlHy = num;
	}
}
