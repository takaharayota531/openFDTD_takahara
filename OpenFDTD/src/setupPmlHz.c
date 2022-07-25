/*
setupPmlHz.c

setup PML for Hz
*/

#include "ofd.h"

void setupPmlHz(int mode)
{
	int lx = cPML.l;
	int ly = cPML.l;
	int lz = cPML.l;

	int64_t num = 0;
	for (int i = iMin - lx + 0; i < iMax + lx; i++) {
	for (int j = jMin - ly + 0; j < jMax + ly; j++) {
	for (int k = kMin - lz + 1; k < kMax + lz; k++) {
		if ((i < 0) || (Nx <= i) ||
		    (j < 0) || (Ny <= j) ||
		    (k < 0) || (Nz <  k)) {
			if      (mode == 1) {
				fPmlHz[num].i = i;
				fPmlHz[num].j = j;
				fPmlHz[num].k = k;
				id_t m = 0;
				if      (k  <  0) m = IHZ(i,      j,      0     );
				else if (Nz <  k) m = IHZ(i,      j,      Nz    );
				else if (i  <  0) m = IHZ(0,      j,      k     );
				else if (Nx <= i) m = IHZ(Nx - 1, j,      k     );
				else if (j  <  0) m = IHZ(i,      0,      k     );
				else if (Ny <= j) m = IHZ(i,      Ny - 1, k     );
				fPmlHz[num].m = m;
			}
			num++;
		}
	}
	}
	}

	// array size
	if (mode == 0) {
		numPmlHz = num;
	}
}
