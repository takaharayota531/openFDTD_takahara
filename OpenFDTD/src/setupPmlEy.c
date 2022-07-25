/*
setupPmlEy.c

setup PML for Ey
*/

#include "ofd.h"

void setupPmlEy(int mode)
{
	int lx = cPML.l;
	int ly = cPML.l;
	int lz = cPML.l;

	int64_t num = 0;
	for (int i = iMin - lx + 1; i < iMax + lx; i++) {
	for (int j = jMin - ly + 0; j < jMax + ly; j++) {
	for (int k = kMin - lz + 1; k < kMax + lz; k++) {
		if ((i < 0) || (Nx <  i) ||
		    (j < 0) || (Ny <= j) ||
		    (k < 0) || (Nz <  k)) {
			if      (mode == 1) {
				fPmlEy[num].i = i;
				fPmlEy[num].j = j;
				fPmlEy[num].k = k;
				id_t m = 0;
				if      (j  <  0) m = IEY(i,      0,      k     );
				else if (Ny <= j) m = IEY(i,      Ny - 1, k     );
				else if (k  <  0) m = IEY(i,      j,      0     );
				else if (Nz <  k) m = IEY(i,      j,      Nz    );
				else if (i  <  0) m = IEY(0,      j,      k     );
				else if (Nx <  i) m = IEY(Nx,     j,      k     );
				fPmlEy[num].m = m;
			}
			num++;
		}
	}
	}
	}

	// array size
	if (mode == 0) {
		numPmlEy = num;
	}
}
