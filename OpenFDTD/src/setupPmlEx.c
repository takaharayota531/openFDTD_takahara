/*
setupPmlEx.c

setup PML for Ex
*/

#include "ofd.h"

void setupPmlEx(int mode)
{
	int lx = cPML.l;
	int ly = cPML.l;
	int lz = cPML.l;

	int64_t num = 0;
	for (int i = iMin - lx + 0; i < iMax + lx; i++) {
	for (int j = jMin - ly + 1; j < jMax + ly; j++) {
	for (int k = kMin - lz + 1; k < kMax + lz; k++) {
		if ((i < 0) || (Nx <= i) ||
		    (j < 0) || (Ny <  j) ||
		    (k < 0) || (Nz <  k)) {
			if      (mode == 1) {
				fPmlEx[num].i = i;
				fPmlEx[num].j = j;
				fPmlEx[num].k = k;
				id_t m = 0;
				if      (i  <  0) m = IEX(0,      j,      k     );
				else if (Nx <= i) m = IEX(Nx - 1, j,      k     );
				else if (j  <  0) m = IEX(i,      0,      k     );
				else if (Ny <  j) m = IEX(i,      Ny,     k     );
				else if (k  <  0) m = IEX(i,      j,      0     );
				else if (Nz <  k) m = IEX(i,      j,      Nz    );
				fPmlEx[num].m = m;
			}
			num++;
		}
	}
	}
	}

	// array size
	if (mode == 0) {
		numPmlEx = num;
	}
}
