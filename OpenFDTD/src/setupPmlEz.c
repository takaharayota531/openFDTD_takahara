/*
setupPmlEz.c

setup PML for Ez
*/

#include "ofd.h"

void setupPmlEz(int mode)
{
	int lx = cPML.l;
	int ly = cPML.l;
	int lz = cPML.l;

	int64_t num = 0;
	for (int i = iMin - lx + 1; i < iMax + lx; i++) {
	for (int j = jMin - ly + 1; j < jMax + ly; j++) {
	for (int k = kMin - lz + 0; k < kMax + lz; k++) {
		if ((i < 0) || (Nx <  i) ||
		    (j < 0) || (Ny <  j) ||
		    (k < 0) || (Nz <= k)) {
			if      (mode == 1) {
				fPmlEz[num].i = i;
				fPmlEz[num].j = j;
				fPmlEz[num].k = k;
				id_t m = 0;
				if      (k  <  0) m = IEZ(i,      j,      0     );
				else if (Nz <= k) m = IEZ(i,      j,      Nz - 1);
				else if (i  <  0) m = IEZ(0,      j,      k     );
				else if (Nx <  i) m = IEZ(Nx,     j,      k     );
				else if (j  <  0) m = IEZ(i,      0,      k     );
				else if (Ny <  j) m = IEZ(i,      Ny,     k     );
				fPmlEz[num].m = m;
			}
			num++;
		}
	}
	}
	}

	// array size
	if (mode == 0) {
		numPmlEz = num;
	}
}
