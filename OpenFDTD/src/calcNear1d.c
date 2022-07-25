/*
calcNear1d.c

calculate near1d field (runMode = 2)
*/

#include "ofd.h"
#include "complex.h"
#include "ofd_prototype.h"

void calcNear1d(void)
{
	if (runMode == 2) {
		// setup node index
		for (int n = 0; n < NNear1d; n++) {
			if      (Near1d[n].dir == 'X') {
				Near1d[n].id1 = nearest(Near1d[n].pos1, 0, Ny, Yn);
				Near1d[n].id2 = nearest(Near1d[n].pos2, 0, Nz, Zn);
			}
			else if (Near1d[n].dir == 'Y') {
				Near1d[n].id1 = nearest(Near1d[n].pos1, 0, Nz, Zn);
				Near1d[n].id2 = nearest(Near1d[n].pos2, 0, Nx, Xn);
			}
			else if (Near1d[n].dir == 'Z') {
				Near1d[n].id1 = nearest(Near1d[n].pos1, 0, Nx, Xn);
				Near1d[n].id2 = nearest(Near1d[n].pos2, 0, Ny, Yn);
			}
		}
	}

	// alloc
	int *div = (int *)malloc(NNear1d * sizeof(int));
	for (int n = 0; n < NNear1d; n++) {
		if      (Near1d[n].dir == 'X') {
			div[n] = Nx;
		}
		else if (Near1d[n].dir == 'Y') {
			div[n] = Ny;
		}
		else if (Near1d[n].dir == 'Z') {
			div[n] = Nz;
		}
	}

	int num = 0;
	for (int n = 0; n < NNear1d; n++) {
		num += div[n] + 1;
	}
	const size_t size = num * NFreq2 * sizeof(d_complex_t);
	Near1dEx = (d_complex_t *)malloc(size);
	Near1dEy = (d_complex_t *)malloc(size);
	Near1dEz = (d_complex_t *)malloc(size);
	Near1dHx = (d_complex_t *)malloc(size);
	Near1dHy = (d_complex_t *)malloc(size);
	Near1dHz = (d_complex_t *)malloc(size);

	int64_t adr = 0;
	for (int n = 0; n < NNear1d; n++) {
		for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
			for (int l = 0; l <= div[n]; l++) {
				int i = 0, j = 0, k = 0;
				if      (Near1d[n].dir == 'X') {
					i = l;
					j = Near1d[n].id1;
					k = Near1d[n].id2;
				}
				else if (Near1d[n].dir == 'Y') {
					j = l;
					k = Near1d[n].id1;
					i = Near1d[n].id2;
				}
				else if (Near1d[n].dir == 'Z') {
					k = l;
					i = Near1d[n].id1;
					j = Near1d[n].id2;
				}
				NodeE_c(ifreq, i, j, k, &Near1dEx[adr], &Near1dEy[adr], &Near1dEz[adr]);
				NodeH_c(ifreq, i, j, k, &Near1dHx[adr], &Near1dHy[adr], &Near1dHz[adr]);
				adr++;
			}
		}
	}
	//assert(adr * sizeof(d_complex_t) == size);

	// free
	free(div);
}
