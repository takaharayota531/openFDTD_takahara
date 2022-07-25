/*
calcNear3d.c
*/

#include "ofd.h"
#include "complex.h"

void calcNear3d(int component, int mode)
{
	assert((component >= 0) && (component <= 5));
	assert((mode >= 0) && (mode <= 2));
	assert(NFreq2 > 0);

	// Ex
	if      (component == 0) {
		if      (mode == 0) {
			cEx_r = (real_t *)malloc(NEx * NFreq2 * sizeof(real_t));
			cEx_i = (real_t *)malloc(NEx * NFreq2 * sizeof(real_t));
		}
		else if (mode == 1) {
			for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
				for (int i = 0; i <  Nx; i++) {
				for (int j = 0; j <= Ny; j++) {
				for (int k = 0; k <= Nz; k++) {
					const int64_t m = (ifreq * NEx) + NEX(i, j, k);
					const int64_t n = (ifreq * NN) + NA(i, j, k);
					cEx_r[m] = Ex_r[n];
					cEx_i[m] = Ex_i[n];
				}
				}
				}
			}
		}
		else if (mode == 2) {
			free(Ex_r);
			free(Ex_i);
		}
	}

	// Ey
	else if (component == 1) {
		if      (mode == 0) {
			cEy_r = (real_t *)malloc(NEy * NFreq2 * sizeof(real_t));
			cEy_i = (real_t *)malloc(NEy * NFreq2 * sizeof(real_t));
		}
		else if (mode == 1) {
			for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
				for (int i = 0; i <= Nx; i++) {
				for (int j = 0; j <  Ny; j++) {
				for (int k = 0; k <= Nz; k++) {
					const int64_t m = (ifreq * NEy) + NEY(i, j, k);
					const int64_t n = (ifreq * NN) + NA(i, j, k);
					cEy_r[m] = Ey_r[n];
					cEy_i[m] = Ey_i[n];
				}
				}
				}
			}
		}
		else if (mode == 2) {
			free(Ey_r);
			free(Ey_i);
		}
	}

	// Ez
	else if (component == 2) {
		if      (mode == 0) {
			cEz_r = (real_t *)malloc(NEz * NFreq2 * sizeof(real_t));
			cEz_i = (real_t *)malloc(NEz * NFreq2 * sizeof(real_t));
		}
		else if (mode == 1) {
			for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
				for (int i = 0; i <= Nx; i++) {
				for (int j = 0; j <= Ny; j++) {
				for (int k = 0; k <  Nz; k++) {
					const int64_t m = (ifreq * NEz) + NEZ(i, j, k);
					const int64_t n = (ifreq * NN) + NA(i, j, k);
					cEz_r[m] = Ez_r[n];
					cEz_i[m] = Ez_i[n];
				}
				}
				}
			}
		}
		else if (mode == 2) {
			free(Ez_r);
			free(Ez_i);
		}
	}

	// Hx
	else if (component == 3) {
		if      (mode == 0) {
			cHx_r = (real_t *)malloc(NHx * NFreq2 * sizeof(real_t));
			cHx_i = (real_t *)malloc(NHx * NFreq2 * sizeof(real_t));
		}
		else if (mode == 1) {
			for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
				for (int i =  0; i <= Nx; i++) {
				for (int j = -1; j <= Ny; j++) {
				for (int k = -1; k <= Nz; k++) {
					const int64_t m = (ifreq * NHx) + NHX(i, j, k);
					const int64_t n = (ifreq * NN) + NA(i, j, k);
					cHx_r[m] = Hx_r[n];
					cHx_i[m] = Hx_i[n];
				}
				}
				}
		}
		}
		else if (mode == 2) {
			free(Hx_r);
			free(Hx_i);
		}
	}

	// Hy
	else if (component == 4) {
		if      (mode == 0) {
			cHy_r = (real_t *)malloc(NHy * NFreq2 * sizeof(real_t));
			cHy_i = (real_t *)malloc(NHy * NFreq2 * sizeof(real_t));
		}
		else if (mode == 1) {
			for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
				for (int i = -1; i <= Nx; i++) {
				for (int j =  0; j <= Ny; j++) {
				for (int k = -1; k <= Nz; k++) {
					const int64_t m = (ifreq * NHy) + NHY(i, j, k);
					const int64_t n = (ifreq * NN) + NA(i, j, k);
					cHy_r[m] = Hy_r[n];
					cHy_i[m] = Hy_i[n];
				}
				}
				}
			}
		}
		else if (mode == 2) {
			free(Hy_r);
			free(Hy_i);
		}
	}

	// Hz
	else if (component == 5) {
		if      (mode == 0) {
			cHz_r = (real_t *)malloc(NHz * NFreq2 * sizeof(real_t));
			cHz_i = (real_t *)malloc(NHz * NFreq2 * sizeof(real_t));
		}
		else if (mode == 1) {
			for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
				for (int i = -1; i <= Nx; i++) {
				for (int j = -1; j <= Ny; j++) {
				for (int k =  0; k <= Nz; k++) {
					const int64_t m = (ifreq * NHz) + NHZ(i, j, k);
					const int64_t n = (ifreq * NN) + NA(i, j, k);
					cHz_r[m] = Hz_r[n];
					cHz_i[m] = Hz_i[n];
				}
				}
				}
			}
		}
		else if (mode == 2) {
			free(Hz_r);
			free(Hz_i);
		}
	}
}
