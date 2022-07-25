/*
dftNear3d.c (OpenMP)

DFT of near field
*/

#include "ofd.h"

void dftNear3d(int itime)
{
	for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
		int i;
		const int id = (itime * NFreq2) + ifreq;

		const real_t ef_r = (real_t)cEdft[id].r;
		const real_t ef_i = (real_t)cEdft[id].i;

		// Ex
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin; i <  iMax; i++) {
		for (int j = jMin; j <= jMax; j++) {
			const int64_t m = NA(i, j, kMin);
			const int64_t n = (ifreq * NN) + m;
			real_t *e   = &Ex[m];
			real_t *e_r = &Ex_r[n];
			real_t *e_i = &Ex_i[n];
			for (int k = kMin; k <= kMax; k++) {
				*e_r += (*e) * ef_r;
				*e_i += (*e) * ef_i;
				e++;
				e_r++;
				e_i++;
			}
		}
		}

		// Ey
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin; i <= iMax; i++) {
		for (int j = jMin; j <  jMax; j++) {
			const int64_t m = NA(i, j, kMin);
			const int64_t n = (ifreq * NN) + m;
			real_t *e   = &Ey[m];
			real_t *e_r = &Ey_r[n];
			real_t *e_i = &Ey_i[n];
			for (int k = kMin; k <= kMax; k++) {
				*e_r += (*e) * ef_r;
				*e_i += (*e) * ef_i;
				e++;
				e_r++;
				e_i++;
			}
		}
		}

		// Ez
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin; i <= iMax; i++) {
		for (int j = jMin; j <= jMax; j++) {
			const int64_t m = NA(i, j, kMin);
			const int64_t n = (ifreq * NN) + m;
			real_t *e   = &Ez[m];
			real_t *e_r = &Ez_r[n];
			real_t *e_i = &Ez_i[n];
			for (int k = kMin; k <  kMax; k++) {
				*e_r += (*e) * ef_r;
				*e_i += (*e) * ef_i;
				e++;
				e_r++;
				e_i++;
			}
		}
		}

		const real_t hf_r = (real_t)cHdft[id].r;
		const real_t hf_i = (real_t)cHdft[id].i;

		// Hx
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin - 1; i <= iMax + 1; i++) {
		for (int j = jMin - 1; j <  jMax + 1; j++) {
			const int64_t m = NA(i, j, kMin - 1);
			const int64_t n = (ifreq * NN) + m;
			real_t *h   = &Hx[m];
			real_t *h_r = &Hx_r[n];
			real_t *h_i = &Hx_i[n];
			for (int k = kMin - 1; k <  kMax + 1; k++) {
				*h_r += (*h) * hf_r;
				*h_i += (*h) * hf_i;
				h++;
				h_r++;
				h_i++;
			}
		}
		}

		// Hy
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin - 1; i <  iMax + 1; i++) {
		for (int j = jMin - 1; j <= jMax + 1; j++) {
			const int64_t m = NA(i, j, kMin - 1);
			const int64_t n = (ifreq * NN) + m;
			real_t *h   = &Hy[m];
			real_t *h_r = &Hy_r[n];
			real_t *h_i = &Hy_i[n];
			for (int k = kMin - 1; k <  kMax + 1; k++) {
				*h_r += (*h) * hf_r;
				*h_i += (*h) * hf_i;
				h++;
				h_r++;
				h_i++;
			}
		}
		}

		// Hz
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin - 1; i <  iMax + 1; i++) {
		for (int j = jMin - 1; j <  jMax + 1; j++) {
			const int64_t m = NA(i, j, kMin - 1);
			const int64_t n = (ifreq * NN) + m;
			real_t *h   = &Hz[m];
			real_t *h_r = &Hz_r[n];
			real_t *h_i = &Hz_i[n];
			for (int k = kMin - 1; k <= kMax + 1; k++) {
				*h_r += (*h) * hf_r;
				*h_i += (*h) * hf_i;
				h++;
				h_r++;
				h_i++;
			}
		}
		}

	}
}
