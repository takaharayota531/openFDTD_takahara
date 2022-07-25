/*
updateEz.c

update Ez
*/

#include "ofd.h"
#include "finc.h"

static void updateEz_f(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n - Ni;
		int64_t n2 = n - Nj;
#if !defined(_VECTOR)
		for (int k = kMin; k <  kMax; k++) {
			Ez[n] = C1[iEz[n]] * Ez[n]
			      + C2[iEz[n]] * (RXn[i] * (Hy[n] - Hy[n1])
			                    - RYn[j] * (Hx[n] - Hx[n2]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__NEC__) || defined(__CLANG_FUJITSU)
		for (int k = kMin; k <  kMax; k++) {
			Ez[n] = K1Ez[n] * Ez[n]
			      + K2Ez[n] * (RXn[i] * (Hy[n] - Hy[n1])
			                 - RYn[j] * (Hx[n] - Hx[n2]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_ez   = &Ez[n];
		real_t * restrict ptr_hx   = &Hx[n];
		real_t * restrict ptr_hy   = &Hy[n];
		real_t * restrict ptr_hxm  = &Hx[n2];
		real_t * restrict ptr_hym  = &Hy[n1];
		real_t * restrict ptr_k1ez = &K1Ez[n];
		real_t * restrict ptr_k2ez = &K2Ez[n];
		real_t                 rxn =  RXn[i];   
		real_t                 ryn =  RYn[j];   
		for (int k = kMin; k <  kMax; k++) {
			*ptr_ez = *ptr_k1ez * (*ptr_ez)
			        + *ptr_k2ez * (      rxn * (*ptr_hy - *ptr_hym)
			                      -      ryn * (*ptr_hx - *ptr_hxm));
			ptr_ez++;
			ptr_hx++;
			ptr_hy++;
			ptr_hxm++;
			ptr_hym++;
			ptr_k1ez++;
			ptr_k2ez++;
			//__rxn++;
			//__ryn++;
		}
#endif
	}
	}
}

static void updateEz_p(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n - Ni;
		int64_t n2 = n - Nj;
#if !defined(_VECTOR)
		for (int k = kMin; k <  kMax; k++) {
			const id_t m = iEz[n];
			if (m == 0) {
				Ez[n] += RXn[i] * (Hy[n] - Hy[n1])
				       - RYn[j] * (Hx[n] - Hx[n2]);
			}
			else {
				real_t fi, dfi;
				finc(Xn[i], Yn[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.ei[2], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Ez[n] = -fi;
				}
				else {
					Ez[n] = C1[m] * Ez[n]
					      + C2[m] * (RXn[i] * (Hy[n] - Hy[n1])
					               - RYn[j] * (Hx[n] - Hx[n2]))
					      - C3[m] * dfi
					      - C4[m] * fi;
				}
			}
			n++;
			n1++;
			n2++;
		}
#elif defined(__NEC__) || defined(__CLANG_FUJITSU)
#if defined(__CLANG_FUJITSU)
#pragma clang loop vectorize(assume_safety)
#endif
		for (int k = kMin; k <  kMax; k++) {
			// vector : no if-statement
			real_t fi, dfi;
			finc(Xn[i], Yn[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.ei[2], Planewave.ai, Dt, &fi, &dfi);
			Ez[n] = K1Ez[n] * Ez[n]
			      + K2Ez[n] * (RXn[i] * (Hy[n] - Hy[n1])
			                 - RYn[j] * (Hx[n] - Hx[n2]))
			      - K3Ez[n] * dfi
			      - K4Ez[n] * fi;
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_ez   = &Ez[n];
		real_t * restrict ptr_hx   = &Hx[n];
		real_t * restrict ptr_hy   = &Hy[n];
		real_t * restrict ptr_hxm  = &Hx[n2];
		real_t * restrict ptr_hym  = &Hy[n1];
		real_t * restrict ptr_k1ez = &K1Ez[n];
		real_t * restrict ptr_k2ez = &K2Ez[n];
		real_t * restrict ptr_k3ez = &K3Ez[n];
		real_t * restrict ptr_k4ez = &K4Ez[n];
		real_t                 rxn =  RXn[i];   
		real_t                 ryn =  RYn[j];   
		double                  xn =  Xn[i];
		double                  yn =  Yn[j];
		double * restrict ptr_zc   = &Zc[kMin];
		int64_t kmin = kMin;
		int64_t kmax = kMax;
#pragma loop novrec
		for (int64_t k = kmin; k <  kmax; k++) {
			// vector : no if-statement, 64bit index
			real_t fi, dfi;
			finc(xn, yn, *ptr_zc, t, Planewave.r0, Planewave.ri, Planewave.ei[2], Planewave.ai, Dt, &fi, &dfi);
			*ptr_ez = *ptr_k1ez * (*ptr_ez)
			        + *ptr_k2ez * (      rxn * (*ptr_hy - *ptr_hym)
			                      -      ryn * (*ptr_hx - *ptr_hxm))
			        - *ptr_k3ez * dfi
			        - *ptr_k4ez * fi;
			ptr_ez++;
			ptr_hx++;
			ptr_hy++;
			ptr_hxm++;
			ptr_hym++;
			ptr_k1ez++;
			ptr_k2ez++;
			ptr_k3ez++;
			ptr_k4ez++;
			//__rxn++;
			//__ryn++;
			ptr_zc++;
		}
#endif
	}
	}
}

void updateEz(double t)
{
	if (NFeed) {
		updateEz_f();
	}
	else if (IPlanewave) {
		updateEz_p(t);
	}
}
