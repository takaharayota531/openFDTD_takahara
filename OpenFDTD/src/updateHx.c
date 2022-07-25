/*
updateHx.c

update Hx
*/

#include "ofd.h"
#include "finc.h"

static void updateHx_f(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n + Nj;
		int64_t n2 = n + Nk;
#if !defined(_VECTOR)
		for (int k = kMin; k <  kMax; k++) {
			Hx[n] = D1[iHx[n]] * Hx[n]
			      - D2[iHx[n]] * (RYc[j] * (Ez[n1] - Ez[n])
			                    - RZc[k] * (Ey[n2] - Ey[n]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__NEC__) || defined(__CLANG_FUJITSU)
		for (int k = kMin; k <  kMax; k++) {
			Hx[n] = K1Hx[n] * Hx[n]
			      - K2Hx[n] * (RYc[j] * (Ez[n1] - Ez[n])
			                 - RZc[k] * (Ey[n2] - Ey[n]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_hx   = &Hx[n];
		real_t * restrict ptr_ey   = &Ey[n];
		real_t * restrict ptr_ez   = &Ez[n];
		real_t * restrict ptr_eyp  = &Ey[n2];
		real_t * restrict ptr_ezp  = &Ez[n1];
		real_t * restrict ptr_k1hx = &K1Hx[n];
		real_t * restrict ptr_k2hx = &K2Hx[n];
		real_t                 ryc =  RYc[j];   
		real_t * restrict ptr_rzc  = &RZc[kMin];
		for (int k = kMin; k <  kMax; k++) {
			*ptr_hx = *ptr_k1hx * (*ptr_hx)
			        - *ptr_k2hx * (      ryc * (*ptr_ezp - *ptr_ez)
			                      - *ptr_rzc * (*ptr_eyp - *ptr_ey));
			ptr_hx++;
			ptr_ey++;
			ptr_ez++;
			ptr_eyp++;
			ptr_ezp++;
			ptr_k1hx++;
			ptr_k2hx++;
			//__rxc++;
			ptr_rzc++;
		}
#endif
	}
	}
}

static void updateHx_p(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n + Nj;
		int64_t n2 = n + Nk;
#if !defined(_VECTOR)
		for (int k = kMin; k <  kMax; k++) {
			const id_t m = iHx[n];
			if (m == 0) {
				Hx[n] -= RYc[j] * (Ez[n1] - Ez[n])
				       - RZc[k] * (Ey[n2] - Ey[n]);
			}
			else {
				real_t fi, dfi;
				finc(Xn[i], Yc[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.hi[0], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Hx[n] = -fi;
				}
				else {
					Hx[n] = D1[m] * Hx[n]
					      - D2[m] * (RYc[j] * (Ez[n1] - Ez[n])
					               - RZc[k] * (Ey[n2] - Ey[n]))
					      - D3[m] * dfi
					      - D4[m] * fi;
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
			finc(Xn[i], Yc[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.hi[0], Planewave.ai, Dt, &fi, &dfi);
			Hx[n] = K1Hx[n] * Hx[n]
			      - K2Hx[n] * (RYc[j] * (Ez[n1] - Ez[n])
			                 - RZc[k] * (Ey[n2] - Ey[n]))
			      - K3Hx[n] * dfi
			      - K4Hx[n] * fi;
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_hx   = &Hx[n];
		real_t * restrict ptr_ey   = &Ey[n];
		real_t * restrict ptr_ez   = &Ez[n];
		real_t * restrict ptr_eyp  = &Ey[n2];
		real_t * restrict ptr_ezp  = &Ez[n1];
		real_t * restrict ptr_k1hx = &K1Hx[n];
		real_t * restrict ptr_k2hx = &K2Hx[n];
		real_t * restrict ptr_k3hx = &K3Hx[n];
		real_t * restrict ptr_k4hx = &K4Hx[n];
		real_t                 ryc =  RYc[j];   
		real_t * restrict ptr_rzc  = &RZc[kMin];
		double                  xn =  Xn[i];
		double                  yc =  Yc[j];
		double * restrict ptr_zc   = &Zc[kMin];
		int64_t kmin = kMin;
		int64_t kmax = kMax;
#pragma loop novrec
		for (int64_t k = kmin; k <  kmax; k++) {
			// vector : no if-statement, 64bit index
			real_t fi, dfi;
			finc(xn, yc, *ptr_zc, t, Planewave.r0, Planewave.ri, Planewave.hi[0], Planewave.ai, Dt, &fi, &dfi);
			*ptr_hx = *ptr_k1hx * (*ptr_hx)
			        - *ptr_k2hx * (      ryc * (*ptr_ezp - *ptr_ez)
			                      - *ptr_rzc * (*ptr_eyp - *ptr_ey))
			        - *ptr_k3hx * dfi
			        - *ptr_k4hx * fi;
			ptr_hx++;
			ptr_ey++;
			ptr_ez++;
			ptr_eyp++;
			ptr_ezp++;
			ptr_k1hx++;
			ptr_k2hx++;
			ptr_k3hx++;
			ptr_k4hx++;
			//__rxc++;
			ptr_rzc++;
			ptr_zc++;
		}
#endif
	}
	}
}

void updateHx(double t)
{
	if (NFeed) {
		updateHx_f();
	}
	else if (IPlanewave) {
		updateHx_p(t);
	}
}
