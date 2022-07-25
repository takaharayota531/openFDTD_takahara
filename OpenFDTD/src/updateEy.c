/*
updateEy.c

update Ey
*/

#include "ofd.h"
#include "finc.h"

static void updateEy_f(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n - Nk;
		int64_t n2 = n - Ni;
#if !defined(_VECTOR)
		for (int k = kMin; k <= kMax; k++) {
			Ey[n] = C1[iEy[n]] * Ey[n]
			      + C2[iEy[n]] * (RZn[k] * (Hx[n] - Hx[n1])
			                    - RXn[i] * (Hz[n] - Hz[n2]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__NEC__) || defined(__CLANG_FUJITSU)
		for (int k = kMin; k <= kMax; k++) {
			Ey[n] = K1Ey[n] * Ey[n]
			      + K2Ey[n] * (RZn[k] * (Hx[n] - Hx[n1])
			                 - RXn[i] * (Hz[n] - Hz[n2]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_ey   = &Ey[n];
		real_t * restrict ptr_hz   = &Hz[n];
		real_t * restrict ptr_hx   = &Hx[n];
		real_t * restrict ptr_hzm  = &Hz[n2];
		real_t * restrict ptr_hxm  = &Hx[n1];
		real_t * restrict ptr_k1ey = &K1Ey[n];
		real_t * restrict ptr_k2ey = &K2Ey[n];
		real_t                 rxn =  RXn[i];   
		real_t * restrict ptr_rzn  = &RZn[kMin];
		for (int k = kMin; k <= kMax; k++) {
			*ptr_ey = *ptr_k1ey * (*ptr_ey)
			        + *ptr_k2ey * ( *ptr_rzn * (*ptr_hx - *ptr_hxm)
			                      -      rxn * (*ptr_hz - *ptr_hzm));
			ptr_ey++;
			ptr_hz++;
			ptr_hx++;
			ptr_hzm++;
			ptr_hxm++;
			ptr_k1ey++;
			ptr_k2ey++;
			//__rxn++;
			ptr_rzn++;
		}
#endif
	}
	}
}

static void updateEy_p(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n - Nk;
		int64_t n2 = n - Ni;
#if !defined(_VECTOR)
		for (int k = kMin; k <= kMax; k++) {
			const id_t m = iEy[n];
			if (m == 0) {
				Ey[n] += RZn[k] * (Hx[n] - Hx[n1])
				       - RXn[i] * (Hz[n] - Hz[n2]);
			}
			else {
				real_t fi, dfi;
				finc(Xn[i], Yc[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[1], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Ey[n] = -fi;
				}
				else {
					Ey[n] = C1[m] * Ey[n]
					      + C2[m] * (RZn[k] * (Hx[n] - Hx[n1])
					               - RXn[i] * (Hz[n] - Hz[n2]))
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
		for (int k = kMin; k <= kMax; k++) {
			// vector : no if-statement
			real_t fi, dfi;
			finc(Xn[i], Yc[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[1], Planewave.ai, Dt, &fi, &dfi);
			Ey[n] = K1Ey[n] * Ey[n]
			      + K2Ey[n] * (RZn[k] * (Hx[n] - Hx[n1])
			                 - RXn[i] * (Hz[n] - Hz[n2]))
			      - K3Ey[n] * dfi
			      - K4Ey[n] * fi;
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_ey   = &Ey[n];
		real_t * restrict ptr_hz   = &Hz[n];
		real_t * restrict ptr_hx   = &Hx[n];
		real_t * restrict ptr_hzm  = &Hz[n2];
		real_t * restrict ptr_hxm  = &Hx[n1];
		real_t * restrict ptr_k1ey = &K1Ey[n];
		real_t * restrict ptr_k2ey = &K2Ey[n];
		real_t * restrict ptr_k3ey = &K3Ey[n];
		real_t * restrict ptr_k4ey = &K4Ey[n];
		real_t                 rxn =  RXn[i];   
		real_t * restrict ptr_rzn  = &RZn[kMin];
		double                 xn  =  Xn[i];
		double                 yc  =  Yc[j];
		double * restrict ptr_zn   = &Zn[kMin];
		int64_t kmin = kMin;
		int64_t kmax = kMax;
#pragma loop novrec
		for (int64_t k = kmin; k <= kmax; k++) {
			// vector : no if-statement, 64bit index
			real_t fi, dfi;
			finc(xn, yc, *ptr_zn, t, Planewave.r0, Planewave.ri, Planewave.ei[1], Planewave.ai, Dt, &fi, &dfi);
			*ptr_ey = *ptr_k1ey * (*ptr_ey)
			        + *ptr_k2ey * ( *ptr_rzn * (*ptr_hx - *ptr_hxm)
			                      -      rxn * (*ptr_hz - *ptr_hzm))
			        - *ptr_k3ey * dfi
			        - *ptr_k4ey * fi;
			ptr_ey++;
			ptr_hz++;
			ptr_hx++;
			ptr_hzm++;
			ptr_hxm++;
			ptr_k1ey++;
			ptr_k2ey++;
			ptr_k3ey++;
			ptr_k4ey++;
			//__rxn++;
			ptr_rzn++;
			ptr_zn++;
		}
#endif
	}
	}
}

void updateEy(double t)
{
	if (NFeed) {
		updateEy_f();
	}
	else if (IPlanewave) {
		updateEy_p(t);
	}
}
