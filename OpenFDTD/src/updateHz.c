/*
updateHz.c

update Hz
*/

#include "ofd.h"
#include "finc.h"

static void updateHz_f(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n + Ni;
		int64_t n2 = n + Nj;
#if !defined(_VECTOR)
		for (int k = kMin; k <= kMax; k++) {
			Hz[n] = D1[iHz[n]] * Hz[n]
			      - D2[iHz[n]] * (RXc[i] * (Ey[n1] - Ey[n])
			                    - RYc[j] * (Ex[n2] - Ex[n]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__NEC__) || defined(__CLANG_FUJITSU)
		for (int k = kMin; k <= kMax; k++) {
			Hz[n] = K1Hz[n] * Hz[n]
			      - K2Hz[n] * (RXc[i] * (Ey[n1] - Ey[n])
			                 - RYc[j] * (Ex[n2] - Ex[n]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_hz   = &Hz[n];
		real_t * restrict ptr_ex   = &Ex[n];
		real_t * restrict ptr_ey   = &Ey[n];
		real_t * restrict ptr_exp  = &Ex[n2];
		real_t * restrict ptr_eyp  = &Ey[n1];
		real_t * restrict ptr_k1hz = &K1Hz[n];
		real_t * restrict ptr_k2hz = &K2Hz[n];
		real_t                 rxc =  RXc[i];   
		real_t                 ryc =  RYc[j];   
		for (int k = kMin; k <= kMax; k++) {
			*ptr_hz = *ptr_k1hz * (*ptr_hz)
			        - *ptr_k2hz * (      rxc * (*ptr_eyp - *ptr_ey)
			                      -      ryc * (*ptr_exp - *ptr_ex));
			ptr_hz++;
			ptr_ex++;
			ptr_ey++;
			ptr_exp++;
			ptr_eyp++;
			ptr_k1hz++;
			ptr_k2hz++;
			//__rxc++;
			//__ryc++;
		}
#endif
	}
	}
}

static void updateHz_p(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n + Ni;
		int64_t n2 = n + Nj;
#if !defined(_VECTOR)
		for (int k = kMin; k <= kMax; k++) {
			const id_t m = iHz[n];
			if (m == 0) {
				Hz[n] -= RXc[i] * (Ey[n1] - Ey[n])
				       - RYc[j] * (Ex[n2] - Ex[n]);
			}
			else {
				real_t fi, dfi;
				finc(Xc[i], Yc[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.hi[2], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Hz[n] = -fi;
				}
				else {
					Hz[n] = D1[m] * Hz[n]
					      - D2[m] * (RXc[i] * (Ey[n1] - Ey[n])
					               - RYc[j] * (Ex[n2] - Ex[n]))
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
		for (int k = kMin; k <= kMax; k++) {
			// vector : no if-statement
			real_t fi, dfi;
			finc(Xc[i], Yc[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.hi[2], Planewave.ai, Dt, &fi, &dfi);
			Hz[n] = K1Hz[n] * Hz[n]
			      - K2Hz[n] * (RXc[i] * (Ey[n1] - Ey[n])
			                 - RYc[j] * (Ex[n2] - Ex[n]))
			      - K3Hz[n] * dfi
			      - K4Hz[n] * fi;
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_hz   = &Hz[n];
		real_t * restrict ptr_ex   = &Ex[n];
		real_t * restrict ptr_ey   = &Ey[n];
		real_t * restrict ptr_exp  = &Ex[n2];
		real_t * restrict ptr_eyp  = &Ey[n1];
		real_t * restrict ptr_k1hz = &K1Hz[n];
		real_t * restrict ptr_k2hz = &K2Hz[n];
		real_t * restrict ptr_k3hz = &K3Hz[n];
		real_t * restrict ptr_k4hz = &K4Hz[n];
		real_t                 rxc =  RXc[i];   
		real_t                 ryc =  RYc[j];   
		double                  xc =  Xc[i];
		double                  yc =  Yc[j];
		double * restrict ptr_zn   = &Zn[kMin];
		int64_t kmin = kMin;
		int64_t kmax = kMax;
#pragma loop novrec
		for (int64_t k = kmin; k <= kmax; k++) {
			// vector : no if-statement, 64bit index
			real_t fi, dfi;
			finc(xc, yc, *ptr_zn, t, Planewave.r0, Planewave.ri, Planewave.hi[2], Planewave.ai, Dt, &fi, &dfi);
			*ptr_hz = *ptr_k1hz * (*ptr_hz)
			        - *ptr_k2hz * (      rxc * (*ptr_eyp - *ptr_ey)
			                      -      ryc * (*ptr_exp - *ptr_ex))
			        - *ptr_k3hz * dfi
			        - *ptr_k4hz * fi;
			ptr_hz++;
			ptr_ex++;
			ptr_ey++;
			ptr_exp++;
			ptr_eyp++;
			ptr_k1hz++;
			ptr_k2hz++;
			ptr_k3hz++;
			ptr_k4hz++;
			//__rxc++;
			//__ryc++;
			ptr_zn++;
		}
#endif
	}
	}
}

void updateHz(double t)
{
	if (NFeed) {
		updateHz_f();
	}
	else if (IPlanewave) {
		updateHz_p(t);
	}
}
