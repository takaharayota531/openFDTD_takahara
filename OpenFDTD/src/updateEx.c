/*
updateEx.c

update Ex
*/

#include "ofd.h"
#include "finc.h"

static void updateEx_f(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n - Nj;
		int64_t n2 = n - Nk;
#if !defined(_VECTOR)
		for (int k = kMin; k <= kMax; k++) {
			Ex[n] = C1[iEx[n]] * Ex[n]
			      + C2[iEx[n]] * (RYn[j] * (Hz[n] - Hz[n1])
			                    - RZn[k] * (Hy[n] - Hy[n2]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__NEC__) || defined(__CLANG_FUJITSU)
		for (int k = kMin; k <= kMax; k++) {
			Ex[n] = K1Ex[n] * Ex[n]
			      + K2Ex[n] * (RYn[j] * (Hz[n] - Hz[n1])
			                 - RZn[k] * (Hy[n] - Hy[n2]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_ex   = &Ex[n];
		real_t * restrict ptr_hy   = &Hy[n];
		real_t * restrict ptr_hz   = &Hz[n];
		real_t * restrict ptr_hym  = &Hy[n2];
		real_t * restrict ptr_hzm  = &Hz[n1];
		real_t * restrict ptr_k1ex = &K1Ex[n];
		real_t * restrict ptr_k2ex = &K2Ex[n];
		real_t                 ryn =  RYn[j];   
		real_t * restrict ptr_rzn  = &RZn[kMin];
		for (int k = kMin; k <= kMax; k++) {
			*ptr_ex = *ptr_k1ex * (*ptr_ex)
			        + *ptr_k2ex * (      ryn * (*ptr_hz - *ptr_hzm)
			                      - *ptr_rzn * (*ptr_hy - *ptr_hym));
			ptr_ex++;
			ptr_hy++;
			ptr_hz++;
			ptr_hym++;
			ptr_hzm++;
			ptr_k1ex++;
			ptr_k2ex++;
			//__ryn++;
			ptr_rzn++;
		}
#endif
	}
	}
}

static void updateEx_p(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n - Nj;
		int64_t n2 = n - Nk;
#if !defined(_VECTOR)
		for (int k = kMin; k <= kMax; k++) {
			const id_t m = iEx[n];
			if (m == 0) {
				Ex[n] += RYn[j] * (Hz[n] - Hz[n1])
				       - RZn[k] * (Hy[n] - Hy[n2]);
			}
			else {
				real_t fi, dfi;
				finc(Xc[i], Yn[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[0], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Ex[n] = -fi;
				}
				else {
					Ex[n] = C1[m] * Ex[n]
					      + C2[m] * (RYn[j] * (Hz[n] - Hz[n1])
					               - RZn[k] * (Hy[n] - Hy[n2]))
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
			finc(Xc[i], Yn[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[0], Planewave.ai, Dt, &fi, &dfi);
			Ex[n] = K1Ex[n] * Ex[n]
			      + K2Ex[n] * (RYn[j] * (Hz[n] - Hz[n1])
			                 - RZn[k] * (Hy[n] - Hy[n2]))
			      - K3Ex[n] * dfi
			      - K4Ex[n] * fi;
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_ex   = &Ex[n];
		real_t * restrict ptr_hy   = &Hy[n];
		real_t * restrict ptr_hz   = &Hz[n];
		real_t * restrict ptr_hym  = &Hy[n2];
		real_t * restrict ptr_hzm  = &Hz[n1];
		real_t * restrict ptr_k1ex = &K1Ex[n];
		real_t * restrict ptr_k2ex = &K2Ex[n];
		real_t * restrict ptr_k3ex = &K3Ex[n];
		real_t * restrict ptr_k4ex = &K4Ex[n];
		real_t                 ryn =  RYn[j];   
		real_t * restrict ptr_rzn  = &RZn[kMin];
		double                 xc  =  Xc[i];
		double                 yn  =  Yn[j];
		double * restrict ptr_zn   = &Zn[kMin];
		int64_t kmin = kMin;
		int64_t kmax = kMax;
#pragma loop novrec
		for (int64_t k = kmin; k <= kmax; k++) {
			// vector : no if-statement, 64bit index
			real_t fi, dfi;
			finc(xc, yn, *ptr_zn, t, Planewave.r0, Planewave.ri, Planewave.ei[0], Planewave.ai, Dt, &fi, &dfi);
			*ptr_ex = *ptr_k1ex * (*ptr_ex)
			        + *ptr_k2ex * (      ryn * (*ptr_hz - *ptr_hzm)
			                      - *ptr_rzn * (*ptr_hy - *ptr_hym))
			        - *ptr_k3ex * dfi
			        - *ptr_k4ex * fi;
			ptr_ex++;
			ptr_hy++;
			ptr_hz++;
			ptr_hym++;
			ptr_hzm++;
			ptr_k1ex++;
			ptr_k2ex++;
			ptr_k3ex++;
			ptr_k4ex++;
			//__ryn++;
			ptr_rzn++;
			ptr_zn++;
		}
#endif
	}
	}
}

void updateEx(double t)
{
	if (NFeed) {
		updateEx_f();
	}
	else if (IPlanewave) {
		updateEx_p(t);
	}
}
