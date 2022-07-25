/*
updateHy.c

update Hy
*/

#include "ofd.h"
#include "finc.h"

static void updateHy_f(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n + Nk;
		int64_t n2 = n + Ni;
#if !defined(_VECTOR)
		for (int k = kMin; k <  kMax; k++) {
			Hy[n] = D1[iHy[n]] * Hy[n]
			      - D2[iHy[n]] * (RZc[k] * (Ex[n1] - Ex[n])
			                    - RXc[i] * (Ez[n2] - Ez[n]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__NEC__) || defined(__CLANG_FUJITSU)
		for (int k = kMin; k <  kMax; k++) {
			Hy[n] = K1Hy[n] * Hy[n]
			      - K2Hy[n] * (RZc[k] * (Ex[n1] - Ex[n])
			                 - RXc[i] * (Ez[n2] - Ez[n]));
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_hy   = &Hy[n];
		real_t * restrict ptr_ez   = &Ez[n];
		real_t * restrict ptr_ex   = &Ex[n];
		real_t * restrict ptr_ezp  = &Ez[n2];
		real_t * restrict ptr_exp  = &Ex[n1];
		real_t * restrict ptr_k1hy = &K1Hy[n];
		real_t * restrict ptr_k2hy = &K2Hy[n];
		real_t                 rxc =  RXc[i];   
		real_t * restrict ptr_rzc  = &RZc[kMin];
		for (int k = kMin; k <  kMax; k++) {
			*ptr_hy = *ptr_k1hy * (*ptr_hy)
			        - *ptr_k2hy * ( *ptr_rzc * (*ptr_exp - *ptr_ex)
			                      -      rxc * (*ptr_ezp - *ptr_ez));
			ptr_hy++;
			ptr_ez++;
			ptr_ex++;
			ptr_ezp++;
			ptr_exp++;
			ptr_k1hy++;
			ptr_k2hy++;
			//__rxc++;
			ptr_rzc++;
		}
#endif
	}
	}
}

static void updateHy_p(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		int64_t n1 = n + Nk;
		int64_t n2 = n + Ni;
#if !defined(_VECTOR)
		for (int k = kMin; k <  kMax; k++) {
			const id_t m = iHy[n];
			if (m == 0) {
				Hy[n] -= RZc[k] * (Ex[n1] - Ex[n])
				       - RXc[i] * (Ez[n2] - Ez[n]);
			}
			else {
				real_t fi, dfi;
				finc(Xc[i], Yn[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.hi[1], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Hy[n] = -fi;
				}
				else {
					Hy[n] = D1[m] * Hy[n]
					      - D2[m] * (RZc[k] * (Ex[n1] - Ex[n])
					               - RXc[i] * (Ez[n2] - Ez[n]))
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
			finc(Xc[i], Yn[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.hi[1], Planewave.ai, Dt, &fi, &dfi);
			Hy[n] = K1Hy[n] * Hy[n]
			      - K2Hy[n] * (RZc[k] * (Ex[n1] - Ex[n])
			                 - RXc[i] * (Ez[n2] - Ez[n]))
			      - K3Hy[n] * dfi
			      - K4Hy[n] * fi;
			n++;
			n1++;
			n2++;
		}
#elif defined(__FUJITSU)
		real_t * restrict ptr_hy   = &Hy[n];
		real_t * restrict ptr_ez   = &Ez[n];
		real_t * restrict ptr_ex   = &Ex[n];
		real_t * restrict ptr_ezp  = &Ez[n2];
		real_t * restrict ptr_exp  = &Ex[n1];
		real_t * restrict ptr_k1hy = &K1Hy[n];
		real_t * restrict ptr_k2hy = &K2Hy[n];
		real_t * restrict ptr_k3hy = &K3Hy[n];
		real_t * restrict ptr_k4hy = &K4Hy[n];
		real_t                 rxc =  RXc[i];   
		real_t * restrict ptr_rzc  = &RZc[kMin];
		double                  xc =  Xc[i];
		double                  yn =  Yn[j];
		double * restrict ptr_zc   = &Zc[kMin];
		int64_t kmin = kMin;
		int64_t kmax = kMax;
#pragma loop novrec
		for (int64_t k = kmin; k <  kmax; k++) {
			// vector : no if-statement, 64bit index
			real_t fi, dfi;
			finc(xc, yn, *ptr_zc, t, Planewave.r0, Planewave.ri, Planewave.hi[1], Planewave.ai, Dt, &fi, &dfi);
			*ptr_hy = *ptr_k1hy * (*ptr_hy)
			        - *ptr_k2hy * ( *ptr_rzc * (*ptr_exp - *ptr_ex)
			                      -      rxc * (*ptr_ezp - *ptr_ez))
			        - *ptr_k3hy * dfi
			        - *ptr_k4hy * fi;
			ptr_hy++;
			ptr_ez++;
			ptr_ex++;
			ptr_ezp++;
			ptr_exp++;
			ptr_k1hy++;
			ptr_k2hy++;
			ptr_k3hy++;
			ptr_k4hy++;
			//__rxc++;
			ptr_rzc++;
			ptr_zc++;
		}
#endif
	}
	}
}

void updateHy(double t)
{
	if (NFeed) {
		updateHy_f();
	}
	else if (IPlanewave) {
		updateHy_p(t);
	}
}
