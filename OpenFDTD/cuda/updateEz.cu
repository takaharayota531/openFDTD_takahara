/*
updateEz.cu

update Ez
*/

#include "ofd.h"
#include "ofd_cuda.h"
#include "finc_cuda.h"


__host__ __device__
static void updateEz_f(
	int i, int j, int k,
	real_t ez[], const real_t hx[], const real_t hy[], const id_t iez[],
	const real_t c1[], const real_t c2[],
	real_t rxn, real_t ryn, param_t *p)
{
	const int64_t n = LA(p, i, j, k);
	const id_t m = iez[n];
	ez[n] = c1[m] * ez[n]
	      + c2[m] * (rxn * (hy[n] - hy[n - p->Ni])
	               - ryn * (hx[n] - hx[n - p->Nj]));
}


__host__ __device__
static void updateEz_p(
	int i, int j, int k,
	real_t ez[], const real_t hx[], const real_t hy[], const id_t iez[],
	const real_t c1[], const real_t c2[], const real_t c3[], const real_t c4[],
	real_t rxn, real_t ryn, param_t *p,
	real_t x, real_t y, real_t z, real_t t)
{
	const int64_t n = LA(p, i, j, k);
	const id_t m = iez[n];

	if (m == 0) {
		ez[n] += rxn * (hy[n] - hy[n - p->Ni])
		       - ryn * (hx[n] - hx[n - p->Nj]);
	}
	else {
		real_t fi, dfi;
		finc_cuda(x, y, z, t, p->r0, p->ri, p->ei[2], p->ai, p->dt, &fi, &dfi);
		if (m == PEC) {
			ez[n] = -fi;
		}
		else {
			ez[n] = c1[m] * ez[n]
			      + c2[m] * (rxn * (hy[n] - hy[n - p->Ni])
			               - ryn * (hx[n] - hx[n - p->Nj]))
			      - c3[m] * dfi
			      - c4[m] * fi;
		}
	}
}


__global__
static void updateEz_gpu(
	real_t ez[], const real_t hx[], const real_t hy[], const id_t iez[],
	const real_t c1[], const real_t c2[], const real_t c3[], const real_t c4[],
	const real_t rxn[], const real_t ryn[], const real_t xn[], const real_t yn[], const real_t zc[], real_t t)
{
	int i = d_Param.iMin + threadIdx.z + (blockIdx.z * blockDim.z);
	int j = d_Param.jMin + threadIdx.y + (blockIdx.y * blockDim.y);
	int k = d_Param.kMin + threadIdx.x + (blockIdx.x * blockDim.x);
	if ((i <= d_Param.iMax) &&
	    (j <= d_Param.jMax) &&
	    (k <  d_Param.kMax)) {
		if (d_Param.NFeed) {
			updateEz_f(
				i, j, k,
				ez, hx, hy, iez,
				c1, c2,
				rxn[i], ryn[j], &d_Param);
		}
		else if (d_Param.IPlanewave) {
			updateEz_p(
				i, j, k,
				ez, hx, hy, iez,
				c1, c2, c3, c4,
				rxn[i], ryn[j], &d_Param,
				xn[i], yn[j], zc[k], t);
		}
	}
}


static void updateEz_cpu(
	real_t ez[], const real_t hx[], const real_t hy[], const id_t iez[],
	const real_t c1[], const real_t c2[], const real_t c3[], const real_t c4[],
	const real_t rxn[], const real_t ryn[], const real_t xn[], const real_t yn[], const real_t zc[], real_t t)
{
	for (int i = h_Param.iMin; i <= h_Param.iMax; i++) {
	for (int j = h_Param.jMin; j <= h_Param.jMax; j++) {
	for (int k = h_Param.kMin; k <  h_Param.kMax; k++) {
		if (h_Param.NFeed) {
			updateEz_f(
				i, j, k,
				ez, hx, hy, iez,
				c1, c2,
				rxn[i], ryn[j], &h_Param);
		}
		else if (h_Param.IPlanewave) {
			updateEz_p(
				i, j, k,
				ez, hx, hy, iez,
				c1, c2, c3, c4,
				rxn[i], ryn[j], &h_Param,
				xn[i], yn[j], zc[k], t);
		}
	}
	}
	}
}


void updateEz(double t)
{
	if (GPU) {
		cudaMemcpyToSymbol(d_Param, &h_Param, sizeof(param_t));
		dim3 grid(
			CEIL(kMax - kMin + 0, updateBlock.x),
			CEIL(jMax - jMin + 1, updateBlock.y),
			CEIL(iMax - iMin + 1, updateBlock.z));
		updateEz_gpu<<<grid, updateBlock>>>(
			Ez, Hx, Hy, d_iEz,
			d_C1, d_C2, d_C3, d_C4,
			d_RXn, d_RYn, d_Xn, d_Yn, d_Zc, (real_t)t);
		if (UM) cudaDeviceSynchronize();
	}
	else {
		updateEz_cpu(
			Ez, Hx, Hy, iEz,
			C1, C2, C3, C4,
			RXn, RYn, h_Xn, h_Yn, h_Zc, (real_t)t);
	}
}
