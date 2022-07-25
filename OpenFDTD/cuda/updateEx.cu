/*
updateEx.cu

update Ex
*/

#include "ofd.h"
#include "ofd_cuda.h"
#include "finc_cuda.h"


__host__ __device__
static void updateEx_f(
	int i, int j, int k,
	real_t ex[], const real_t hy[], const real_t hz[], const id_t iex[],
	const real_t c1[], const real_t c2[],
	real_t ryn, real_t rzn, param_t *p)
{
	const int64_t n = LA(p, i, j, k);
	const id_t m = iex[n];
	ex[n] = c1[m] * ex[n]
	      + c2[m] * (ryn * (hz[n] - hz[n - p->Nj])
	               - rzn * (hy[n] - hy[n - p->Nk]));
}


__host__ __device__
static void updateEx_p(
	int i, int j, int k,
	real_t ex[], const real_t hy[], const real_t hz[], const id_t iex[],
	const real_t c1[], const real_t c2[], const real_t c3[], const real_t c4[],
	real_t ryn, real_t rzn, param_t *p,
	real_t x, real_t y, real_t z, real_t t)
{
	const int64_t n = LA(p, i, j, k);
	const id_t m = iex[n];

	if (m == 0) {
		ex[n] += ryn * (hz[n] - hz[n - p->Nj])
		       - rzn * (hy[n] - hy[n - p->Nk]);
	}
	else {
		real_t fi, dfi;
		finc_cuda(x, y, z, t, p->r0, p->ri, p->ei[0], p->ai, p->dt, &fi, &dfi);
		if (m == PEC) {
			ex[n] = -fi;
		}
		else {
			ex[n] = c1[m] * ex[n]
			      + c2[m] * (ryn * (hz[n] - hz[n - p->Nj])
			               - rzn * (hy[n] - hy[n - p->Nk]))
			      - c3[m] * dfi
			      - c4[m] * fi;
		}
	}
}


__global__
static void updateEx_gpu(
	real_t ex[], const real_t hy[], const real_t hz[], const id_t iex[],
	const real_t c1[], const real_t c2[], const real_t c3[], const real_t c4[],
	const real_t ryn[], const real_t rzn[], const real_t xc[], const real_t yn[], const real_t zn[], real_t t)
{
	int i = d_Param.iMin + threadIdx.z + (blockIdx.z * blockDim.z);
	int j = d_Param.jMin + threadIdx.y + (blockIdx.y * blockDim.y);
	int k = d_Param.kMin + threadIdx.x + (blockIdx.x * blockDim.x);
	if ((i <  d_Param.iMax) &&
	    (j <= d_Param.jMax) &&
	    (k <= d_Param.kMax)) {
		if (d_Param.NFeed) {
			updateEx_f(
				i, j, k,
				ex, hy, hz, iex,
				c1, c2,
				ryn[j], rzn[k], &d_Param);
		}
		else if (d_Param.IPlanewave) {
			updateEx_p(
				i, j, k,
				ex, hy, hz, iex,
				c1, c2, c3, c4,
				ryn[j], rzn[k], &d_Param,
				xc[i], yn[j], zn[k], t);
		}
	}
}


static void updateEx_cpu(
	real_t ex[], const real_t hy[], const real_t hz[], const id_t iex[],
	const real_t c1[], const real_t c2[], const real_t c3[], const real_t c4[],
	const real_t ryn[], const real_t rzn[], const real_t xc[], const real_t yn[], const real_t zn[], real_t t)
{
	for (int i = h_Param.iMin; i <  h_Param.iMax; i++) {
	for (int j = h_Param.jMin; j <= h_Param.jMax; j++) {
	for (int k = h_Param.kMin; k <= h_Param.kMax; k++) {
		if (h_Param.NFeed) {
			updateEx_f(
				i, j, k,
				ex, hy, hz, iex,
				c1, c2,
				ryn[j], rzn[k], &h_Param);
		}
		else if (h_Param.IPlanewave) {
			updateEx_p(
				i, j, k,
				ex, hy, hz, iex,
				c1, c2, c3, c4,
				ryn[j], rzn[k], &h_Param,
				xc[i], yn[j], zn[k], t);
		}
	}
	}
	}
}


void updateEx(double t)
{
	if (GPU) {
		cudaMemcpyToSymbol(d_Param, &h_Param, sizeof(param_t));
		dim3 grid(
			CEIL(kMax - kMin + 1, updateBlock.x),
			CEIL(jMax - jMin + 1, updateBlock.y),
			CEIL(iMax - iMin + 0, updateBlock.z));
		updateEx_gpu<<<grid, updateBlock>>>(
			Ex, Hy, Hz, d_iEx,
			d_C1, d_C2, d_C3, d_C4,
			d_RYn, d_RZn, d_Xc, d_Yn, d_Zn, (real_t)t);
		if (UM) cudaDeviceSynchronize();
	}
	else {
		updateEx_cpu(
			Ex, Hy, Hz, iEx,
			C1, C2, C3, C4,
			RYn, RZn, h_Xc, h_Yn, h_Zn, (real_t)t);
	}
}
