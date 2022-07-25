/*
pmlHz.cu

PML ABC for Hz
*/

#include "ofd.h"
#include "ofd_cuda.h"

__host__ __device__
static void pmlhz(
	int64_t n, int64_t ni, int64_t nj,
	real_t *hz, real_t *ex, real_t *ey, real_t *hzx, real_t *hzy,
	real_t rx, real_t ry, real_t gpmlxc, real_t gpmlyc)
{
	*hzx = (*hzx - (rx * (ey[n + ni] - ey[n]))) * gpmlxc;
	*hzy = (*hzy + (ry * (ex[n + nj] - ex[n]))) * gpmlyc;
	hz[n] = *hzx + *hzy;
}

__global__
static void pmlhz_gpu(
	int nx, int ny,
	int64_t ni, int64_t nj, int64_t nk, int64_t n0,
	real_t *hz, real_t *ex, real_t *ey, real_t *hzx, real_t *hzy,
	int l, int64_t numpmlhz,
	pml_t *fpmlhz, real_t *rpmlh, real_t *rxc, real_t *ryc, real_t *gpmlxc, real_t *gpmlyc)
{
	int64_t n = threadIdx.x + (blockIdx.x * blockDim.x);
	if (n < numpmlhz) {
		const int  i = fpmlhz[n].i;
		const int  j = fpmlhz[n].j;
		const int  k = fpmlhz[n].k;
		const id_t m = fpmlhz[n].m;
		const real_t rx = (m != PEC) ? rxc[MIN(MAX(i, 0), nx - 1)] * rpmlh[m] : 0;
		const real_t ry = (m != PEC) ? ryc[MIN(MAX(j, 0), ny - 1)] * rpmlh[m] : 0;
		const int64_t nc = (ni * i) + (nj * j) + (nk * k) + n0;
		pmlhz(
			nc, ni, nj,
			hz, ex, ey, &hzx[n], &hzy[n],
			rx, ry, gpmlxc[i + l], gpmlyc[j + l]);
	}
}

static void pmlhz_cpu(
	int nx, int ny,
	int64_t ni, int64_t nj, int64_t nk, int64_t n0,
	real_t *hz, real_t *ex, real_t *ey, real_t *hzx, real_t *hzy,
	int l, int64_t numpmlhz,
	pml_t *fpmlhz, real_t *rpmlh, real_t *rxc, real_t *ryc, real_t *gpmlxc, real_t *gpmlyc)
{
	for (int64_t n = 0; n < numpmlhz; n++) {
		const int  i = fpmlhz[n].i;
		const int  j = fpmlhz[n].j;
		const int  k = fpmlhz[n].k;
		const id_t m = fpmlhz[n].m;
		const real_t rx = (m != PEC) ? rxc[MIN(MAX(i, 0), nx - 1)] * rpmlh[m] : 0;
		const real_t ry = (m != PEC) ? ryc[MIN(MAX(j, 0), ny - 1)] * rpmlh[m] : 0;
		const int64_t nc = (ni * i) + (nj * j) + (nk * k) + n0;
		pmlhz(
			nc, ni, nj,
			hz, ex, ey, &hzx[n], &hzy[n],
			rx, ry, gpmlxc[i + l], gpmlyc[j + l]);
	}
}

void pmlHz()
{
	if (GPU) {
		pmlhz_gpu<<<(int)CEIL(numPmlHz, pmlBlock), pmlBlock>>>(
			Nx, Ny,
			Ni, Nj, Nk, N0,
			Hz, Ex, Ey, Hzx, Hzy,
			cPML.l, numPmlHz,
			d_fPmlHz, d_rPmlH, d_RXc, d_RYc, d_gPmlXc, d_gPmlYc);
		if (UM) cudaDeviceSynchronize();
	}
	else {
		pmlhz_cpu(
			Nx, Ny,
			Ni, Nj, Nk, N0,
			Hz, Ex, Ey, Hzx, Hzy,
			cPML.l, numPmlHz,
			fPmlHz, rPmlH, RXc, RYc, gPmlXc, gPmlYc);
	}
}
