/*
pmlHy.cu

PML ABC for Hy
*/

#include "ofd.h"
#include "ofd_cuda.h"

__host__ __device__
static void pmlhy(
	int64_t n, int64_t nk, int64_t ni,
	real_t *hy, real_t *ez, real_t *ex, real_t *hyz, real_t *hyx,
	real_t rz, real_t rx, real_t gpmlzc, real_t gpmlxc)
{
	*hyz = (*hyz - (rz * (ex[n + nk] - ex[n]))) * gpmlzc;
	*hyx = (*hyx + (rx * (ez[n + ni] - ez[n]))) * gpmlxc;
	hy[n] = *hyz + *hyx;
}

__global__
static void pmlhy_gpu(
	int nz, int nx,
	int64_t ni, int64_t nj, int64_t nk, int64_t n0,
	real_t *hy, real_t *ez, real_t *ex, real_t *hyz, real_t *hyx,
	int l, int64_t numpmlhy,
	pml_t *fpmlhy, real_t *rpmlh, real_t *rzc, real_t *rxc, real_t *gpmlzc, real_t *gpmlxc)
{
	int64_t n = threadIdx.x + (blockIdx.x * blockDim.x);
	if (n < numpmlhy) {
		const int  i = fpmlhy[n].i;
		const int  j = fpmlhy[n].j;
		const int  k = fpmlhy[n].k;
		const id_t m = fpmlhy[n].m;
		const real_t rz = (m != PEC) ? rzc[MIN(MAX(k, 0), nz - 1)] * rpmlh[m] : 0;
		const real_t rx = (m != PEC) ? rxc[MIN(MAX(i, 0), nx - 1)] * rpmlh[m] : 0;
		const int64_t nc = (ni * i) + (nj * j) + (nk * k) + n0;
		pmlhy(
			nc, nk, ni,
			hy, ez, ex, &hyz[n], &hyx[n],
			rz, rx, gpmlzc[k + l], gpmlxc[i + l]);
	}
}

static void pmlhy_cpu(
	int nz, int nx,
	int64_t ni, int64_t nj, int64_t nk, int64_t n0,
	real_t *hy, real_t *ez, real_t *ex, real_t *hyz, real_t *hyx,
	int l, int64_t numpmlhy,
	pml_t *fpmlhy, real_t *rpmlh, real_t *rzc, real_t *rxc, real_t *gpmlzc, real_t *gpmlxc)
{
	for (int64_t n = 0; n < numpmlhy; n++) {
		const int  i = fpmlhy[n].i;
		const int  j = fpmlhy[n].j;
		const int  k = fpmlhy[n].k;
		const id_t m = fpmlhy[n].m;
		const real_t rz = (m != PEC) ? rzc[MIN(MAX(k, 0), nz - 1)] * rpmlh[m] : 0;
		const real_t rx = (m != PEC) ? rxc[MIN(MAX(i, 0), nx - 1)] * rpmlh[m] : 0;
		const int64_t nc = (ni * i) + (nj * j) + (nk * k) + n0;
		pmlhy(
			nc, nk, ni,
			hy, ez, ex, &hyz[n], &hyx[n],
			rz, rx, gpmlzc[k + l], gpmlxc[i + l]);
	}
}

void pmlHy()
{
	if (GPU) {
		pmlhy_gpu<<<(int)CEIL(numPmlHy, pmlBlock), pmlBlock>>>(
			Nz, Nx,
			Ni, Nj, Nk, N0,
			Hy, Ez, Ex, Hyz, Hyx,
			cPML.l, numPmlHy,
			d_fPmlHy, d_rPmlH, d_RZc, d_RXc, d_gPmlZc, d_gPmlXc);
		if (UM) cudaDeviceSynchronize();
	}
	else {
		pmlhy_cpu(
			Nz, Nx,
			Ni, Nj, Nk, N0,
			Hy, Ez, Ex, Hyz, Hyx,
			cPML.l, numPmlHy,
			fPmlHy, rPmlH, RZc, RXc, gPmlZc, gPmlXc);
	}
}
