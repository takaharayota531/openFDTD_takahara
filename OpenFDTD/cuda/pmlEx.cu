/*
pmlEx.cu

PML ABC for Ex
*/

#include "ofd.h"
#include "ofd_cuda.h"

__host__ __device__
static void pmlex(
	int64_t n, int64_t nj, int64_t nk,
	real_t *ex, real_t *hy, real_t *hz, real_t *exy, real_t *exz,
	real_t ry, real_t rz, real_t gpmlyn, real_t gpmlzn)
{
	*exy = (*exy + (ry * (hz[n] - hz[n - nj]))) * gpmlyn;
	*exz = (*exz - (rz * (hy[n] - hy[n - nk]))) * gpmlzn;
	ex[n] = *exy + *exz;
}

__global__
static void pmlex_gpu(
	int ny, int nz,
	int64_t ni, int64_t nj, int64_t nk, int64_t n0,
	real_t *ex, real_t *hy, real_t *hz, real_t *exy, real_t *exz,
	int l, int64_t numpmlex,
	pml_t *fpmlex, real_t *rpmle, real_t *ryn, real_t *rzn, real_t *gpmlyn, real_t *gpmlzn)
{
	int64_t n = threadIdx.x + (blockIdx.x * blockDim.x);
	if (n < numpmlex) {
		const int  i = fpmlex[n].i;
		const int  j = fpmlex[n].j;
		const int  k = fpmlex[n].k;
		const id_t m = fpmlex[n].m;
		const real_t ry = (m != PEC) ? ryn[MIN(MAX(j, 0), ny    )] * rpmle[m] : 0;
		const real_t rz = (m != PEC) ? rzn[MIN(MAX(k, 0), nz    )] * rpmle[m] : 0;
		const int64_t nc = (ni * i) + (nj * j) + (nk * k) + n0;
		pmlex(
			nc, nj, nk,
			ex, hy, hz, &exy[n], &exz[n],
			ry, rz, gpmlyn[j + l], gpmlzn[k + l]);
	}
}

static void pmlex_cpu(
	int ny, int nz,
	int64_t ni, int64_t nj, int64_t nk, int64_t n0,
	real_t *ex, real_t *hy, real_t *hz, real_t *exy, real_t *exz,
	int l, int64_t numpmlex,
	pml_t *fpmlex, real_t *rpmle, real_t *ryn, real_t *rzn, real_t *gpmlyn, real_t *gpmlzn)
{
	for (int64_t n = 0; n < numpmlex; n++) {
		const int  i = fpmlex[n].i;
		const int  j = fpmlex[n].j;
		const int  k = fpmlex[n].k;
		const id_t m = fpmlex[n].m;
		const real_t ry = (m != PEC) ? ryn[MIN(MAX(j, 0), ny    )] * rpmle[m] : 0;
		const real_t rz = (m != PEC) ? rzn[MIN(MAX(k, 0), nz    )] * rpmle[m] : 0;
		const int64_t nc = (ni * i) + (nj * j) + (nk * k) + n0;
		pmlex(
			nc, nj, nk,
			ex, hy, hz, &exy[n], &exz[n],
			ry, rz, gpmlyn[j + l], gpmlzn[k + l]);
	}
}

void pmlEx()
{
	if (GPU) {
		pmlex_gpu<<<(int)CEIL(numPmlEx, pmlBlock), pmlBlock>>>(
			Ny, Nz,
			Ni, Nj, Nk, N0,
			Ex, Hy, Hz, Exy, Exz,
			cPML.l, numPmlEx,
			d_fPmlEx, d_rPmlE, d_RYn, d_RZn, d_gPmlYn, d_gPmlZn);
		if (UM) cudaDeviceSynchronize();
	}
	else {
		pmlex_cpu(
			Ny, Nz,
			Ni, Nj, Nk, N0,
			Ex, Hy, Hz, Exy, Exz,
			cPML.l, numPmlEx,
			fPmlEx, rPmlE, RYn, RZn, gPmlYn, gPmlZn);
	}
}
