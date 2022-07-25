/*
dftNear3d.cu (CUDA)
*/

#include "ofd.h"
#include "ofd_cuda.h"

__host__ __device__ __forceinline__
static void dftadd(real_t *f_r, real_t *f_i, real_t f, real_t fctr_r, real_t fctr_i)
{
	*f_r += f * fctr_r;
	*f_i += f * fctr_i;
}

__global__
static void dft_near3dEx_gpu(
	real_t *ex, real_t *ex_r, real_t *ex_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, int64_t ni, int64_t nj, int64_t nk, int64_t n0)
{
	const int i = imin + threadIdx.z + (blockIdx.z * blockDim.z);
	const int j = jmin + threadIdx.y + (blockIdx.y * blockDim.y);
	const int k = kmin + threadIdx.x + (blockIdx.x * blockDim.x);
	if (i <  imax) {
	if (j <= jmax) {
	if (k <= kmax) {
		const int64_t m = (ni * i) + (nj * j) + (nk * k) + n0;
		const int64_t n = adr0 + m;
		dftadd(&ex_r[n], &ex_i[n], ex[m], f_r, f_i);
	}
	}
	}
}

__global__
static void dft_near3dEy_gpu(
	real_t *ey, real_t *ey_r, real_t *ey_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, int64_t ni, int64_t nj, int64_t nk, int64_t n0)
{
	const int i = imin + threadIdx.z + (blockIdx.z * blockDim.z);
	const int j = jmin + threadIdx.y + (blockIdx.y * blockDim.y);
	const int k = kmin + threadIdx.x + (blockIdx.x * blockDim.x);
	if (i <= imax) {
	if (j <  jmax) {
	if (k <= kmax) {
		const int64_t m = (ni * i) + (nj * j) + (nk * k) + n0;
		const int64_t n = adr0 + m;
		dftadd(&ey_r[n], &ey_i[n], ey[m], f_r, f_i);
	}
	}
	}
}

__global__
static void dft_near3dEz_gpu(
	real_t *ez, real_t *ez_r, real_t *ez_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, int64_t ni, int64_t nj, int64_t nk, int64_t n0)
{
	const int i = imin + threadIdx.z + (blockIdx.z * blockDim.z);
	const int j = jmin + threadIdx.y + (blockIdx.y * blockDim.y);
	const int k = kmin + threadIdx.x + (blockIdx.x * blockDim.x);
	if (i <= imax) {
	if (j <= jmax) {
	if (k <  kmax) {
		const int64_t m = (ni * i) + (nj * j) + (nk * k) + n0;
		const int64_t n = adr0 + m;
		dftadd(&ez_r[n], &ez_i[n], ez[m], f_r, f_i);
	}
	}
	}
}

__global__
static void dft_near3dHx_gpu(
	real_t *hx, real_t *hx_r, real_t *hx_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, int64_t ni, int64_t nj, int64_t nk, int64_t n0)
{
	const int i = imin - 0 + threadIdx.z + (blockIdx.z * blockDim.z);
	const int j = jmin - 1 + threadIdx.y + (blockIdx.y * blockDim.y);
	const int k = kmin - 1 + threadIdx.x + (blockIdx.x * blockDim.x);
	if (i <= imax) {
	if (j <= jmax) {
	if (k <= kmax) {
		const int64_t m = (ni * i) + (nj * j) + (nk * k) + n0;
		const int64_t n = adr0 + m;
		dftadd(&hx_r[n], &hx_i[n], hx[m], f_r, f_i);
	}
	}
	}
}

__global__
static void dft_near3dHy_gpu(
	real_t *hy, real_t *hy_r, real_t *hy_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, int64_t ni, int64_t nj, int64_t nk, int64_t n0)
{
	const int i = imin - 1 + threadIdx.z + (blockIdx.z * blockDim.z);
	const int j = jmin - 0 + threadIdx.y + (blockIdx.y * blockDim.y);
	const int k = kmin - 1 + threadIdx.x + (blockIdx.x * blockDim.x);
	if (i <= imax) {
	if (j <= jmax) {
	if (k <= kmax) {
		const int64_t m = (ni * i) + (nj * j) + (nk * k) + n0;
		const int64_t n = adr0 + m;
		dftadd(&hy_r[n], &hy_i[n], hy[m], f_r, f_i);
	}
	}
	}
}

__global__
static void dft_near3dHz_gpu(
	real_t *hz, real_t *hz_r, real_t *hz_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, int64_t ni, int64_t nj, int64_t nk, int64_t n0)
{
	const int i = imin - 1 + threadIdx.z + (blockIdx.z * blockDim.z);
	const int j = jmin - 1 + threadIdx.y + (blockIdx.y * blockDim.y);
	const int k = kmin - 0 + threadIdx.x + (blockIdx.x * blockDim.x);
	if (i <= imax) {
	if (j <= jmax) {
	if (k <= kmax) {
		const int64_t m = (ni * i) + (nj * j) + (nk * k) + n0;
		const int64_t n = adr0 + m;
		dftadd(&hz_r[n], &hz_i[n], hz[m], f_r, f_i);
	}
	}
	}
}

// cpu

static void dft_near3dEx_cpu(
	real_t *ex, real_t *ex_r, real_t *ex_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, param_t *p)
{
	for (int i = imin; i <  imax; i++) {
	for (int j = jmin; j <= jmax; j++) {
	for (int k = kmin; k <= kmax; k++) {
		const int64_t m = LA(p, i, j, k);
		assert((m >= 0) && (m < NN));
		const int64_t n = adr0 + m;
		dftadd(&ex_r[n], &ex_i[n], ex[m], f_r, f_i);
	}
	}
	}
}

static void dft_near3dEy_cpu(
	real_t *ey, real_t *ey_r, real_t *ey_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, param_t *p)
{
	for (int i = imin; i <= imax; i++) {
	for (int j = jmin; j <  jmax; j++) {
	for (int k = kmin; k <= kmax; k++) {
		const int64_t m = LA(p, i, j, k);
		assert((m >= 0) && (m < NN));
		const int64_t n = adr0 + m;
		dftadd(&ey_r[n], &ey_i[n], ey[m], f_r, f_i);
	}
	}
	}
}

static void dft_near3dEz_cpu(
	real_t *ez, real_t *ez_r, real_t *ez_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, param_t *p)
{
	for (int i = imin; i <= imax; i++) {
	for (int j = jmin; j <= jmax; j++) {
	for (int k = kmin; k <  kmax; k++) {
		const int64_t m = LA(p, i, j, k);
		assert((m >= 0) && (m < NN));
		const int64_t n = adr0 + m;
		dftadd(&ez_r[n], &ez_i[n], ez[m], f_r, f_i);
	}
	}
	}
}

static void dft_near3dHx_cpu(
	real_t *hx, real_t *hx_r, real_t *hx_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, param_t *p)
{
	for (int i = imin - 0; i <= imax; i++) {
	for (int j = jmin - 1; j <= jmax; j++) {
	for (int k = kmin - 1; k <= kmax; k++) {
		const int64_t m = LA(p, i, j, k);
		assert((m >= 0) && (m < NN));
		const int64_t n = adr0 + m;
		dftadd(&hx_r[n], &hx_i[n], hx[m], f_r, f_i);
	}
	}
	}
}

static void dft_near3dHy_cpu(
	real_t *hy, real_t *hy_r, real_t *hy_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, param_t *p)
{
	for (int i = imin - 1; i <= imax; i++) {
	for (int j = jmin - 0; j <= jmax; j++) {
	for (int k = kmin - 1; k <= kmax; k++) {
		const int64_t m = LA(p, i, j, k);
		assert((m >= 0) && (m < NN));
		const int64_t n = adr0 + m;
		dftadd(&hy_r[n], &hy_i[n], hy[m], f_r, f_i);
	}
	}
	}
}

static void dft_near3dHz_cpu(
	real_t *hz, real_t *hz_r, real_t *hz_i, real_t f_r, real_t f_i,
	int imin, int imax, int jmin, int jmax, int kmin, int kmax,
	int64_t adr0, param_t *p)
{
	for (int i = imin - 1; i <= imax; i++) {
	for (int j = jmin - 1; j <= jmax; j++) {
	for (int k = kmin - 0; k <= kmax; k++) {
		const int64_t m = LA(p, i, j, k);
		assert((m >= 0) && (m < NN));
		const int64_t n = adr0 + m;
		dftadd(&hz_r[n], &hz_i[n], hz[m], f_r, f_i);
	}
	}
	}
}

void dftNear3d(int itime)
{
	for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
		const int64_t adr0 = ifreq * NN;
		const int id = (itime * NFreq2) + ifreq;
		const real_t fe_r = (real_t)cEdft[id].r;
		const real_t fe_i = (real_t)cEdft[id].i;
		const real_t fh_r = (real_t)cHdft[id].r;
		const real_t fh_i = (real_t)cHdft[id].i;

		if (GPU) {
			//cudaMemcpyToSymbol(d_Param, &h_Param, sizeof(param_t));

			dim3 gridEx(
				CEIL(kMax - kMin + 1, updateBlock.x),
				CEIL(jMax - jMin + 1, updateBlock.y),
				CEIL(iMax - iMin + 0, updateBlock.z));
			dft_near3dEx_gpu<<<gridEx, updateBlock>>>(Ex, d_Ex_r, d_Ex_i, fe_r, fe_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, Ni, Nj, Nk, N0);

			dim3 gridEy(
				CEIL(kMax - kMin + 1, updateBlock.x),
				CEIL(jMax - jMin + 0, updateBlock.y),
				CEIL(iMax - iMin + 1, updateBlock.z));
			dft_near3dEy_gpu<<<gridEy, updateBlock>>>(Ey, d_Ey_r, d_Ey_i, fe_r, fe_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, Ni, Nj, Nk, N0);

			dim3 gridEz(
				CEIL(kMax - kMin + 0, updateBlock.x),
				CEIL(jMax - jMin + 1, updateBlock.y),
				CEIL(iMax - iMin + 1, updateBlock.z));
			dft_near3dEz_gpu<<<gridEz, updateBlock>>>(Ez, d_Ez_r, d_Ez_i, fe_r, fe_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, Ni, Nj, Nk, N0);

			dim3 gridHx(
				CEIL(kMax - kMin + 2, updateBlock.x),
				CEIL(jMax - jMin + 2, updateBlock.y),
				CEIL(iMax - iMin + 1, updateBlock.z));
			dft_near3dHx_gpu<<<gridHx, updateBlock>>>(Hx, d_Hx_r, d_Hx_i, fh_r, fh_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, Ni, Nj, Nk, N0);

			dim3 gridHy(
				CEIL(kMax - kMin + 2, updateBlock.x),
				CEIL(jMax - jMin + 1, updateBlock.y),
				CEIL(iMax - iMin + 2, updateBlock.z));
			dft_near3dHy_gpu<<<gridHy, updateBlock>>>(Hy, d_Hy_r, d_Hy_i, fh_r, fh_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, Ni, Nj, Nk, N0);

			dim3 gridHz(
				CEIL(kMax - kMin + 1, updateBlock.x),
				CEIL(jMax - jMin + 2, updateBlock.y),
				CEIL(iMax - iMin + 2, updateBlock.z));
			dft_near3dHz_gpu<<<gridHz, updateBlock>>>(Hz, d_Hz_r, d_Hz_i, fh_r, fh_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, Ni, Nj, Nk, N0);

			cudaDeviceSynchronize();
		}
		else {
			dft_near3dEx_cpu(Ex, Ex_r, Ex_i, fe_r, fe_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, &h_Param);
			dft_near3dEy_cpu(Ey, Ey_r, Ey_i, fe_r, fe_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, &h_Param);
			dft_near3dEz_cpu(Ez, Ez_r, Ez_i, fe_r, fe_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, &h_Param);
			dft_near3dHx_cpu(Hx, Hx_r, Hx_i, fh_r, fh_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, &h_Param);
			dft_near3dHy_cpu(Hy, Hy_r, Hy_i, fh_r, fh_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, &h_Param);
			dft_near3dHz_cpu(Hz, Hz_r, Hz_i, fh_r, fh_i, iMin, iMax, jMin, jMax, kMin, kMax, adr0, &h_Param);
		}
	}
}
