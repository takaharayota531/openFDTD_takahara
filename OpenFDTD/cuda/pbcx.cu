/*
pbcx.cu (CUDA)

PBC on +/- X boundary
*/

#include "ofd.h"
#include "ofd_cuda.h"


__host__ __device__
static void _pbcxhy(int j, int k, float *hy, param_t *p)
{
	hy[LA(p,    -1, j, k)] = hy[LA(p, p->Nx - 1, j, k)];
	hy[LA(p, p->Nx, j, k)] = hy[LA(p,         0, j, k)];
}


__host__ __device__
static void _pbcxhz(int j, int k, float *hz, param_t *p)
{
	hz[LA(p,    -1, j, k)] = hz[LA(p, p->Nx - 1, j, k)];
	hz[LA(p, p->Nx, j, k)] = hz[LA(p,         0, j, k)];
}


__global__
static void pbcxhy_gpu(float *hy)
{
	int j = d_Param.jMin - 0 + (blockDim.x * blockIdx.x) + threadIdx.x;
	int k = d_Param.kMin - 1 + (blockDim.y * blockIdx.y) + threadIdx.y;

	if (j <= d_Param.jMax) {
	if (k <= d_Param.kMax) {
		_pbcxhy(j, k, hy, &d_Param);
	}
	}
}


__global__
static void pbcxhz_gpu(float *hz)
{
	int j = d_Param.jMin - 1 + (blockDim.x * blockIdx.x) + threadIdx.x;
	int k = d_Param.kMin - 0 + (blockDim.y * blockIdx.y) + threadIdx.y;

	if (j <= d_Param.jMax) {
	if (k <= d_Param.kMax) {
		_pbcxhz(j, k, hz, &d_Param);
	}
	}
}


static void pbcxhy_cpu(float *hy)
{
	for (int j = h_Param.jMin - 0; j <= h_Param.jMax; j++) {
	for (int k = h_Param.kMin - 1; k <= h_Param.kMax; k++) {
		_pbcxhy(j, k, hy, &h_Param);
	}
	}
}


static void pbcxhz_cpu(float *hz)
{
	for (int j = h_Param.jMin - 1; j <= h_Param.jMax; j++) {
	for (int k = h_Param.kMin - 0; k <= h_Param.kMax; k++) {
		_pbcxhz(j, k, hz, &h_Param);
	}
	}
}


void pbcx()
{
	if (GPU) {
		cudaMemcpyToSymbol(d_Param, &h_Param, sizeof(param_t));
		dim3 block(pbcBlock, pbcBlock);
		dim3 grid_hy(CEIL(jMax - jMin + 1, block.x),
		             CEIL(kMax - kMin + 2, block.y));
		dim3 grid_hz(CEIL(jMax - jMin + 2, block.x),
		             CEIL(kMax - kMin + 1, block.y));
		pbcxhy_gpu<<<grid_hy, block>>>(Hy);
		pbcxhz_gpu<<<grid_hz, block>>>(Hz);
		if (UM) cudaDeviceSynchronize();
	}
	else {
		pbcxhy_cpu(Hy);
		pbcxhz_cpu(Hz);
	}
}
