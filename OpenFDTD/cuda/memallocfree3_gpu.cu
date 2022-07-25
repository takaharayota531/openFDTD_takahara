/*
memallocfree3_gpu.cu (CUDA)

alloc and free
(3) output data
*/

#include "ofd.h"
#include "ofd_cuda.h"
#include "ofd_prototype.h"

void memalloc3_gpu()
{
	size_t size;

	if      (runMode == 0) {
		// near field 1d
		if ((NNear1d > 0) && (NFreq2 > 0)) {
			size = Near1d_size;
			Near1dEx = (d_complex_t *)malloc(size);
			Near1dEy = (d_complex_t *)malloc(size);
			Near1dEz = (d_complex_t *)malloc(size);
			Near1dHx = (d_complex_t *)malloc(size);
			Near1dHy = (d_complex_t *)malloc(size);
			Near1dHz = (d_complex_t *)malloc(size);
			memset(Near1dEx, 0, size);
			memset(Near1dEy, 0, size);
			memset(Near1dEz, 0, size);
			memset(Near1dHx, 0, size);
			memset(Near1dHy, 0, size);
			memset(Near1dHz, 0, size);

			cuda_malloc(GPU, UM, (void **)&d_Near1dEx, size);
			cuda_malloc(GPU, UM, (void **)&d_Near1dEy, size);
			cuda_malloc(GPU, UM, (void **)&d_Near1dEz, size);
			cuda_malloc(GPU, UM, (void **)&d_Near1dHx, size);
			cuda_malloc(GPU, UM, (void **)&d_Near1dHy, size);
			cuda_malloc(GPU, UM, (void **)&d_Near1dHz, size);
		}

		// near field 2d
		if ((NNear2d > 0) && (NFreq2 > 0)) {
			size = Near2d_size;
			Near2dEx = (d_complex_t *)malloc(size);
			Near2dEy = (d_complex_t *)malloc(size);
			Near2dEz = (d_complex_t *)malloc(size);
			Near2dHx = (d_complex_t *)malloc(size);
			Near2dHy = (d_complex_t *)malloc(size);
			Near2dHz = (d_complex_t *)malloc(size);
			memset(Near2dEx, 0, size);
			memset(Near2dEy, 0, size);
			memset(Near2dEz, 0, size);
			memset(Near2dHx, 0, size);
			memset(Near2dHy, 0, size);
			memset(Near2dHz, 0, size);

			cuda_malloc(GPU, UM, (void **)&d_Near2dEx, size);
			cuda_malloc(GPU, UM, (void **)&d_Near2dEy, size);
			cuda_malloc(GPU, UM, (void **)&d_Near2dEz, size);
			cuda_malloc(GPU, UM, (void **)&d_Near2dHx, size);
			cuda_malloc(GPU, UM, (void **)&d_Near2dHy, size);
			cuda_malloc(GPU, UM, (void **)&d_Near2dHz, size);
		}
	}
	else if (runMode == 1) {
		// near field full (complex)
		if ((NN > 0) && (NFreq2 > 0)) {
			// freed in calcNear3d()
			size = NN * NFreq2 * sizeof(real_t);
			Ex_r = (real_t *)malloc(size);
			Ey_r = (real_t *)malloc(size);
			Ez_r = (real_t *)malloc(size);
			Hx_r = (real_t *)malloc(size);
			Hy_r = (real_t *)malloc(size);
			Hz_r = (real_t *)malloc(size);
			Ex_i = (real_t *)malloc(size);
			Ey_i = (real_t *)malloc(size);
			Ez_i = (real_t *)malloc(size);
			Hx_i = (real_t *)malloc(size);
			Hy_i = (real_t *)malloc(size);
			Hz_i = (real_t *)malloc(size);
			memset(Ex_r, 0, size);
			memset(Ey_r, 0, size);
			memset(Ez_r, 0, size);
			memset(Hx_r, 0, size);
			memset(Hy_r, 0, size);
			memset(Hz_r, 0, size);
			memset(Ex_i, 0, size);
			memset(Ey_i, 0, size);
			memset(Ez_i, 0, size);
			memset(Hx_i, 0, size);
			memset(Hy_i, 0, size);
			memset(Hz_i, 0, size);

			// freed in memfree2()
			cuda_malloc(GPU, UM, (void **)&d_Ex_r, size);
			cuda_malloc(GPU, UM, (void **)&d_Ey_r, size);
			cuda_malloc(GPU, UM, (void **)&d_Ez_r, size);
			cuda_malloc(GPU, UM, (void **)&d_Hx_r, size);
			cuda_malloc(GPU, UM, (void **)&d_Hy_r, size);
			cuda_malloc(GPU, UM, (void **)&d_Hz_r, size);
			cuda_malloc(GPU, UM, (void **)&d_Ex_i, size);
			cuda_malloc(GPU, UM, (void **)&d_Ey_i, size);
			cuda_malloc(GPU, UM, (void **)&d_Ez_i, size);
			cuda_malloc(GPU, UM, (void **)&d_Hx_i, size);
			cuda_malloc(GPU, UM, (void **)&d_Hy_i, size);
			cuda_malloc(GPU, UM, (void **)&d_Hz_i, size);
		}
	}
}

void memfree3_gpu()
{
	if ((runMode == 0) || (runMode == 2)) {
		// near field 1d
		// runMode = 2 : allocated in calcNear1d()
		if ((NNear1d > 0) && (NFreq2 > 0)) {
			free(Near1dEx);
			free(Near1dEy);
			free(Near1dEz);
			free(Near1dHx);
			free(Near1dHy);
			free(Near1dHz);
		}

		// near field 2d
		// runMode = 2 : allocated in calcNear2d()
		if ((NNear2d > 0) && (NFreq2 > 0)) {
			free(Near2dEx);
			free(Near2dEy);
			free(Near2dEz);
			free(Near2dHx);
			free(Near2dHy);
			free(Near2dHz);
		}
	}

	if (runMode == 0) {
		if ((NNear1d > 0) && (NFreq2 > 0)) {
			cuda_free(GPU, d_Near1dEx);
			cuda_free(GPU, d_Near1dEy);
			cuda_free(GPU, d_Near1dEz);
			cuda_free(GPU, d_Near1dHx);
			cuda_free(GPU, d_Near1dHy);
			cuda_free(GPU, d_Near1dHz);
		}

		if ((NNear2d > 0) && (NFreq2 > 0)) {
			cuda_free(GPU, d_Near2dEx);
			cuda_free(GPU, d_Near2dEy);
			cuda_free(GPU, d_Near2dEz);
			cuda_free(GPU, d_Near2dHx);
			cuda_free(GPU, d_Near2dHy);
			cuda_free(GPU, d_Near2dHz);
		}
	}
}
