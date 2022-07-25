/*
memallocfree3.c

alloc and free
(3) near field
*/

#include "ofd.h"
#include "ofd_prototype.h"

void memalloc3(void)
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
		}
	}
	else if (runMode == 1) {
		// near field full (complex)
		// freed in calcNear3d(*,2)
		if ((NN > 0) && (NFreq2 > 0)) {
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
/*
			// -> initfield.c
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
*/
		}
	}
}

void memfree3(void)
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
		// runMode = 2 : allocated in calcNear2d(0)
		if ((NNear2d > 0) && (NFreq2 > 0)) {
			free(Near2dEx);
			free(Near2dEy);
			free(Near2dEz);
			free(Near2dHx);
			free(Near2dHy);
			free(Near2dHz);
		}
	}
}
