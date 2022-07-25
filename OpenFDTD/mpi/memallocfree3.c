/*
memallocfree3.c (MPI)

alloc and free
(3) near field
*/

#include "ofd.h"
#include "ofd_mpi.h"


void memalloc3(void)
{
	size_t size;

	if      (runMode == 0) {
		// near field 1d
		if ((NNear1d > 0) && (NFreq2 > 0)) {
			size = Near1d_size;
			l_Near1dEx = (d_complex_t *)malloc(size);
			l_Near1dEy = (d_complex_t *)malloc(size);
			l_Near1dEz = (d_complex_t *)malloc(size);
			l_Near1dHx = (d_complex_t *)malloc(size);
			l_Near1dHy = (d_complex_t *)malloc(size);
			l_Near1dHz = (d_complex_t *)malloc(size);
			memset(l_Near1dEx, 0, size);
			memset(l_Near1dEy, 0, size);
			memset(l_Near1dEz, 0, size);
			memset(l_Near1dHx, 0, size);
			memset(l_Near1dHy, 0, size);
			memset(l_Near1dHz, 0, size);
		}

		// near field 2d
		if ((NNear2d > 0) && (NFreq2 > 0)) {
			size = Near2d_size;
			l_Near2dEx = (d_complex_t *)malloc(size);
			l_Near2dEy = (d_complex_t *)malloc(size);
			l_Near2dEz = (d_complex_t *)malloc(size);
			l_Near2dHx = (d_complex_t *)malloc(size);
			l_Near2dHy = (d_complex_t *)malloc(size);
			l_Near2dHz = (d_complex_t *)malloc(size);
			memset(l_Near2dEx, 0, size);
			memset(l_Near2dEy, 0, size);
			memset(l_Near2dEz, 0, size);
			memset(l_Near2dHx, 0, size);
			memset(l_Near2dHy, 0, size);
			memset(l_Near2dHz, 0, size);
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
		}
	}
}


void memfree3(void)
{
	if (runMode == 0) {
		// near field 1d
		if ((NNear1d > 0) && (NFreq2 > 0)) {
			free(l_Near1dEx);
			free(l_Near1dEy);
			free(l_Near1dEz);
			free(l_Near1dHx);
			free(l_Near1dHy);
			free(l_Near1dHz);
		}

		// near field 2d
		if ((NNear2d > 0) && (NFreq2 > 0)) {
			free(l_Near2dEx);
			free(l_Near2dEy);
			free(l_Near2dEz);
			free(l_Near2dHx);
			free(l_Near2dHy);
			free(l_Near2dHz);
		}
	}

	if (((runMode == 0) && (commSize > 1) && (commRank == 0)) ||
	    ((runMode == 2) && (commRank == 0))) {
		// runMode = 0 : allocated in comm_near1d()
		// runMode = 2 : allocated in calcNear1d()
		if ((NNear1d > 0) && (NFreq2 > 0)) {
			free(Near1dEx);
			free(Near1dEy);
			free(Near1dEz);
			free(Near1dHx);
			free(Near1dHy);
			free(Near1dHz);
		}

		// runMode = 0 : allocated in comm_near2d()
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
}
