/*
solve.cu (CUDA + MPI)
*/

#ifdef _MPI
#include <mpi.h>
#endif

#include "ofd.h"
#include "ofd_mpi.h"
#include "ofd_cuda.h"
#include "ofd_prototype.h"


void solve(int io, int mem, FILE *fp)
{
	double fmax[] = {0, 0};
	char   str[BUFSIZ];
	int    converged = 0;

	// setup host memory
	setup_host();

	// setup (GPU)
	if (GPU) {
		setup_gpu();
	}

	// initial field
	initfield();

	// time step iteration
	int itime;
	double t = 0;
	for (itime = 0; itime <= Solver.maxiter; itime++) {

		// update H
		t += 0.5 * Dt;
		updateHx(t);
		updateHy(t);
		updateHz(t);

		// ABC H
		if      (iABC == 0) {
			murHx();
			murHy();
			murHz();
		}
		else if (iABC == 1) {
			pmlHx();
			pmlHy();
			pmlHz();
		}

		// PBC H
		if (PBCx) {
			if (commSize > 1) {
				comm_cuda_pbcx();
			}
			else {
				pbcx();
			}
		}
		if (PBCy) {
			pbcy();
		}
		if (PBCz) {
			pbcz();
		}

		// share boundary H (MPI)
		if (commSize > 1) {
			comm_cuda_boundary();
		}

		// update E
		t += 0.5 * Dt;
		updateEx(t);
		updateEy(t);
		updateEz(t);

		// dispersion E
		if (numDispersionEx) {
			dispersionEx(t);
		}
		if (numDispersionEy) {
			dispersionEy(t);
		}
		if (numDispersionEz) {
			dispersionEz(t);
		}

		// ABC E
		if      (iABC == 1) {
			pmlEx();
			pmlEy();
			pmlEz();
		}

		// feed
		if (NFeed) {
			efeed(itime);
		}

		// inductor
		if (NInductor) {
			eload();
		}

		// point
		if (NPoint) {
			vpoint(itime);
		}

		// DFT
		if      (runMode == 0) {
			dftNear1d(itime, l_LNear1d, l_Near1dEx, l_Near1dEy, l_Near1dEz, l_Near1dHx, l_Near1dHy, l_Near1dHz);
			dftNear2d(itime, l_LNear2d, l_Near2dEx, l_Near2dEy, l_Near2dEz, l_Near2dHx, l_Near2dHy, l_Near2dHz);
		}
		else if (runMode == 1) {
			dftNear3d(itime);
		}

		// average and convergence
		if ((itime % Solver.nout == 0) || (itime == Solver.maxiter)) {
			// average
			double fsum[2];
			average(fsum);

			// allreduce average (MPI)
			if (commSize > 1) {
				comm_average(fsum);
			}

			// average
			if (commRank == 0) {
				Eiter[Niter] = fsum[0];
				Hiter[Niter] = fsum[1];
				Niter++;
			}

			// monitor
			if (io) {
				sprintf(str, "%7d %.6f %.6f", itime, fsum[0], fsum[1]);
				fprintf(fp,     "%s\n", str);
				fprintf(stdout, "%s\n", str);
				fflush(fp);
				fflush(stdout);
			}

			// check convergence
			fmax[0] = MAX(fmax[0], fsum[0]);
			fmax[1] = MAX(fmax[1], fsum[1]);
			if ((fsum[0] < fmax[0] * Solver.converg) &&
			    (fsum[1] < fmax[1] * Solver.converg)) {
				converged = 1;
				break;
			}
		}
	}

	// result
	if (io) {
		sprintf(str, "    --- %s ---", (converged ? "converged" : "max steps"));
		fprintf(fp,     "%s\n", str);
		fprintf(stdout, "%s\n", str);
		fflush(fp);
		fflush(stdout);
	}

	// time steps
	Ntime = itime + converged;

	// copy device to host
	if (GPU) {
		if (NFeed > 0) {
			cuda_memcpy(GPU, VFeed, d_VFeed, Feed_size, cudaMemcpyDeviceToHost);
			cuda_memcpy(GPU, IFeed, d_IFeed, Feed_size, cudaMemcpyDeviceToHost);
		}

		if (NPoint > 0) {
			cuda_memcpy(GPU, VPoint, d_VPoint, Point_size, cudaMemcpyDeviceToHost);
		}

		if      (runMode == 0) {
			if ((NNear1d > 0) && (NFreq2 > 0)) {
				cuda_memcpy(GPU, l_Near1dEx, d_Near1dEx, Near1d_size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, l_Near1dEy, d_Near1dEy, Near1d_size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, l_Near1dEz, d_Near1dEz, Near1d_size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, l_Near1dHx, d_Near1dHx, Near1d_size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, l_Near1dHy, d_Near1dHy, Near1d_size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, l_Near1dHz, d_Near1dHz, Near1d_size, cudaMemcpyDeviceToHost);
			}

			if ((NNear2d > 0) && (NFreq2 > 0)) {
				cuda_memcpy(GPU, l_Near2dEx, d_Near2dEx, Near2d_size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, l_Near2dEy, d_Near2dEy, Near2d_size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, l_Near2dEz, d_Near2dEz, Near2d_size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, l_Near2dHx, d_Near2dHx, Near2d_size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, l_Near2dHy, d_Near2dHy, Near2d_size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, l_Near2dHz, d_Near2dHz, Near2d_size, cudaMemcpyDeviceToHost);
			}
		}
		else if (runMode == 1) {
			if ((NN > 0) && (NFreq2 > 0)) {
				size_t size = NN * NFreq2 * sizeof(real_t);
				cuda_memcpy(GPU, Ex_r, d_Ex_r, size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, Ey_r, d_Ey_r, size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, Ez_r, d_Ez_r, size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, Hx_r, d_Hx_r, size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, Hy_r, d_Hy_r, size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, Hz_r, d_Hz_r, size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, Ex_i, d_Ex_i, size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, Ey_i, d_Ey_i, size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, Ez_i, d_Ez_i, size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, Hx_i, d_Hx_i, size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, Hy_i, d_Hy_i, size, cudaMemcpyDeviceToHost);
				cuda_memcpy(GPU, Hz_i, d_Hz_i, size, cudaMemcpyDeviceToHost);
			}
		}
	}

	// free
	if (mem) {
		memfree2_gpu();
	}

	// MPI : send to root
	if (commSize > 1) {
		// feed waveform
		if (NFeed > 0) {
			comm_feed();
		}

		// point waveform
		if (NPoint > 0) {
			comm_point();
		}

		// near field
		if      (runMode == 0) {
			// near1d
			if ((NNear1d > 0) && (NFreq2 > 0)) {
				comm_near1d();
			}

			// near2d
			if ((NNear2d > 0) && (NFreq2 > 0)) {
				comm_near2d();
			}
		}
		else if (runMode == 1) {
			// near3d
			if (NFreq2 > 0) {
				comm_near3d();
			}
		}
	}

	// non-MPI: copy pointer local to global
	else {
		if      (runMode == 0) {
			LNear1d  = l_LNear1d;
			Near1dEx = l_Near1dEx;
			Near1dEy = l_Near1dEy;
			Near1dEz = l_Near1dEz;
			Near1dHx = l_Near1dHx;
			Near1dHy = l_Near1dHy;
			Near1dHz = l_Near1dHz;

			LNear2d  = l_LNear2d;
			Near2dEx = l_Near2dEx;
			Near2dEy = l_Near2dEy;
			Near2dEz = l_Near2dEz;
			Near2dHx = l_Near2dHx;
			Near2dHy = l_Near2dHy;
			Near2dHz = l_Near2dHz;
		}
		else if (runMode == 1) {
			for (int ic = 0; ic < 6; ic++) {
				if (mem) calcNear3d(ic, 0);
				calcNear3d(ic, 1);
				if (mem) calcNear3d(ic, 2);
			}
		}
	}
}
