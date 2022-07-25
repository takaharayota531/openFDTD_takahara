/*
solve.c (MPI)
*/

#include "ofd.h"
#include "ofd_mpi.h"
#include "ofd_prototype.h"

void solve(int io, int mem, FILE *fp)
{
	double fmax[] = {0, 0};
	char   str[BUFSIZ];
	int    converged = 0;

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

		// ABC
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

		// PBC
		if (PBCx) {
			if (commSize > 1) {
				comm_pbcx();
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
			comm_boundary();
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

		// ABC
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

	// free
	if (mem) {
		memfree2();
	}

	// MPI : send to root
	if (commSize > 1) {
		// feed waveform
		if (NFeed) {
			comm_feed();
		}

		// point waveform
		if (NPoint) {
			comm_point();
		}

		// near field
		if      (runMode == 0) {
			// near1d
			if (NNear1d && NFreq2) {
				comm_near1d();
			}

			// near2d
			if (NNear2d && NFreq2) {
				comm_near2d();
			}
		}
		else if (runMode == 1) {
			// near3d
			if (NFreq2) {
				comm_near3d();
			}
		}
	}

	// non-MPI
	else {
		if      (runMode == 0) {
			// copy local pointer to global pointer
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
