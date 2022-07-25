/*
solve.c
*/

#include "ofd.h"
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
			pbcx();
		}
		if (PBCy) {
			pbcy();
		}
		if (PBCz) {
			pbcz();
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
			dftNear1d(itime, LNear1d, Near1dEx, Near1dEy, Near1dEz, Near1dHx, Near1dHy, Near1dHz);
			dftNear2d(itime, LNear2d, Near2dEx, Near2dEy, Near2dEz, Near2dHx, Near2dHy, Near2dHz);
		}
		else if (runMode == 1) {
			dftNear3d(itime);
		}

		// average and convergence
		if ((itime % Solver.nout == 0) || (itime == Solver.maxiter)) {
			// average
			double fsum[2];
			average(fsum);

			// average (plot)
			Eiter[Niter] = fsum[0];
			Hiter[Niter] = fsum[1];
			Niter++;

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

	// near3d
	if (runMode == 1) {
		for (int ic = 0; ic < 6; ic++) {
			if (mem) calcNear3d(ic, 0);
			calcNear3d(ic, 1);
			if (mem) calcNear3d(ic, 2);
		}
	}
}
