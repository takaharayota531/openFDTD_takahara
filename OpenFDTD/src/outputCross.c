/*
outputCross.c

cross section
*/

#include "ofd.h"
#include "ofd_prototype.h"


static void logCross(FILE *fp, const double bcs[], const double fcs[])
{
	fprintf(fp, "=== cross section ===\n");
	fprintf(fp, "  %s\n", "frequency[Hz] backward[m*m]  forward[m*m]");
	for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
		fprintf(fp, "  %13.5e  %12.4e  %12.4e\n", Freq2[ifreq], bcs[ifreq], fcs[ifreq]);
	}
}


void outputCross(FILE *fp)
{
	double e[7];

	// calculate surface field
	if (runMode == 1) {
		calcNear2d(0);
		calcNear2d(1);
	}

	// setup
	setup_farfield();

	// calculation
	double *bcs = (double *)malloc(NFreq2 * sizeof(double));
	double *fcs = (double *)malloc(NFreq2 * sizeof(double));
	for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
		double ffctr = farfactor(ifreq);
		d_complex_t etheta, ephi;
		// backward
		farfield(ifreq, Planewave.theta, Planewave.phi, ffctr, &etheta, &ephi);
		farComponent(etheta, ephi, e);
		bcs[ifreq] = e[0] * e[0];
		// forward
		farfield(ifreq, 180 - Planewave.theta, Planewave.phi + 180, ffctr, &etheta, &ephi);
		farComponent(etheta, ephi, e);
		fcs[ifreq] = e[0] * e[0];
	}

	// output
	logCross(stdout, bcs, fcs);
	logCross(fp,     bcs, fcs);

	// free
	free(bcs);
	free(fcs);
}
