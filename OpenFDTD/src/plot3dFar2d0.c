/*
plot3dFar2d0.c (OpenMOM/OpenFDTD/OpenTHFD)

plot a far2d pattern (3D)
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ev.h"

static void plot3dwire(int, double (*)[2][3], double);

void plot3dFar2d0(
	int divtheta, int divphi, double **d,
	int scaledb, int scaleuser, double scalemin, double scalemax,
	int nwire, double (*wire)[2][3], double rscale,
	int ncomment, char **comment, double hgt)
{
	if ((divtheta <= 0) || (divphi <= 0)) return;

	const double dbauto = 40;
	const double eps = 1e-10;
	const double pi = 4 * atan(1);

	// to dB
	if (scaledb) {
		for (int itheta = 0; itheta <= divtheta; itheta++) {
		for (int iphi   = 0; iphi   <= divphi;   iphi++  ) {
			d[itheta][iphi] = 20 * log10((d[itheta][iphi] > eps) ? d[itheta][iphi] : eps);
		}
		}
	}

	// max
	double dmax = d[0][0];
	for (int itheta = 0; itheta <= divtheta; itheta++) {
	for (int iphi   = 0; iphi   <= divphi;   iphi++  ) {
		if (d[itheta][iphi] > dmax) {
			dmax = d[itheta][iphi];
		}
	}
	}

	// min, max
	double fmin = 0, fmax = 0;
	if (scaleuser) {
		// user scale
		fmin = scalemin;
		fmax = scalemax;
	}
	else {
		// auto scale
		fmax = dmax;
		if (scaledb) {
			// dB
			fmin = fmax - dbauto;
		}
		else {
			// linear
			fmin = 0;
		}
	}

	// new page
	ev3d_newPage();

	// objects
	if (nwire && (rscale > 0)) {
		plot3dwire(nwire, wire, rscale);
	}

	// constant theta lines
	for (int itheta = 1; itheta < divtheta; itheta++) {
	for (int iphi   = 0; iphi   < divphi;   iphi++  ) {
		double theta = pi * itheta / divtheta;
		double phi0 = (2 * pi) * (iphi + 0) / divphi;
		double phi1 = (2 * pi) * (iphi + 1) / divphi;
		double f0 = d[itheta][iphi + 0];
		double f1 = d[itheta][iphi + 1];
		double r0 = (f0 - fmin) / (fmax - fmin);
		double r1 = (f1 - fmin) / (fmax - fmin);
		if (r0 < 0) r0 = 0;
		if (r1 < 0) r1 = 0;
		ev3d_setColorV((r0 + r1) / 2, 1);
		double x0 = r0 * sin(theta) * cos(phi0);
		double y0 = r0 * sin(theta) * sin(phi0);
		double z0 = r0 * cos(theta);
		double x1 = r1 * sin(theta) * cos(phi1);
		double y1 = r1 * sin(theta) * sin(phi1);
		double z1 = r1 * cos(theta);
		ev3d_drawLine(x0, y0, z0, x1, y1, z1);
	}
	}

	// constant phi lines
	for (int iphi   = 0; iphi   < divphi;   iphi++  ) {
	for (int itheta = 0; itheta < divtheta; itheta++) {
		double theta0 = pi * (itheta + 0) / divtheta;
		double theta1 = pi * (itheta + 1) / divtheta;
		double phi = (2 * pi) * iphi / divphi;
		double f0 = d[itheta + 0][iphi];
		double f1 = d[itheta + 1][iphi];
		double r0 = (f0 - fmin) / (fmax - fmin);
		double r1 = (f1 - fmin) / (fmax - fmin);
		if (r0 < 0) r0 = 0;
		if (r1 < 0) r1 = 0;
		ev3d_setColorV((r0 + r1) / 2, 1);
		double x0 = r0 * sin(theta0) * cos(phi);
		double y0 = r0 * sin(theta0) * sin(phi);
		double z0 = r0 * cos(theta0);
		double x1 = r1 * sin(theta1) * cos(phi);
		double y1 = r1 * sin(theta1) * sin(phi);
		double z1 = r1 * cos(theta1);
		ev3d_drawLine(x0, y0, z0, x1, y1, z1);
	}
	}

	// comment
	ev3d_setColor(0, 0, 0);
	for (int n = 0; n < ncomment; n++) {
		ev3d_drawTitle(hgt, comment[n]);
	}

	// XYZ arrows
	ev3d_index(1);
	const double ra = 1.0;
	ev3d_setColor(128, 128, 128);
	ev3d_drawLine(0, 0, 0, ra, 0, 0);
	ev3d_drawLine(0, 0, 0, 0, ra, 0);
	ev3d_drawLine(0, 0, 0, 0, 0, ra);

	const double rb = 1.1;
	ev3d_setColor(0, 0, 0);
	ev3d_drawString(rb, 0, 0, hgt, "X");
	ev3d_drawString(0, rb, 0, hgt, "Y");
	ev3d_drawString(0, 0, rb, hgt, "Z");
}

static void plot3dwire(int nwire, double (*wire)[2][3], double rscale)
{
	double gmin[3], gmax[3];
	for (int m = 0; m < 3; m++) {
		gmin[m] = +1e10;
		gmax[m] = -1e10;
	}
	for (int n = 0; n < nwire; n++) {
		for (int v = 0; v < 2; v++) {
			for (int m = 0; m < 3; m++) {
				double g = wire[n][v][m];
				if (g < gmin[m]) gmin[m] = g;
				if (g > gmax[m]) gmax[m] = g;
			}
		}
	}

	const double dx = gmax[0] - gmin[0];
	const double dy = gmax[1] - gmin[1];
	const double dz = gmax[2] - gmin[2];
	const double rr = sqrt((dx * dx) + (dy * dy) + (dz * dz)) / 2;
	const double x0 = (gmin[0] + gmax[0]) / 2;
	const double y0 = (gmin[1] + gmax[1]) / 2;
	const double z0 = (gmin[2] + gmax[2]) / 2;
	const double rf = rscale / rr;
	ev3d_setColor(0, 0, 0);
	for (int n = 0; n < nwire; n++) {
		double x1 = x0 + rf * (wire[n][0][0] - x0);
		double y1 = y0 + rf * (wire[n][0][1] - y0);
		double z1 = z0 + rf * (wire[n][0][2] - z0);
		double x2 = x0 + rf * (wire[n][1][0] - x0);
		double y2 = y0 + rf * (wire[n][1][1] - y0);
		double z2 = z0 + rf * (wire[n][1][2] - z0);
		ev3d_drawLine(x1, y1, z1, x2, y2, z2);
	}
}
