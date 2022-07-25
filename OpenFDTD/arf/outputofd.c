/*
outputofd.c

output OpenFDTD datafile
*/

#include "ofd.h"

void outputofd(const char fn[])
{
	FILE *fp;
	if ((fp = fopen(fn, "w")) == NULL) {
		fprintf(stderr, "file %s open error.\n", fn);
		exit(1);
	}

	// header
	fprintf(fp, "OpenFDTD 2 6\n");

	// mesh
	fprintf(fp, "xmesh = %g %d %g\n", Xn[0], Nx, Xn[Nx]);
	fprintf(fp, "ymesh = %g %d %g\n", Yn[0], Ny, Yn[Ny]);
	fprintf(fp, "zmesh = %g %d %g\n", Zn[0], Nz, Zn[Nz]);

	// material
	for (int m = 2; m < NMaterial; m++) {
		fprintf(fp, "material = 1 %g %g %g %g\n", Material[m].epsr, Material[m].esgm, Material[m].amur, Material[m].msgm);
	}

	// geometry
	for (int n = 0; n < NGeometry; n++) {
		fprintf(fp, "geometry = %d 1 %g %g %g %g %g %g\n", Geometry[n].m,
			Geometry[n].g[0], Geometry[n].g[1],
			Geometry[n].g[2], Geometry[n].g[3],
			Geometry[n].g[4], Geometry[n].g[5]);
	}

	// feed
	for (int ifeed = 0; ifeed < NFeed; ifeed++) {
		const char dir = Feed[ifeed].dir;
		const int i = Feed[ifeed].i;
		const int j = Feed[ifeed].j;
		const int k = Feed[ifeed].k;
		const double x = (dir == 'X') ? Xc[i] : (dir == 'Y') ? Yn[j] : Zn[k];
		const double y = (dir == 'X') ? Xn[i] : (dir == 'Y') ? Yc[j] : Zn[k];
		const double z = (dir == 'X') ? Xn[i] : (dir == 'Y') ? Yn[j] : Zc[k];
		fprintf(fp, "feed = %c %g %g %g %g %g %g\n", dir, x, y, z, Feed[ifeed].volt, Feed[ifeed].delay, Feed[ifeed].z0);
	}
	const double rint = 10;
	fprintf(fp, "rfeed = %g\n", rint);

	// misc.
	assert((NFreq1 > 0) && (NFreq2 > 0));
	fprintf(fp, "frequency1 = %g %g %d\n", Freq1[0], Freq1[NFreq1 - 1], NFreq1 - 1);
	fprintf(fp, "frequency2 = %g %g %d\n", Freq2[0], Freq2[NFreq2 - 1], NFreq2 - 1);
	fprintf(fp, "solver = %d %d %.1e\n", Solver.maxiter, Solver.nout, Solver.converg);
	fprintf(fp, "pulsewidth = %.3e\n", Tw);

	// tailor
	fprintf(fp, "end\n");

	fclose(fp);
}
