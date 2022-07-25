/*
setgeometry.c
*/

#include "ofd.h"

typedef struct {int xy, i, j, on;} segment_t;

void setgeometry(int nx, int ny, double lx, double ly, double h, int sdiv, int lgeometry, int nseg, segment_t seg[])
{
	assert((nx > 0) && (ny > 0) && (sdiv > 0) && (lgeometry > 0) && (nseg > 0));

	const double x0 = -lx / 2;
	const double y0 = -ly / 2;
	const double dx = lx / nx;
	const double dy = ly / ny;

	// geometry id
	int gid = 0;

	// substrate
	if (NMaterial > 2) {
		Geometry[gid].m = 2;
		Geometry[gid].shape = 1;
		Geometry[gid].g[0] = Xn[0];
		Geometry[gid].g[1] = Xn[Nx];
		Geometry[gid].g[2] = Yn[0];
		Geometry[gid].g[3] = Yn[Ny];
		Geometry[gid].g[4] = 0;
		Geometry[gid].g[5] = h;
		gid++;
	}

	// ground
	Geometry[gid].m = 1;
	Geometry[gid].shape = 1;
	Geometry[gid].g[0] = Xn[0];
	Geometry[gid].g[1] = Xn[Nx];
	Geometry[gid].g[2] = Yn[0];
	Geometry[gid].g[3] = Yn[Ny];
	Geometry[gid].g[4] = Zn[0];
	Geometry[gid].g[5] = 0;
	gid++;

	// feed
	Geometry[gid].m = 1;
	Geometry[gid].shape = 1;
	Geometry[gid].g[0] = 0;
	Geometry[gid].g[1] = 0;
	Geometry[gid].g[2] = 0;
	Geometry[gid].g[3] = 0;
	Geometry[gid].g[4] = 0;
	Geometry[gid].g[5] = h;
	gid++;

	// patch
	for (int n = 0; n < nseg; n++) {
		if (seg[n].on) {
			const int i = seg[n].i;
			const int j = seg[n].j;
			Geometry[gid].m = 1;
			Geometry[gid].shape = 1;
			assert((seg[n].xy == 1) || (seg[n].xy == 2));
			if      (seg[n].xy == 1) {
				assert(i >= 0 && i < nx && j >= 0 && j <= ny);
				Geometry[gid].g[0] = x0 + (i + 0) * dx;
				Geometry[gid].g[1] = x0 + (i + 1) * dx;
				Geometry[gid].g[2] =
				Geometry[gid].g[3] = y0 + j * dy;
			}
			else if (seg[n].xy == 2) {
				assert(i >= 0 && i <= nx && j >= 0 && j < ny);
				Geometry[gid].g[0] =
				Geometry[gid].g[1] = x0 + i * dx;
				Geometry[gid].g[2] = y0 + (j + 0) * dy;
				Geometry[gid].g[3] = y0 + (j + 1) * dy;
			}
			Geometry[gid].g[4] =
			Geometry[gid].g[5] = h;
			gid++;
		}
	}

	NGeometry = gid;
}
