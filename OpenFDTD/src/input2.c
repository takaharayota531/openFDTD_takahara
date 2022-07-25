/*
input2.c
*/

#include "ofd.h"
#include "ofd_prototype.h"

// index and length
static void getindex(char dir, double x, double y, double z,
	int *i, int *j, int *k, double *dx, double *dy, double *dz)
{
	if      (dir == 'X') {
		*i = nearest(x, 0, Nx - 1, Xc);
		*j = nearest(y, 0, Ny,     Yn);
		*k = nearest(z, 0, Nz,     Zn);
		*dx = Xn[*i + 1] - Xn[*i];
		*dy = (Ny > 1) ? (Yc[*j] - Yc[*j - 1]) : (Yn[1] - Yn[0]);
		*dz = (Nz > 1) ? (Zc[*k] - Zc[*k - 1]) : (Zn[1] - Zn[0]);
	}
	else if (dir == 'Y') {
		*j = nearest(y, 0, Ny - 1, Yc);
		*k = nearest(z, 0, Nz,     Zn);
		*i = nearest(x, 0, Nx,     Xn);
		*dy = Yn[*j + 1] - Yn[*j];
		*dz = (Nz > 1) ? (Zc[*k] - Zc[*k - 1]) : (Zn[1] - Zn[0]);
		*dx = (Nx > 1) ? (Xc[*i] - Xc[*i - 1]) : (Xn[1] - Xn[0]);
	}
	else if (dir == 'Z') {
		*k = nearest(z, 0, Nz - 1, Zc);
		*i = nearest(x, 0, Nx,     Xn);
		*j = nearest(y, 0, Ny,     Yn);
		*dz = Zn[*k + 1] - Zn[*k];
		*dx = (Nx > 1) ? (Xc[*i] - Xc[*i - 1]) : (Xn[1] - Xn[0]);
		*dy = (Ny > 1) ? (Yc[*j] - Yc[*j - 1]) : (Yn[1] - Yn[0]);
	}
}


// number of cells
void setup_cells(int nxr, int nyr, int nzr, int *dxr, int *dyr, int *dzr)
{
	int xsum = 0;
	for (int i = 0; i < nxr; i++) {
		xsum += dxr[i];
	}
	Nx = xsum;

	int ysum = 0;
	for (int j = 0; j < nyr; j++) {
		ysum += dyr[j];
	}
	Ny = ysum;

	int zsum = 0;
	for (int k = 0; k < nzr; k++) {
		zsum += dzr[k];
	}
	Nz = zsum;
}


// node
void setup_node(int nxr, int nyr, int nzr, double *xr, double *yr, double *zr, int *dxr, int *dyr, int *dzr)
{
	if ((nxr <= 0) || (nyr <= 0) || (nzr <= 0)) return;

	Xn = (double *)malloc((Nx + 1) * sizeof(double));
	Yn = (double *)malloc((Ny + 1) * sizeof(double));
	Zn = (double *)malloc((Nz + 1) * sizeof(double));

	int xid = 0;
	for (int mx = 0; mx < nxr; mx++) {
		double dx = (xr[mx + 1] - xr[mx]) / dxr[mx];
		for (int i = 0; i < dxr[mx]; i++) {
			Xn[xid++] = xr[mx] + (i * dx);
		}
	}
	Xn[xid] = xr[nxr];

	int yid = 0;
	for (int my = 0; my < nyr; my++) {
		double dy = (yr[my + 1] - yr[my]) / dyr[my];
		for (int j = 0; j < dyr[my]; j++) {
			Yn[yid++] = yr[my] + (j * dy);
		}
	}
	Yn[yid] = yr[nyr];

	int zid = 0;
	for (int mz = 0; mz < nzr; mz++) {
		double dz = (zr[mz + 1] - zr[mz]) / dzr[mz];
		for (int k = 0; k < dzr[mz]; k++) {
			Zn[zid++] = zr[mz] + (k * dz);
		}
	}
	Zn[zid] = zr[nzr];
/*
	// debug
	for (int i = 0; i <= Nx; i++) {
		printf("Xn[%d]=%.5f\n", i, Xn[i] * 1e3);
	}
	for (int j = 0; j <= Ny; j++) {
		printf("Yn[%d]=%.5f\n", j, Yn[j] * 1e3);
	}
	for (int k = 0; k <= Nz; k++) {
		printf("Zn[%d]=%.5f\n", k, Zn[k] * 1e3);
	}
*/
}


// cell center
void setup_center(void)
{
	Xc = (double *)malloc(Nx * sizeof(double));
	Yc = (double *)malloc(Ny * sizeof(double));
	Zc = (double *)malloc(Nz * sizeof(double));

	for (int i = 0; i < Nx; i++) {
		Xc[i] = (Xn[i] + Xn[i + 1]) / 2;
	}
	for (int j = 0; j < Ny; j++) {
		Yc[j] = (Yn[j] + Yn[j + 1]) / 2;
	}
	for (int k = 0; k < Nz; k++) {
		Zc[k] = (Zn[k] + Zn[k + 1]) / 2;
	}
/*
	// debug
	for (int i = 0; i < Nx; i++) {
		printf("Xc[%d]=%.5f\n", i, Xc[i] * 1e3);
	}
	for (int j = 0; j < Ny; j++) {
		printf("Yc[%d]=%.5f\n", j, Yc[j] * 1e3);
	}
	for (int k = 0; k < Nz; k++) {
		printf("Zc[%d]=%.5f\n", k, Zc[k] * 1e3);
	}
*/
}


// fit geometry without thickness to the nearest node
void fitgeometry(void)
{
	if ((Nx <= 0) || (Ny <= 0) || (Nz <= 0)) return;

	double d0 = EPS * (
		fabs(Xn[Nx] - Xn[0]) +
		fabs(Yn[Ny] - Yn[0]) +
		fabs(Zn[Nz] - Zn[0]));

	for (int n = 0; n < NGeometry; n++) {
		int    shape = Geometry[n].shape;
		double *g = Geometry[n].g;

		if      ((shape == 1) || (shape == 2)) {
			if (fabs(g[0] - g[1]) < d0) {
				int i = nearest(g[0], 0, Nx, Xn);
				g[0] = g[1] = Xn[i];
			}
			if (fabs(g[2] - g[3]) < d0) {
				int j = nearest(g[2], 0, Ny, Yn);
				g[2] = g[3] = Yn[j];
			}
			if (fabs(g[4] - g[5]) < d0) {
				int k = nearest(g[4], 0, Nz, Zn);
				g[4] = g[5] = Zn[k];
			}
		}
		//else if ((shape == 11) || (shape == 64) || (shape == 65)) {
		else if (shape == 11) {
			if (fabs(g[0] - g[1]) < d0) {
				int i = nearest(g[0], 0, Nx, Xn);
				g[0] = g[1] = Xn[i];
			}
		}
		//else if ((shape == 12) || (shape == 61) || (shape == 66)) {
		else if (shape == 12) {
			if (fabs(g[2] - g[3]) < d0) {
				int j = nearest(g[2], 0, Ny, Yn);
				g[2] = g[3] = Yn[j];
			}
		}
		//else if ((shape == 13) || (shape == 62) || (shape == 63)) {
		else if (shape == 13) {
			if (fabs(g[4] - g[5]) < d0) {
				int k = nearest(g[4], 0, Nz, Zn);
				g[4] = g[5] = Zn[k];
			}
		}
	}
}


// plane wave
void setup_planewave(void)
{
	const double cost = cos(Planewave.theta * DTOR);
	const double sint = sin(Planewave.theta * DTOR);
	const double cosp = cos(Planewave.phi   * DTOR);
	const double sinp = sin(Planewave.phi   * DTOR);

	// unit vector in (r, theta, phi)
	double r1[3], t1[3], p1[3];
	r1[0] = + sint * cosp;
	r1[1] = + sint * sinp;
	r1[2] = + cost;
	t1[0] = + cost * cosp;
	t1[1] = + cost * sinp;
	t1[2] = - sint;
	p1[0] = - sinp;
	p1[1] = + cosp;
	p1[2] = 0;

	// propagation vector = - r1
	Planewave.ri[0] = - r1[0];
	Planewave.ri[1] = - r1[1];
	Planewave.ri[2] = - r1[2];

	// E
	if      (Planewave.pol == 1) {
		// V-pol
		Planewave.ei[0] = - t1[0];
		Planewave.ei[1] = - t1[1];
		Planewave.ei[2] = - t1[2];
	}
	else if (Planewave.pol == 2) {
		// H-pol
		Planewave.ei[0] = + p1[0];
		Planewave.ei[1] = + p1[1];
		Planewave.ei[2] = + p1[2];
	}

	// H = E X r
	Planewave.hi[0] = (Planewave.ei[1] * r1[2]) - (Planewave.ei[2] * r1[1]);
	Planewave.hi[1] = (Planewave.ei[2] * r1[0]) - (Planewave.ei[0] * r1[2]);
	Planewave.hi[2] = (Planewave.ei[0] * r1[1]) - (Planewave.ei[1] * r1[0]);

	// initial position
	const double f0 = (Freq2[0] + Freq2[NFreq2 - 1]) / 2;
	const double r = sqrt((Xn[0] - Xn[Nx]) * (Xn[0] - Xn[Nx]) +
	                      (Yn[0] - Yn[Ny]) * (Yn[0] - Yn[Ny]) +
	                      (Zn[0] - Zn[Nz]) * (Zn[0] - Zn[Nz])) / 2 + (0.5 * C / f0);
	Planewave.r0[0] = ((Xn[0] + Xn[Nx]) / 2 - (r * Planewave.ri[0]));
	Planewave.r0[1] = ((Yn[0] + Yn[Ny]) / 2 - (r * Planewave.ri[1]));
	Planewave.r0[2] = ((Zn[0] + Zn[Nz]) / 2 - (r * Planewave.ri[2]));
	//printf("%f %f %f\n", Planewave.r0[0], Planewave.r0[1], Planewave.r0[2]);

	// waveform parameter
	Planewave.ai = 4 / (1.27 / f0);
}


// feed
void setup_feed(const double *x, const double *y, const double *z)
{
	for (int n = 0; n < NFeed; n++) {
		getindex(Feed[n].dir, x[n], y[n], z[n],
			&Feed[n].i, &Feed[n].j, &Feed[n].k, &Feed[n].dx, &Feed[n].dy, &Feed[n].dz);
	}
}


// point
void setup_point(const double *x, const double *y, const double *z, const char strprop[])
{
	for (int n = 0; n < NPoint; n++) {
		getindex(Point[n].dir, x[n], y[n], z[n],
			&Point[n].i, &Point[n].j, &Point[n].k, &Point[n].dx, &Point[n].dy, &Point[n].dz);
	}

	// add 1+, 1- points
	Point = (point_t *)realloc(Point, (NPoint + 2) * sizeof(point_t));
	Point[NPoint] = Point[NPoint + 1] = Point[0];
	if      (!strcmp(strprop, "+X") || !strcmp(strprop, "-X")) {
		Point[NPoint].i     = Point[0].i + (!strcmp(strprop, "+X") ? +1 : -1);
		Point[NPoint + 1].i = Point[0].i + (!strcmp(strprop, "+X") ? -1 : +1);
		Point[NPoint].j = Point[NPoint + 1].j = Point[0].j;
		Point[NPoint].k = Point[NPoint + 1].k = Point[0].k;
	}
	else if (!strcmp(strprop, "+Y") || !strcmp(strprop, "-Y")) {
		Point[NPoint].j     = Point[0].j + (!strcmp(strprop, "+Y") ? +1 : -1);
		Point[NPoint + 1].j = Point[0].j + (!strcmp(strprop, "+Y") ? -1 : +1);
		Point[NPoint].k = Point[NPoint + 1].k = Point[0].k;
		Point[NPoint].i = Point[NPoint + 1].i = Point[0].i;
	}
	else if (!strcmp(strprop, "+Z") || !strcmp(strprop, "-Z")) {
		Point[NPoint].k     = Point[0].k + (!strcmp(strprop, "+Z") ? +1 : -1);
		Point[NPoint + 1].k = Point[0].k + (!strcmp(strprop, "+Z") ? -1 : +1);
		Point[NPoint].i = Point[NPoint + 1].i = Point[0].i;
		Point[NPoint].j = Point[NPoint + 1].j = Point[0].j;
	}
/*
	for (int n = 0; n < NPoint + 2; n++) {
		printf("%d %c %d %d %d %f %f %f\n", n, Point[n].dir, Point[n].i, Point[n].j, Point[n].k, Point[n].dx, Point[n].dy, Point[n].dz);
	}
*/
}


// load
void setup_load(int nload, char *dload, double *xload, double *yload, double *zload, char *cload, double *pload, int array_inc)
{
	for (int n = 0; n < nload; n++) {
		const char dir = dload[n];
		const char rcl = cload[n];
		int    i = 0, j = 0, k = 0;
		double dx = 0, dy = 0, dz = 0;
		getindex(dload[n], xload[n], yload[n], zload[n], &i, &j, &k, &dx, &dy, &dz);
		const double dlds = (dir == 'X') ? (dx / (dy * dz))
		                  : (dir == 'Y') ? (dy / (dz * dx))
		                  : (dir == 'Z') ? (dz / (dx * dy)) : 0;

		if      ((rcl == 'R') || (rcl == 'C')) {
			// material
			if (NMaterial % array_inc == 2) {
				Material = (material_t *)realloc(Material, (NMaterial + array_inc) * sizeof(material_t));
			}
			if (NMaterial >= MAXMATERIAL) {
				fprintf(stderr, "*** too many load(R,C)\n");
				exit(1);
			}
			Material[NMaterial].type = 1;
			Material[NMaterial].epsr = (rcl == 'R') ? 1 : (pload[n] / EPS0) * dlds;
			Material[NMaterial].esgm = (rcl == 'R') ? (1 / pload[n]) * dlds : 0;
			Material[NMaterial].amur = 1;
			Material[NMaterial].msgm = 0;
			//printf("%zd %d %f %f %f %f\n", NMaterial, Material[NMaterial].type, Material[NMaterial].epsr, Material[NMaterial].esgm, Material[NMaterial].amur, Material[NMaterial].msgm);
			// geometry
			if (NGeometry % array_inc == 0) {
				Geometry = (geometry_t *)realloc(Geometry, (NGeometry + array_inc) * sizeof(geometry_t));
			}
			geometry_t *ptr = &Geometry[NGeometry];
			ptr->m = (id_t)NMaterial;
			ptr->shape = 1;
			if      (dir == 'X') {
				ptr->g[0] = Xn[i    ];
				ptr->g[1] = Xn[i + 1];
				ptr->g[2] =
				ptr->g[3] = Yn[j];
				ptr->g[4] =
				ptr->g[5] = Zn[k];
			}
			else if (dir == 'Y') {
				ptr->g[2] = Yn[j    ];
				ptr->g[3] = Yn[j + 1];
				ptr->g[4] =
				ptr->g[5] = Zn[k];
				ptr->g[0] =
				ptr->g[1] = Xn[i];
			}
			else if (dir == 'Z') {
				ptr->g[4] = Zn[k    ];
				ptr->g[5] = Zn[k + 1];
				ptr->g[0] =
				ptr->g[1] = Xn[i];
				ptr->g[2] =
				ptr->g[3] = Yn[j];
			}
			NMaterial++;
			NGeometry++;
		}
		else if (rcl == 'L') {
			Inductor = (inductor_t *)realloc(Inductor, (NInductor + 1) * sizeof(inductor_t));
			Inductor[NInductor].dir = dir;
			Inductor[NInductor].i = i;
			Inductor[NInductor].j = j;
			Inductor[NInductor].k = k;
			Inductor[NInductor].dx = dx;
			Inductor[NInductor].dy = dy;
			Inductor[NInductor].dz = dz;
			Inductor[NInductor].fctr = MU0 * dlds / pload[n];
			//printf("%d %c %d %d %d %e\n", NInductor, Inductor[NInductor].dir, Inductor[NInductor].i, Inductor[NInductor].j, Inductor[NInductor].k, Inductor[NInductor].fctr);
			Inductor[NInductor].e =
			Inductor[NInductor].esum = 0;
			NInductor++;
		}
	}
}


// near1d
void setup_near1d(void)
{
	if ((runMode == 0) || (runMode == 1)) {
		for (int n = 0; n < NNear1d; n++) {
			if      (Near1d[n].dir == 'X') {
				Near1d[n].id1 = nearest(Near1d[n].pos1, 0, Ny, Yn);
				Near1d[n].id2 = nearest(Near1d[n].pos2, 0, Nz, Zn);
			}
			else if (Near1d[n].dir == 'Y') {
				Near1d[n].id1 = nearest(Near1d[n].pos1, 0, Nz, Zn);
				Near1d[n].id2 = nearest(Near1d[n].pos2, 0, Nx, Xn);
			}
			else if (Near1d[n].dir == 'Z') {
				Near1d[n].id1 = nearest(Near1d[n].pos1, 0, Nx, Xn);
				Near1d[n].id2 = nearest(Near1d[n].pos2, 0, Ny, Yn);
			}
		}
	}
/*
	// debug
	for (int n = 0; n < NNear1d; n++) {
		printf("near1d : %d %c %e %d %e %d\n", n, Near1d[n].dir, Near1d[n].pos1, Near1d[n].id1, Near1d[n].pos2, Near1d[n].id2);
	}
	fflush(stdout);
*/
}


// near2d : add 6 boundaries
void setup_near2d(void)
{
	if (runMode == 1) {
		NNear2d = 6;
		Near2d = (near2d_t *)malloc(NNear2d * sizeof(near2d_t));
		Near2d[0].dir =
		Near2d[1].dir = 'X';
		Near2d[2].dir =
		Near2d[3].dir = 'Y';
		Near2d[4].dir =
		Near2d[5].dir = 'Z';
		Near2d[0].id0 = 0;
		Near2d[1].id0 = Nx;
		Near2d[2].id0 = 0;
		Near2d[3].id0 = Ny;
		Near2d[4].id0 = 0;
		Near2d[5].id0 = Nz;
	}
	else {
		// add surface data
		Near2d = (near2d_t *)realloc(Near2d, (NNear2d + 6) * sizeof(near2d_t));
		// surface direction
		Near2d[NNear2d + 0].dir =
		Near2d[NNear2d + 1].dir = 'X';
		Near2d[NNear2d + 2].dir =
		Near2d[NNear2d + 3].dir = 'Y';
		Near2d[NNear2d + 4].dir =
		Near2d[NNear2d + 5].dir = 'Z';

		if (runMode == 0) {
			// position
			Near2d[NNear2d + 0].pos0 = Xn[0];
			Near2d[NNear2d + 1].pos0 = Xn[Nx];
			Near2d[NNear2d + 2].pos0 = Yn[0];
			Near2d[NNear2d + 3].pos0 = Yn[Ny];
			Near2d[NNear2d + 4].pos0 = Zn[0];
			Near2d[NNear2d + 5].pos0 = Zn[Nz];
			// node index
			for (int n = 0; n < NNear2d + 6; n++) {
				if      (Near2d[n].dir == 'X') {
					Near2d[n].id0 = nearest(Near2d[n].pos0, 0, Nx, Xn);
				}
				else if (Near2d[n].dir == 'Y') {
					Near2d[n].id0 = nearest(Near2d[n].pos0, 0, Ny, Yn);
				}
				else if (Near2d[n].dir == 'Z') {
					Near2d[n].id0 = nearest(Near2d[n].pos0, 0, Nz, Zn);
				}
			}
		}
		// number of near2d
		NNear2d += 6;
	}
/*
	// debug
	for (int n = 0; n < NNear2d; n++) {
		printf("near2d : %d %s %c %e %d\n", n, Near2d[n].cmp, Near2d[n].dir, Near2d[n].pos0, Near2d[n].id0);
	}
*/
}
