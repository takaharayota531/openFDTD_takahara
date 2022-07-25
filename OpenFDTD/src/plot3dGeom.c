/*
plot3dGeom.c

plot geometry 3D
*/

#include "ofd.h"
#include "ev.h"
#include "ofd_prototype.h"

// geometry
static void plot3dGeom_g(void)
{
	// setup

	int *shape     =         (int *)malloc(NGeometry * sizeof(int));
	int *mid       =         (int *)malloc(NGeometry * sizeof(int));
	double (*g)[8] = (double (*)[8])malloc(NGeometry * 8 * sizeof(double));
	for (int n = 0; n < NGeometry; n++) {
		shape[n] = Geometry[n].shape;
		mid[n] = Geometry[n].m;
		memcpy(g[n], Geometry[n].g, 8 * sizeof(double));
	}

	NGline = geomlines(0, (int)NGeometry, shape, NULL, NULL, NULL, NULL, 0);

	Gline  = (double (*)[2][3])malloc(NGline * 2 * 3 * sizeof(double));
	MGline =           (id_t *)malloc(NGline * sizeof(id_t));
	int *mgline =       (int *)malloc(NGline * sizeof(int));
	memset(Gline, 0, NGline * 2 * 3 * sizeof(double));
	memset(mgline, 0, NGline * sizeof(int));

	const double eps = EPS * (fabs(Xn[Nx] - Xn[0]) + fabs(Yn[Ny] - Yn[0]) + fabs(Zn[Nz] - Zn[0]));
	geomlines(1, (int)NGeometry, shape, mid, g, Gline, mgline, eps);

	for (int n = 0; n < NGline; n++) {
		MGline[n] = (id_t)mgline[n];
	}

	free(shape);
	free(mid);
	free(g);
	free(mgline);

	// plot

	const unsigned char rgb[][3] = {
		{  0,   0,   0},	// PEC
		{255,   0, 255}		// dielectrics
	};

	for (int n = 0; n < NGline; n++) {
		int m = (MGline[n] == PEC) ? 0 : 1;
		ev3d_setColor(rgb[m][0], rgb[m][1], rgb[m][2]);
		ev3d_drawLine(Gline[n][0][0], Gline[n][0][1], Gline[n][0][2],
		              Gline[n][1][0], Gline[n][1][1], Gline[n][1][2]);
	}
}

// mesh
static void plot3dGeom_m(void)
{
	if ((Nx <= 0) || (Ny <= 0) || (Nz <= 0)) return;

	// gray
	ev3d_setColor(200, 200, 200);

	double x1 = Xn[0];
	double x2 = Xn[Nx];
	double y1 = Yn[0];
	double y2 = Yn[Ny];
	double z1 = Zn[0];
	double z2 = Zn[Nz];

	// X constant
	for (int i = 0; i <= Nx; i++) {
		double x = Xn[i];
		ev3d_drawLine(x, y1, z1, x, y2, z1);
		ev3d_drawLine(x, y1, z1, x, y1, z2);
	}

	// Y constant
	for (int j = 0; j <= Ny; j++) {
		double y = Yn[j];
		ev3d_drawLine(x1, y, z1, x1, y, z2);
		ev3d_drawLine(x1, y, z1, x2, y, z1);
	}

	// Z constant
	for (int k = 0; k <= Nz; k++) {
		double z = Zn[k];
		ev3d_drawLine(x1, y1, z, x2, y1, z);
		ev3d_drawLine(x1, y1, z, x1, y2, z);
	}
}

// feed
static void plot3dGeom_f(void)
{
	if (NFeed <= 0) return;

	// red
	ev3d_setColor(255, 0, 0);

	for (int n = 0; n < NFeed; n++) {
		int i = Feed[n].i;
		int j = Feed[n].j;
		int k = Feed[n].k;
		if      (Feed[n].dir == 'X') {
			ev3d_drawLine(Xn[i], Yn[j], Zn[k], Xn[i + 1], Yn[j], Zn[k]);
		}
		else if (Feed[n].dir == 'Y') {
			ev3d_drawLine(Xn[i], Yn[j], Zn[k], Xn[i], Yn[j + 1], Zn[k]);
		}
		else if (Feed[n].dir == 'Z') {
			ev3d_drawLine(Xn[i], Yn[j], Zn[k], Xn[i], Yn[j], Zn[k + 1]);
		}
	}
}

// load (inductor)
static void plot3dGeom_l(void)
{
	if (NInductor <= 0) return;

	// orange
	ev3d_setColor(255, 165, 0);

	for (int n = 0; n < NInductor; n++) {
		char dir = Inductor[n].dir;
		int i    = Inductor[n].i;
		int j    = Inductor[n].j;
		int k    = Inductor[n].k;
		if      (dir == 'X') {
			ev3d_drawLine(Xn[i], Yn[j], Zn[k], Xn[i + 1], Yn[j], Zn[k]);
		}
		else if (dir == 'Y') {
			ev3d_drawLine(Xn[i], Yn[j], Zn[k], Xn[i], Yn[j + 1], Zn[k]);
		}
		else if (dir == 'Z') {
			ev3d_drawLine(Xn[i], Yn[j], Zn[k], Xn[i], Yn[j], Zn[k + 1]);
		}
	}
}

// point
static void plot3dGeom_p(void)
{
	if (NPoint <= 0) return;

	// green
	ev3d_setColor(0, 255, 0);

	for (int n = 0; n < NPoint + 2; n++) {
		int i = Point[n].i;
		int j = Point[n].j;
		int k = Point[n].k;
		if      (Point[n].dir == 'X') {
			ev3d_drawLine(Xn[i], Yn[j], Zn[k], Xn[i + 1], Yn[j], Zn[k]);
		}
		else if (Point[n].dir == 'Y') {
			ev3d_drawLine(Xn[i], Yn[j], Zn[k], Xn[i], Yn[j + 1], Zn[k]);
		}
		else if (Point[n].dir == 'Z') {
			ev3d_drawLine(Xn[i], Yn[j], Zn[k], Xn[i], Yn[j], Zn[k + 1]);
		}
	}
}

// near1d
static void plot3dGeom_n1(void)
{
	// green
	ev3d_setColor(0, 255, 0);

	for (int n = 0; n < NNear1d; n++) {
		if      (Near1d[n].dir == 'X') {
			double y = Yn[Near1d[n].id1];
			double z = Zn[Near1d[n].id2];
			ev3d_drawLine(Xn[0], y, z, Xn[Nx], y, z);
		}
		else if (Near1d[n].dir == 'Y') {
			double z = Zn[Near1d[n].id1];
			double x = Xn[Near1d[n].id2];
			ev3d_drawLine(x, Yn[0], z, x, Yn[Ny], z);
		}
		else if (Near1d[n].dir == 'Z') {
			double x = Xn[Near1d[n].id1];
			double y = Yn[Near1d[n].id2];
			ev3d_drawLine(x, y, Zn[0], x, y, Zn[Nz]);
		}
	}
}

// near2d
static void plot3dGeom_n2(void)
{
	const int nsurface = (IPlanewave || IFar0d || NFar1d || NFar2d) ? 6 : 0;

	// green
	ev3d_setColor(0, 255, 0);

	for (int n = 0; n < NNear2d - nsurface; n++) {
		if      (Near2d[n].dir == 'X') {
			ev3d_drawRectangle('X', Xn[Near2d[n].id0], Yn[0], Zn[0], Yn[Ny], Zn[Nz]);
		}
		else if (Near2d[n].dir == 'Y') {
			ev3d_drawRectangle('Y', Yn[Near2d[n].id0], Zn[0], Xn[0], Zn[Nz], Xn[Nx]);
		}
		else if (Near2d[n].dir == 'Z') {
			ev3d_drawRectangle('Z', Zn[Near2d[n].id0], Xn[0], Yn[0], Xn[Nx], Yn[Ny]);
		}
	}
}

void plot3dGeom(int ev)
{
	// initialize
	ev3d_init(Width3d, Height3d);

	// new page
	ev3d_newPage();

	// mesh
	plot3dGeom_m();

	// geometry
	plot3dGeom_g();

	// feed
	plot3dGeom_f();

	// load (inductor)
	plot3dGeom_l();

	// point
	plot3dGeom_p();

	// near field
	if (runMode == 0) {
		// near1d
		plot3dGeom_n1();

		// near2d
		plot3dGeom_n2();
	}

	// title
	char str[BUFSIZ];
	ev3d_setColor(0, 0, 0);
	ev3d_drawTitle(Font3d, Title);
	sprintf(str, "No. of geometries = %zd", NGeometry);
	ev3d_drawTitle(Font3d, str);
	sprintf(str, "Nx=%d Ny=%d Nz=%d", Nx, Ny, Nz);
	ev3d_drawTitle(Font3d, str);

	// output
	if (!ev) ev3d_setAngle(Theta3d, Phi3d);
	ev3d_file(ev, (ev ? FN_geom3d_1 : FN_geom3d_0), 0);
	ev3d_output();
}
