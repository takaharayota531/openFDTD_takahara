/*
farfield.c

far field
*/

#include "ofd.h"
#include "complex.h"

typedef struct {
	double nx, ny, nz;
	double x, y, z;
	double ds;
} surface_t;
surface_t *surface;
d_complex_t **surface_ex, **surface_ey, **surface_ez;
d_complex_t **surface_hx, **surface_hy, **surface_hz;
int64_t nsurface;


void farfield(int ifreq, double theta, double phi, double ffctr, d_complex_t *etheta, d_complex_t *ephi)
{
	// wave number

	const double kwave = (2 * PI * Freq2[ifreq]) / C;

	// unit vector in r, theta, phi

	double r1[3], t1[3], p1[3];

	const double cos_t = cos(theta * DTOR);
	const double sin_t = sin(theta * DTOR);
	const double cos_p = cos(phi   * DTOR);
	const double sin_p = sin(phi   * DTOR);

	r1[0] = +sin_t * cos_p;
	r1[1] = +sin_t * sin_p;
	r1[2] = +cos_t;

	t1[0] = +cos_t * cos_p;
	t1[1] = +cos_t * sin_p;
	t1[2] = -sin_t;

	p1[0] = -sin_p;
	p1[1] = +cos_p;
	p1[2] = 0;

	d_complex_t plx = d_complex(0, 0);
	d_complex_t ply = d_complex(0, 0);
	d_complex_t plz = d_complex(0, 0);
	d_complex_t pnx = d_complex(0, 0);
	d_complex_t pny = d_complex(0, 0);
	d_complex_t pnz = d_complex(0, 0);

	d_complex_t *ex = surface_ex[ifreq];
	d_complex_t *ey = surface_ey[ifreq];
	d_complex_t *ez = surface_ez[ifreq];
	d_complex_t *hx = surface_hx[ifreq];
	d_complex_t *hy = surface_hy[ifreq];
	d_complex_t *hz = surface_hz[ifreq];

	for (int n = 0; n < nsurface; n++) {
		surface_t *p = &surface[n];

		// Z0 * J = n X (Z0 * H)
		const d_complex_t cjx = d_sub(d_rmul(+p->ny, hz[n]), d_rmul(+p->nz, hy[n]));
		const d_complex_t cjy = d_sub(d_rmul(+p->nz, hx[n]), d_rmul(+p->nx, hz[n]));
		const d_complex_t cjz = d_sub(d_rmul(+p->nx, hy[n]), d_rmul(+p->ny, hx[n]));

		// M = -n X E
		const d_complex_t cmx = d_sub(d_rmul(-p->ny, ez[n]), d_rmul(-p->nz, ey[n]));
		const d_complex_t cmy = d_sub(d_rmul(-p->nz, ex[n]), d_rmul(-p->nx, ez[n]));
		const d_complex_t cmz = d_sub(d_rmul(-p->nx, ey[n]), d_rmul(-p->ny, ex[n]));

		// exp(jkr * r) * dS
		const double rr = (r1[0] * p->x) + (r1[1] * p->y) + (r1[2] * p->z);
		const d_complex_t expds = d_rmul(p->ds, d_exp(kwave * rr));

		// L += M * exp(jkr * r) * dS
		plx = d_add(plx, d_mul(cmx, expds));
		ply = d_add(ply, d_mul(cmy, expds));
		plz = d_add(plz, d_mul(cmz, expds));

		// Z0 * N += (Z0 * J) * exp(jkr * r) * dS
		pnx = d_add(pnx, d_mul(cjx, expds));
		pny = d_add(pny, d_mul(cjy, expds));
		pnz = d_add(pnz, d_mul(cjz, expds));
	}

	// Z0 * N-theta, Z0 * N-phi
	const d_complex_t pnt = d_add3(d_rmul(t1[0], pnx), d_rmul(t1[1], pny), d_rmul(t1[2], pnz));
	const d_complex_t pnp = d_add3(d_rmul(p1[0], pnx), d_rmul(p1[1], pny), d_rmul(p1[2], pnz));

	// L-theta, L-phi
	const d_complex_t plt = d_add3(d_rmul(t1[0], plx), d_rmul(t1[1], ply), d_rmul(t1[2], plz));
	const d_complex_t plp = d_add3(d_rmul(p1[0], plx), d_rmul(p1[1], ply), d_rmul(p1[2], plz));

	// F-theta, F-phi
	*etheta = d_rmul(ffctr, d_add(pnt, plp));
	*ephi   = d_rmul(ffctr, d_sub(pnp, plt));
}


// alloc far field array
void alloc_farfield(void)
{
	assert((Nx > 0) && (Ny > 0) && (Nz > 0) && (NFreq2 > 0));

	nsurface = 2 * ((Nx * Ny) + (Ny * Nz) + (Nz * Nx));
	//printf("%zd %d\n", nsurface, NFreq2);

	surface = (surface_t *)malloc(nsurface * sizeof(surface_t));

	surface_ex = (d_complex_t **)malloc(NFreq2 * sizeof(d_complex_t *));
	surface_ey = (d_complex_t **)malloc(NFreq2 * sizeof(d_complex_t *));
	surface_ez = (d_complex_t **)malloc(NFreq2 * sizeof(d_complex_t *));
	surface_hx = (d_complex_t **)malloc(NFreq2 * sizeof(d_complex_t *));
	surface_hy = (d_complex_t **)malloc(NFreq2 * sizeof(d_complex_t *));
	surface_hz = (d_complex_t **)malloc(NFreq2 * sizeof(d_complex_t *));
	for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
		surface_ex[ifreq] = (d_complex_t *)malloc(nsurface * sizeof(d_complex_t));
		surface_ey[ifreq] = (d_complex_t *)malloc(nsurface * sizeof(d_complex_t));
		surface_ez[ifreq] = (d_complex_t *)malloc(nsurface * sizeof(d_complex_t));
		surface_hx[ifreq] = (d_complex_t *)malloc(nsurface * sizeof(d_complex_t));
		surface_hy[ifreq] = (d_complex_t *)malloc(nsurface * sizeof(d_complex_t));
		surface_hz[ifreq] = (d_complex_t *)malloc(nsurface * sizeof(d_complex_t));
	}
}


// setup surface E and H
void setup_farfield(void)
{
	// skip user defined near field
	int isum = 0;
	for (int n = 0; n < NNear2d - 6; n++) {
		int num = 0;
		if      (Near2d[n].dir == 'X') {
			num = (Ny + 1) * (Nz + 1);
		}
		else if (Near2d[n].dir == 'Y') {
			num = (Nz + 1) * (Nx + 1);
		}
		else if (Near2d[n].dir == 'Z') {
			num = (Nx + 1) * (Ny + 1);
		}
		isum += num;
	}
	//printf("%d %d\n", NNear2d, isum);

	int64_t n = 0;

	// X surface
	for (int side = 0; side < 2; side++) {
		const int i = (side == 0) ? 0 : Nx;
		for (int j = 0; j < Ny; j++) {
		for (int k = 0; k < Nz; k++) {
			for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
				const int64_t id = (NFreq2 * isum) + (ifreq * (Ny + 1) * (Nz + 1));
				const int64_t id00 = id + (j + 0) * (Nz + 1) + (k + 0);
				const int64_t id01 = id + (j + 0) * (Nz + 1) + (k + 1);
				const int64_t id10 = id + (j + 1) * (Nz + 1) + (k + 0);
				const int64_t id11 = id + (j + 1) * (Nz + 1) + (k + 1);
				surface_ex[ifreq][n] = d_complex(0, 0);
				surface_ey[ifreq][n] = d_rmul(0.25, d_add4(Near2dEy[id00],
				                                           Near2dEy[id01],
				                                           Near2dEy[id10],
				                                           Near2dEy[id11]));
				surface_ez[ifreq][n] = d_rmul(0.25, d_add4(Near2dEz[id00],
				                                           Near2dEz[id01],
				                                           Near2dEz[id10],
				                                           Near2dEz[id11]));
				surface_hx[ifreq][n] = d_complex(0, 0);
				surface_hy[ifreq][n] = d_rmul(0.25, d_add4(Near2dHy[id00],
				                                           Near2dHy[id01],
				                                           Near2dHy[id10],
				                                           Near2dHy[id11]));
				surface_hz[ifreq][n] = d_rmul(0.25, d_add4(Near2dHz[id00],
				                                           Near2dHz[id01],
				                                           Near2dHz[id10],
				                                           Near2dHz[id11]));
			}
			surface[n].nx = (side == 0) ? -1 : +1;
			surface[n].ny = 0;
			surface[n].nz = 0;
			surface[n].x = Xn[i];
			surface[n].y = Yc[j];
			surface[n].z = Zc[k];
			surface[n].ds = (Yn[j + 1] - Yn[j]) * (Zn[k + 1] - Zn[k]);
			if (PBCx) surface[n].ds = 0;  // X PBC -> skip X boundaries
			n++;
		}
		}
		isum += (Ny + 1) * (Nz + 1);
	}

	// Y surface
	for (int side = 0; side < 2; side++) {
		const int j = (side == 0) ? 0 : Ny;
		for (int k = 0; k < Nz; k++) {
		for (int i = 0; i < Nx; i++) {
			for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
				const int64_t id = (NFreq2 * isum) + (ifreq * (Nz + 1) * (Nx + 1));  // i -> k !!
				const int64_t id00 = id + (i + 0) * (Nz + 1) + (k + 0);
				const int64_t id01 = id + (i + 0) * (Nz + 1) + (k + 1);
				const int64_t id10 = id + (i + 1) * (Nz + 1) + (k + 0);
				const int64_t id11 = id + (i + 1) * (Nz + 1) + (k + 1);
				surface_ex[ifreq][n] = d_rmul(0.25, d_add4(Near2dEx[id00],
				                                           Near2dEx[id01],
				                                           Near2dEx[id10],
				                                           Near2dEx[id11]));
				surface_ey[ifreq][n] = d_complex(0, 0);
				surface_ez[ifreq][n] = d_rmul(0.25, d_add4(Near2dEz[id00],
				                                           Near2dEz[id01],
				                                           Near2dEz[id10],
				                                           Near2dEz[id11]));
				surface_hx[ifreq][n] = d_rmul(0.25, d_add4(Near2dHx[id00],
				                                           Near2dHx[id01],
				                                           Near2dHx[id10],
				                                           Near2dHx[id11]));
				surface_hy[ifreq][n] = d_complex(0, 0);
				surface_hz[ifreq][n] = d_rmul(0.25, d_add4(Near2dHz[id00],
				                                           Near2dHz[id01],
				                                           Near2dHz[id10],
				                                           Near2dHz[id11]));
			}
			surface[n].nx = 0;
			surface[n].ny = (side == 0) ? -1 : +1;
			surface[n].nz = 0;
			surface[n].x = Xc[i];
			surface[n].y = Yn[j];
			surface[n].z = Zc[k];
			surface[n].ds = (Zn[k + 1] - Zn[k]) * (Xn[i + 1] - Xn[i]);
			if (PBCy) surface[n].ds = 0;  // Y PBC -> skip Y boundaries
			n++;
		}
		}
		isum += (Nz + 1) * (Nx + 1);
	}

	// Z surface
	for (int side = 0; side < 2; side++) {
		const int k = (side == 0) ? 0 : Nz;
		for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
				const int64_t id = (NFreq2 * isum) + (ifreq * (Nx + 1) * (Ny + 1));
				const int64_t id00 = id + (i + 0) * (Ny + 1) + (j + 0);
				const int64_t id01 = id + (i + 0) * (Ny + 1) + (j + 1);
				const int64_t id10 = id + (i + 1) * (Ny + 1) + (j + 0);
				const int64_t id11 = id + (i + 1) * (Ny + 1) + (j + 1);
				surface_ex[ifreq][n] = d_rmul(0.25, d_add4(Near2dEx[id00],
				                                           Near2dEx[id01],
				                                           Near2dEx[id10],
				                                           Near2dEx[id11]));
				surface_ey[ifreq][n] = d_rmul(0.25, d_add4(Near2dEy[id00],
				                                           Near2dEy[id01],
				                                           Near2dEy[id10],
				                                           Near2dEy[id11]));
				surface_ez[ifreq][n] = d_complex(0, 0);
				surface_hx[ifreq][n] = d_rmul(0.25, d_add4(Near2dHx[id00],
				                                           Near2dHx[id01],
				                                           Near2dHx[id10],
				                                           Near2dHx[id11]));
				surface_hy[ifreq][n] = d_rmul(0.25, d_add4(Near2dHy[id00],
				                                           Near2dHy[id01],
				                                           Near2dHy[id10],
				                                           Near2dHy[id11]));
				surface_hz[ifreq][n] = d_complex(0, 0);
			}
			surface[n].nx = 0;
			surface[n].ny = 0;
			surface[n].nz = (side == 0) ? -1 : +1;
			surface[n].x = Xc[i];
			surface[n].y = Yc[j];
			surface[n].z = Zn[k];
			surface[n].ds = (Xn[i + 1] - Xn[i]) * (Yn[j + 1] - Yn[j]);
			if (PBCz) surface[n].ds = 0;  // Z PBC -> skip Z boundaries
			n++;
		}
		}
		isum += (Nx + 1) * (Ny + 1);
	}

	//assert(NFreq2 * isum * sizeof(d_complex_t) == Near2d_size);
	//assert(n == nsurface);
}


// far field components
void farComponent(d_complex_t etheta, d_complex_t ephi, double e[])
{
	// abs
	e[0] = sqrt(d_norm(etheta) + d_norm(ephi));

	// theta/phi
	e[1] = d_abs(etheta);
	e[2] = d_abs(ephi);

	// major/minor
	double tmp = d_abs(d_add(d_mul(etheta, etheta), d_mul(ephi, ephi)));
	e[3] = sqrt((d_norm(etheta) + d_norm(ephi) + tmp) / 2);
	e[4] = sqrt((d_norm(etheta) + d_norm(ephi) - tmp) / 2);

	// RHCP/LHCP
	e[5] = d_abs(d_add(etheta, d_mul(d_complex(0, 1), ephi))) / sqrt(2);
	e[6] = d_abs(d_sub(etheta, d_mul(d_complex(0, 1), ephi))) / sqrt(2);
}


// far field factor
double farfactor(int ifreq)
{
	double ffctr = 0;

	const double kwave = (2 * PI * Freq2[ifreq]) / C;
	if (NFeed) {
		double sum = 0;
		for (int ifeed = 0; ifeed < NFeed; ifeed++) {
			sum += 0.5 * Pin[MatchingLoss ? 1 : 0][(ifeed * NFreq2) + ifreq];
		}
		ffctr = kwave / sqrt(8 * PI * ETA0 * sum);
	}
	else if (IPlanewave) {
		const double einc = 1;
		ffctr = kwave / (einc * sqrt(4 * PI));
	}

	return ffctr;
}
