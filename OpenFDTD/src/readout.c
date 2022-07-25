/*
readout.c

read ofd.out
*/

#include "ofd.h"
#include "ofd_prototype.h"

void readout(FILE *fp)
{
	int64_t num = 0;

	num += fread(Title,           sizeof(char),    256, fp);
	num += fread(&Nx,             sizeof(int),     1,   fp);
	num += fread(&Ny,             sizeof(int),     1,   fp);
	num += fread(&Nz,             sizeof(int),     1,   fp);
	num += fread(&NEx,            sizeof(int64_t), 1,   fp);
	num += fread(&NEy,            sizeof(int64_t), 1,   fp);
	num += fread(&NEz,            sizeof(int64_t), 1,   fp);
	num += fread(&NHx,            sizeof(int64_t), 1,   fp);
	num += fread(&NHy,            sizeof(int64_t), 1,   fp);
	num += fread(&NHz,            sizeof(int64_t), 1,   fp);
	num += fread(&NFreq1,         sizeof(int),     1,   fp);
	num += fread(&NFreq2,         sizeof(int),     1,   fp);
	num += fread(&NFeed,          sizeof(int),     1,   fp);
	num += fread(&NPoint,         sizeof(int),     1,   fp);
	num += fread(&Niter,          sizeof(int),     1,   fp);
	num += fread(&Ntime,          sizeof(int),     1,   fp);
	num += fread(&Solver.maxiter, sizeof(int),     1,   fp);
	num += fread(&Solver.nout,    sizeof(int),     1,   fp);
	num += fread(&Dt,             sizeof(double),  1,   fp);
	num += fread(&NGline,         sizeof(int),     1,   fp);

	Xn     =         (double *)malloc(sizeof(double)      * (Nx + 1));
	Yn     =         (double *)malloc(sizeof(double)      * (Ny + 1));
	Zn     =         (double *)malloc(sizeof(double)      * (Nz + 1));
	Xc     =         (double *)malloc(sizeof(double)      * Nx);
	Yc     =         (double *)malloc(sizeof(double)      * Ny);
	Zc     =         (double *)malloc(sizeof(double)      * Nz);
	Eiter  =         (double *)malloc(sizeof(double)      * Niter);
	Hiter  =         (double *)malloc(sizeof(double)      * Niter);
	VFeed  =         (double *)malloc(sizeof(double)      * NFeed  * (Solver.maxiter + 1));
	IFeed  =         (double *)malloc(sizeof(double)      * NFeed  * (Solver.maxiter + 1));
	VPoint =         (double *)malloc(sizeof(double)      * NPoint * (Solver.maxiter + 1));
	Freq1  =         (double *)malloc(sizeof(double)      * NFreq1);
	Freq2  =         (double *)malloc(sizeof(double)      * NFreq2);
	Feed   =         (feed_t *)malloc(sizeof(feed_t)      * NFeed);
	Zin    =    (d_complex_t *)malloc(sizeof(d_complex_t) * NFeed * NFreq1);
	Ref    =         (double *)malloc(sizeof(double)      * NFeed * NFreq1);
	Pin[0] =         (double *)malloc(sizeof(double)      * NFeed * NFreq2);
	Pin[1] =         (double *)malloc(sizeof(double)      * NFeed * NFreq2);
	Spara  =    (d_complex_t *)malloc(sizeof(d_complex_t) * NPoint * NFreq1);
	Gline  = (double (*)[2][3])malloc(sizeof(double)      * NGline * 2 * 3);

	num += fread(Xn,     sizeof(double),      Nx + 1,                        fp);
	num += fread(Yn,     sizeof(double),      Ny + 1,                        fp);
	num += fread(Zn,     sizeof(double),      Nz + 1,                        fp);
	num += fread(Xc,     sizeof(double),      Nx,                            fp);
	num += fread(Yc,     sizeof(double),      Ny,                            fp);
	num += fread(Zc,     sizeof(double),      Nz,                            fp);
	num += fread(Eiter,  sizeof(double),      Niter,                         fp);
	num += fread(Hiter,  sizeof(double),      Niter,                         fp);
	num += fread(VFeed,  sizeof(double),      NFeed  * (Solver.maxiter + 1), fp);
	num += fread(IFeed,  sizeof(double),      NFeed  * (Solver.maxiter + 1), fp);
	num += fread(VPoint, sizeof(double),      NPoint * (Solver.maxiter + 1), fp);
	num += fread(Freq1,  sizeof(double),      NFreq1,                        fp);
	num += fread(Freq2,  sizeof(double),      NFreq2,                        fp);
	num += fread(Feed,   sizeof(feed_t),      NFeed,                         fp);
	num += fread(Zin,    sizeof(d_complex_t), NFeed * NFreq1,                fp);
	num += fread(Ref,    sizeof(double),      NFeed * NFreq1,                fp);
	num += fread(Pin[0], sizeof(double),      NFeed * NFreq2,                fp);
	num += fread(Pin[1], sizeof(double),      NFeed * NFreq2,                fp);
	num += fread(Spara,  sizeof(d_complex_t), NPoint * NFreq1,               fp);
	num += fread(Gline,  sizeof(double),      NGline * 2 * 3,                fp);

	// alloc
	for (int ic = 0; ic < 6; ic++) {
		calcNear3d(ic, 0);
	}
/*
	cEx_r = (real_t *)malloc(NEx * NFreq2 * sizeof(real_t));
	cEx_i = (real_t *)malloc(NEx * NFreq2 * sizeof(real_t));
	cEy_r = (real_t *)malloc(NEy * NFreq2 * sizeof(real_t));
	cEy_i = (real_t *)malloc(NEy * NFreq2 * sizeof(real_t));
	cEz_r = (real_t *)malloc(NEz * NFreq2 * sizeof(real_t));
	cEz_i = (real_t *)malloc(NEz * NFreq2 * sizeof(real_t));
	cHx_r = (real_t *)malloc(NHx * NFreq2 * sizeof(real_t));
	cHx_i = (real_t *)malloc(NHx * NFreq2 * sizeof(real_t));
	cHy_r = (real_t *)malloc(NHy * NFreq2 * sizeof(real_t));
	cHy_i = (real_t *)malloc(NHy * NFreq2 * sizeof(real_t));
	cHz_r = (real_t *)malloc(NHz * NFreq2 * sizeof(real_t));
	cHz_i = (real_t *)malloc(NHz * NFreq2 * sizeof(real_t));
*/
	num += fread(cEx_r, sizeof(real_t), NEx * NFreq2, fp);
	num += fread(cEx_i, sizeof(real_t), NEx * NFreq2, fp);
	num += fread(cEy_r, sizeof(real_t), NEy * NFreq2, fp);
	num += fread(cEy_i, sizeof(real_t), NEy * NFreq2, fp);
	num += fread(cEz_r, sizeof(real_t), NEz * NFreq2, fp);
	num += fread(cEz_i, sizeof(real_t), NEz * NFreq2, fp);
	num += fread(cHx_r, sizeof(real_t), NHx * NFreq2, fp);
	num += fread(cHx_i, sizeof(real_t), NHx * NFreq2, fp);
	num += fread(cHy_r, sizeof(real_t), NHy * NFreq2, fp);
	num += fread(cHy_i, sizeof(real_t), NHy * NFreq2, fp);
	num += fread(cHz_r, sizeof(real_t), NHz * NFreq2, fp);
	num += fread(cHz_i, sizeof(real_t), NHz * NFreq2, fp);

	int64_t num0;
	size_t size = fread(&num0, sizeof(int64_t), 1, fp);
	size = size;  // suppress gcc warning

	if (num != num0) {
		fprintf(stderr, "*** invalid file length : (%zd, %zd)\n", num0, num);
	}
}
