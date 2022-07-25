/*
writeout.c

write ofd.out
*/

#include "ofd.h"

void writeout(FILE *fp)
{
	int64_t num = 0;

	num += fwrite(Title,           sizeof(char),        256,                           fp);
	num += fwrite(&Nx,             sizeof(int),         1,                             fp);
	num += fwrite(&Ny,             sizeof(int),         1,                             fp);
	num += fwrite(&Nz,             sizeof(int),         1,                             fp);
	num += fwrite(&NEx,            sizeof(int64_t),     1,                             fp);
	num += fwrite(&NEy,            sizeof(int64_t),     1,                             fp);
	num += fwrite(&NEz,            sizeof(int64_t),     1,                             fp);
	num += fwrite(&NHx,            sizeof(int64_t),     1,                             fp);
	num += fwrite(&NHy,            sizeof(int64_t),     1,                             fp);
	num += fwrite(&NHz,            sizeof(int64_t),     1,                             fp);
	num += fwrite(&NFreq1,         sizeof(int),         1,                             fp);
	num += fwrite(&NFreq2,         sizeof(int),         1,                             fp);
	num += fwrite(&NFeed,          sizeof(int),         1,                             fp);
	num += fwrite(&NPoint,         sizeof(int),         1,                             fp);
	num += fwrite(&Niter,          sizeof(int),         1,                             fp);
	num += fwrite(&Ntime,          sizeof(int),         1,                             fp);
	num += fwrite(&Solver.maxiter, sizeof(int),         1,                             fp);
	num += fwrite(&Solver.nout,    sizeof(int),         1,                             fp);
	num += fwrite(&Dt,             sizeof(double),      1,                             fp);
	num += fwrite(&NGline,         sizeof(int),         1,                             fp);

	num += fwrite(Xn,              sizeof(double),      Nx + 1,                        fp);
	num += fwrite(Yn,              sizeof(double),      Ny + 1,                        fp);
	num += fwrite(Zn,              sizeof(double),      Nz + 1,                        fp);
	num += fwrite(Xc,              sizeof(double),      Nx,                            fp);
	num += fwrite(Yc,              sizeof(double),      Ny,                            fp);
	num += fwrite(Zc,              sizeof(double),      Nz,                            fp);
	num += fwrite(Eiter,           sizeof(double),      Niter,                         fp);
	num += fwrite(Hiter,           sizeof(double),      Niter,                         fp);
	num += fwrite(VFeed,           sizeof(double),      NFeed  * (Solver.maxiter + 1), fp);
	num += fwrite(IFeed,           sizeof(double),      NFeed  * (Solver.maxiter + 1), fp);
	num += fwrite(VPoint,          sizeof(double),      NPoint * (Solver.maxiter + 1), fp);
	num += fwrite(Freq1,           sizeof(double),      NFreq1,                        fp);
	num += fwrite(Freq2,           sizeof(double),      NFreq2,                        fp);
	num += fwrite(Feed,            sizeof(feed_t),      NFeed,                         fp);
	num += fwrite(Zin,             sizeof(d_complex_t), NFeed * NFreq1,                fp);
	num += fwrite(Ref,             sizeof(double),      NFeed * NFreq1,                fp);
	num += fwrite(Pin[0],          sizeof(double),      NFeed * NFreq2,                fp);
	num += fwrite(Pin[1],          sizeof(double),      NFeed * NFreq2,                fp);
	num += fwrite(Spara,           sizeof(d_complex_t), NPoint * NFreq1,               fp);
	num += fwrite(Gline,           sizeof(double),      NGline * 2 * 3,                fp);

	num += fwrite(cEx_r,           sizeof(real_t),      NEx * NFreq2,                  fp);
	num += fwrite(cEx_i,           sizeof(real_t),      NEx * NFreq2,                  fp);
	num += fwrite(cEy_r,           sizeof(real_t),      NEy * NFreq2,                  fp);
	num += fwrite(cEy_i,           sizeof(real_t),      NEy * NFreq2,                  fp);
	num += fwrite(cEz_r,           sizeof(real_t),      NEz * NFreq2,                  fp);
	num += fwrite(cEz_i,           sizeof(real_t),      NEz * NFreq2,                  fp);
	num += fwrite(cHx_r,           sizeof(real_t),      NHx * NFreq2,                  fp);
	num += fwrite(cHx_i,           sizeof(real_t),      NHx * NFreq2,                  fp);
	num += fwrite(cHy_r,           sizeof(real_t),      NHy * NFreq2,                  fp);
	num += fwrite(cHy_i,           sizeof(real_t),      NHy * NFreq2,                  fp);
	num += fwrite(cHz_r,           sizeof(real_t),      NHz * NFreq2,                  fp);
	num += fwrite(cHz_i,           sizeof(real_t),      NHz * NFreq2,                  fp);

	fwrite(&num, sizeof(int64_t), 1, fp);
}
