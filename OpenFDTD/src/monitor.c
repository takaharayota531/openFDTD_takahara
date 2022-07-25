/*
monitor.c
*/

#include "ofd.h"

static int memory_size(void);
static int output_size(void);

// title or message
static void monitor1_(FILE *fp, const char msg[])
{
	fprintf(fp, "%s\n", msg);
	fflush(fp);
}

// condition
static void monitor2_(FILE *fp, int gpu)
{
	const int nsurface = (IPlanewave || IFar0d || NFar1d || NFar2d) ? 6 : 0;

	time_t now;
	time(&now);

	fprintf(fp, "%s", ctime(&now));
	fprintf(fp, "Title = %s\n", Title);
	fprintf(fp, "Source = %s\n", (NFeed ? "feed" : "plane wave"));
	fprintf(fp, "Cells = %d x %d x %d = %lld\n", Nx, Ny, Nz, ((long long int)Nx * Ny * Nz));
	fprintf(fp, "No. of Materials  = %zd\n", NMaterial);
	fprintf(fp, "No. of Geometries = %zd\n", NGeometry);
	if (NFeed) fprintf(fp, "No. of Feeds      = %d\n", NFeed);
	fprintf(fp, "No. of Points     = %d\n", NPoint);
	fprintf(fp, "No. of Freq.s (1) = %d\n", NFreq1);
	fprintf(fp, "No. of Freq.s (2) = %d\n", NFreq2);
	if (runMode == 0) fprintf(fp, "No. of Near1d.s   = %d\n", NNear1d);
	if (runMode == 0) fprintf(fp, "No. of Near2d.s   = %d + %d = %d\n", NNear2d - nsurface, nsurface, NNear2d);
	if (runMode == 0) fprintf(fp, "No. of Far1d.s    = %d\n", NFar1d);
	if (runMode == 0) fprintf(fp, "Far2d (0/1)       = %d\n", NFar2d);
	fprintf(fp, "%s Memory size   = %d [MB]\n", (gpu ? "GPU" : "CPU"), memory_size());
	if (runMode == 1) fprintf(fp, "Output filesize   = %d [MB]\n", output_size());
	if (iABC == 0) fprintf(fp, "ABC = Mur-1st\n");
	if (iABC == 1) fprintf(fp, "ABC = PML (L=%d, M=%.2f, R0=%.2e)\n", cPML.l, cPML.m, cPML.r0);
	if (PBCx || PBCy || PBCz) fprintf(fp, "PBC : %s%s%s\n", (PBCx ? "X" : ""), (PBCy ? "Y" : ""), (PBCz ? "Z" : ""));
	fprintf(fp, "Dt[sec] = %.4e, Tw[sec] = %.4e, Tw/Dt = %.3f\n", Dt, Tw, Tw / Dt);
	fprintf(fp, "Iterations = %d, Convergence = %.3e\n", Solver.maxiter, Solver.converg);
	fprintf(fp, "=== iteration start ===\n");
	fprintf(fp, "   step   <E>      <H>\n");
	fflush(fp);
}

// output files
static void monitor3_(FILE *fp, int ev)
{
	fprintf(fp, "=== output files ===\n");
	fprintf(fp, "%s, %s", FN_log, (ev ? FN_geom3d_1 : FN_geom3d_0));
	if      (runMode == 0) {
		fprintf(fp, ", %s, %s", (ev ? FN_ev2d_1 : FN_ev2d_0), (ev ? FN_ev3d_1 : FN_ev3d_0));
	}
	else if (runMode == 1) {
		fprintf(fp, ", %s", FN_out);
	}
	fprintf(fp, "\n");
	fflush(fp);
}

// cpu time
static void monitor4_(FILE *fp, const double cpu[])
{
	time_t now;
	time(&now);

	fprintf(fp, "%s\n", "=== cpu time [sec] ===");
	fprintf(fp, "  part-1 : %11.3f\n", cpu[2] - cpu[1]);
	fprintf(fp, "  part-2 : %11.3f\n", (cpu[1] - cpu[0]) + (cpu[3] - cpu[2]));
	fprintf(fp, "  %s\n", "--------------------");
	fprintf(fp, "  total  : %11.3f\n", cpu[3] - cpu[0]);
	fprintf(fp, "%s\n", "=== normal end ===");
	fprintf(fp, "%s", ctime(&now));
	fflush(fp);
}

void monitor1(FILE *fp, const char msg[])
{
	monitor1_(fp,     msg);
	monitor1_(stdout, msg);
}

void monitor2(FILE *fp, int gpu)
{
	monitor2_(fp,     gpu);
	monitor2_(stdout, gpu);
}

void monitor3(FILE *fp, int ev)
{
	monitor3_(fp,     ev);
	monitor3_(stdout, ev);
}

void monitor4(FILE *fp, const double cpu[])
{
	monitor4_(fp,     cpu);
	monitor4_(stdout, cpu);
}

// memory size [MB]
static int memory_size(void)
{
	int64_t mem
		= 6 * NN * sizeof(real_t)          // Ex, Ey, Ez, Hx, Hy, Hz
		+ 6 * NN * sizeof(id_t)            // iEx, iEy, iEz, iHx, iHy, iHz
#ifdef _VECTOR
		+ (IPlanewave ? 24 : 12) * NN * sizeof(real_t)  // K1Ex, K2Ex, (K3Ex, K4Ex,) ...
#endif
		+ 8 * NMaterial * sizeof(real_t)   // C1, C2, C3, C4, D1, D2, D3, D4
		+ NMaterial * sizeof(material_t)   // Material
		+ NGeometry * sizeof(geometry_t);  // Geometry

	if      (runMode == 0) {
		// near2d
		int sum = 0;
		for (int m = 0; m < NNear2d; m++) {
			if      (Near2d[m].dir == 'X') {
				sum += Ny * Nz;
			}
			else if (Near2d[m].dir == 'Y') {
				sum += Nz * Nx;
			}
			else if (Near2d[m].dir == 'Z') {
				sum += Nx * Ny;
			}
		}
		mem += 6 * sum * NFreq2 * sizeof(d_complex_t);
	}
	else if (runMode == 1) {
		// near3d
		mem += NFreq2 * 2 * (NEx + NEy + NEz + NHx + NHy + NHz) * sizeof(real_t);  // cEx, cEy, cEz, cHx, cHy, cHz
	}

	// ABC
	if      (iABC == 0) {
		mem += (numMurHy + numMurHy + numMurHz)
		     * sizeof(mur_t);
	}
	else if (iABC == 1) {
		mem += (numPmlEx + numPmlEy + numPmlEz
		      + numPmlHx + numPmlHy + numPmlHz)
		     * (sizeof(pml_t) + 2 * sizeof(real_t));
	}

	return (int)(mem / 1024 / 1024) + 1;
}

// output filesize [MB]
static int output_size(void)
{
	return (int)(NFreq2 * 2 * (NEx + NEy + NEz + NHx + NHy + NHz) * sizeof(real_t) / 1024 / 1024) + 1;
}
