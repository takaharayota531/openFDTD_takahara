/*
ARF Version 1.0.0 (c) 2021
Antenna optimization
Random deformation
OpenFDTD Version 2.6.1

> ./arf.exe [nthread nrepeat nloop seed1 seed2]
> mpiexec.exe -n <nprocess> ./arf.exe [nthread nrepeat nloop seed1 seed2]
*/

#define MAIN
#include "ofd.h"
#undef MAIN
#include "ofd_prototype.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {int xy, i, j, on;} segment_t;

extern void initsegment(int, int, int, int64_t *, double, int, segment_t []);
extern void setgeometry(int, int, double, double, double, int, int, int, segment_t []);
extern double getvalue(void);
extern void outputofd(const char []);
extern double urand(int, int64_t *);
extern void comm_result(int, int, double, int, segment_t []);

static void setfrequency(double, double, int, double, double, int);
static void setmisc(int, int, double, double, double, int, double, double);
static void calc(void);

extern void alloc_farfield(void); // TODO

int main(int argc, char **argv)
{
	// antenna model
	const double lx = 100e-3;
	const double ly = 100e-3;
	const int nx = 10;
	const int ny = 10;
	const double h = 10e-3;       // height
	const double epsr = 2.2;      // Er
	const double sigma = 0;       // sigma
	const int sdiv = 1;           // =1/2/3... : segment division

	// frequency
	double frequency1s = 3.0e9;
	double frequency1e = 3.0e9;
	int frequency1div = 0;
	double frequency2s = 3.0e9;
	double frequency2e = 3.0e9;
	int frequency2div = 0;

	// hyper parameter (1)
	const double ratio = 0.5;     // 0-1
	const int randtype = 2;       // =1/2/3/4

	// hyper parameters (2)
	int nthread = 1;
	int nrepeat = 16;
	int nloop = 1000;
	int64_t seed1 = 1000;
	int64_t seed2 = 500;

	// setup MPI
	int comm_size = 1;
	int comm_rank = 0;
	mpi_init(argc, argv, &comm_size, &comm_rank);
	//printf("%d %d\n", comm_size, comm_rank); fflush(stdout);

	// arguments
	if (argc > 5) {
		nthread = atoi(argv[1]);
		nrepeat = atoi(argv[2]);
		nloop = atoi(argv[3]);
		seed1 = atoi(argv[4]);
		seed2 = atoi(argv[5]);
	}
	// nrepeat % comm_size = 0
	nrepeat = comm_size * MAX(nrepeat / comm_size, 1);

	// error check
	assert((nx % 2 == 0) && (ny % 2 == 0));
	assert((randtype >= 1) && (randtype <= 4));
	assert(nthread > 0);
	assert(nrepeat > 0);
	assert(nloop > 0);
	assert(seed1 > 0);
	assert(seed2 > 0);

	// set number of threads
#ifdef _OPENMP
	omp_set_num_threads(nthread);
#else
	nthread = 1;
#endif

	// cpu
	double cpu0 = comm_cputime();

	// set frequency
	setfrequency(frequency1s, frequency1e, frequency1div, frequency2s, frequency2e, frequency2div);

	// set misc.
	setmisc(nx, ny, lx, ly, h, sdiv, epsr, sigma);

	// setup
	setupSize();
	setupSizeNear();
	memalloc1();
	memalloc2();
	memalloc3();
	calcNear2d(0);
	for (int ic = 0; ic < 6; ic++) {
		calcNear3d(ic, 0);
	}
	alloc_farfield();

	// alloc (segment)
	const int nseg = (nx + 0) * (ny + 1) + (nx + 1) * (ny + 0);
	segment_t *seg     = (segment_t *)malloc(nseg * sizeof(segment_t));
	segment_t *segbest = (segment_t *)malloc(nseg * sizeof(segment_t));

	// geometry size
	const int lgeometry = nseg + 3;   // 3 : substrate + ground + feed
	Geometry = (geometry_t *)malloc(lgeometry * sizeof(geometry_t));

	// monitor
	if (!comm_rank) {
		printf("%s\n", "ARF Version 1.0.0");
		printf("nprocess=%d nthread=%d\n", comm_size, nthread);
		printf("nrepeat=%d nloop=%d seed(%d)=%zd,%zd segment=%d*%d nfrequency=%d,%d\n", nrepeat, nloop, randtype, seed1, seed2, nseg, sdiv, NFreq1, NFreq2);
		fflush(stdout);
	}

	// repeat
	double fminbest = 1e10;
	//for (int irepeat = 0; irepeat < nrepeat; irepeat++) {
	for (int irepeat = comm_rank; irepeat < nrepeat; irepeat += comm_size) {
		//printf("%d %d %d %d\n", comm_size, comm_rank, nrepeat, irepeat); fflush(stdout);

		// initialize random number
		int64_t seed = seed1 + (irepeat * seed2);
		if (randtype == 1) {
			srand((unsigned int)seed);
			rand();
			rand();
		}

		// set initial geometry
		initsegment(nx, ny, randtype, &seed, ratio, nseg, seg);
		setgeometry(nx, ny, lx, ly, h, sdiv, lgeometry, nseg, seg);
		//outputofd("arf0.ofd");

		// initial calculation
		calc();
		double fmin = getvalue();
		//printf("%d 0 %f %zd\n", irepeat, fmin, NGeometry);

		// loop
		int smin = nseg;
		int nmin = 0;
		for (int loop = 0; loop < nloop; loop++) {
			// random deformation
			const double rnd = urand(randtype, &seed);
			const int irnd = (int)(rnd * nseg);
			//printf("%d %d %d\n", loop, nseg, irnd);
			assert((irnd >= 0) && (irnd < nseg));
			seg[irnd].on = !seg[irnd].on;
			setgeometry(nx, ny, lx, ly, h, sdiv, lgeometry, nseg, seg);

			// get value
			calc();
			double f = getvalue();
			//printf("%d %d %f\n", irepeat, loop + 1, f);

			// judge
			if (f < fmin) {
				fmin = f;
				nmin = loop;
				smin = (int)NGeometry;
				//printf("%d %d %f\n", irepeat, loop + 1, f);
			}
			else {
				seg[irnd].on = !seg[irnd].on;
			}
			//printf("%d %d %f %d\n", irepeat, loop + 1, fmin, NGeometry);
		}

		// judge
		printf("%3d %3d%s %f (%d, %d)\n", comm_rank, irepeat + 1, (fmin < fminbest ? "*" : " "), fmin, nmin + 1, smin); fflush(stdout);
		if (fmin < fminbest) {
			fminbest = fmin;
			memcpy(segbest, seg, nseg * sizeof(segment_t));
		}
	}

	// communicate result
	if (comm_size > 1) {
		comm_result(comm_size, comm_rank, fminbest, nseg, segbest);
	}
	else {
		printf("fmin = %f\n", fminbest); fflush(stdout);
	}

	// output best ofd
	if (!comm_rank) {
		setgeometry(nx, ny, lx, ly, h, sdiv, lgeometry, nseg, segbest);
		outputofd("arf.ofd");
		printf("%s\n", "output : arf.ofd");
	}

	// cpu time
	double cpu1 = comm_cputime();
	if (!comm_rank) {
		printf("cpu time = %.3f [sec]\n", cpu1 - cpu0);
	}

	// close MPI
	mpi_close();

	return 0;
}


// set frequency
static void setfrequency(double frequency1s, double frequency1e, int frequency1div, double frequency2s, double frequency2e, int frequency2div)
{
	NFreq1 = frequency1div + 1;
	Freq1 = (double *)malloc(NFreq1 * sizeof(double));

	const double dfreq1 = (frequency1div > 0) ? (frequency1e - frequency1s) / frequency1div : 0;
	for (int nfreq = 0; nfreq <= frequency1div; nfreq++) {
		Freq1[nfreq] = frequency1s + (nfreq * dfreq1);
	}

	NFreq2 = frequency2div + 1;
	Freq2 = (double *)malloc(NFreq2 * sizeof(double));

	const double dfreq2 = (frequency2div > 0) ? (frequency2e - frequency2s) / frequency2div : 0;
	for (int nfreq = 0; nfreq <= frequency2div; nfreq++) {
		Freq2[nfreq] = frequency2s + (nfreq * dfreq2);
	}
}


// set misc. data
static void setmisc(int nx, int ny, double lx, double ly, double h, int sdiv, double epsr, double sigma)
{
	// run mode : solver only
	runMode = 1;

	// === mesh ===

	const int nxr = 1;
	const int nyr = 1;
	const int nzr = 1;
	double *xr = (double *)malloc((nxr + 1) * sizeof(double));
	double *yr = (double *)malloc((nyr + 1) * sizeof(double));
	double *zr = (double *)malloc((nzr + 1) * sizeof(double));
	int *dxr = (int *)malloc(nxr * sizeof(int));
	int *dyr = (int *)malloc(nyr * sizeof(int));
	int *dzr = (int *)malloc(nzr * sizeof(int));

	const int margin = 8;
	const int zmargin = 10;
	const double d = (lx + ly) / (nx + ny) / sdiv;
	xr[0] = -lx / 2 - (margin * d);
	xr[1] = -xr[0];
	yr[0] = -ly / 2 - (margin * d);
	yr[1] = -yr[0];
	zr[0] = -2 * d;
	zr[1] = h + (zmargin * d);
	dxr[0] = NINT(xr[1] - xr[0], d);
	dyr[0] = NINT(yr[1] - yr[0], d);
	dzr[0] = NINT(zr[1] - zr[0], d);

	// number of cells
	setup_cells(nxr, nyr, nzr, dxr, dyr, dzr);

	// node
	setup_node(nxr, nyr, nzr, xr, yr, zr, dxr, dyr, dzr);

	// cell center
	setup_center();

	// === material ===

	NMaterial = 3;
	Material = (material_t *)malloc(NMaterial * sizeof(material_t));
	for (int64_t m = 0; m < NMaterial; m++) {
		Material[m].epsr = (m == 2) ? epsr : 1;
		Material[m].esgm = (m == 2) ? sigma : 0;
		Material[m].amur = 1;
		Material[m].msgm = 0;
	}

	// === feed ===

	NFeed = 1;
	Feed  = (feed_t *)malloc(NFeed * sizeof(feed_t));
	double *xfeed = (double *)malloc(NFeed * sizeof(double));
	double *yfeed = (double *)malloc(NFeed * sizeof(double));
	double *zfeed = (double *)malloc(NFeed * sizeof(double));

	const int ifeed = 0;
	Feed[ifeed].dir   = 'Z';
	Feed[ifeed].volt  = 1;
	Feed[ifeed].delay = 0;
	Feed[ifeed].z0    = 50;
	xfeed[ifeed]      = 0;
	yfeed[ifeed]      = 0;
	zfeed[ifeed]      = 0.1 * h;
	rFeed = 10;

	if (NFeed) {
		setup_feed(xfeed, yfeed, zfeed);
	}

	// plane wave
	IPlanewave = 0;
/*
	if (IPlanewave) {
		setup_planewave();
	}
*/
	// point
	NPoint = 0;
/*
	if (NPoint) {
		setup_point(xpoint, ypoint, zpoint, strprop);
	}
*/
	// load
	NInductor = 0;
/*
	if (nload > 0) {
		setup_load(nload, dload, xload, yload, zload, cload, pload, array_inc);
	}
*/
	// near1d
/*
	setup_near1d();
*/
	// near2d
	setup_near2d();

	// fit geometry without thickness
	fitgeometry();

	// === misc ===

	Solver.maxiter = 2000 * sdiv;
	Solver.nout = 50 * sdiv;
	Solver.converg = 0e-3;
	iABC = 0;

	const double f0 = (Freq1[0] + Freq1[NFreq1 - 1]) / 2;
	Tw = 0.3 / f0;  // 1.27 -> 0.3
}


// calculation
static void calc(void)
{
	const int io = 0;

	// setup
	setup();

	// ofd.log
	FILE *fp_log = NULL;
	if (io) {
		fp_log = fopen("ofd.log", "w");
	}

	// solve
	if (!io) {
		solve(io, 0, NULL);
	}
	else {
		monitor2(fp_log, 0);
		solve(io, 0, fp_log);
	}

	// input imepedanece
	if ((NFeed > 0) && (NFreq1 > 0)) {
		zfeed();
		if (io) {
			outputZfeed(fp_log);
		}
	}

	// S-parameters
	if ((NFeed > 0) && (NPoint > 0) && (NFreq1 > 0)) {
		spara();
	}

	// setup far field
	if (NFreq2 > 0) {
		calcNear2d(1);
		setup_farfield();
	}
}
