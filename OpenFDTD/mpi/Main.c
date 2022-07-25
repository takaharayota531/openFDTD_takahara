/*
OpenFDTD Version 2.7.5 (CPU + MPI + OpenMP)

Copyright (C) EEM Inc. 2014-2022
URL : http://www.e-em.co.jp/OpenFDTD/
Mail : info@e-em.co.jp
*/

#define MAIN
#include "ofd.h"
#include "ofd_mpi.h"
#undef MAIN

#include "ofd_prototype.h"

static void args(int, char *[], int *, int *, int *, int *, char []);

int main(int argc, char *argv[])
{
	const char prog[] = "(CPU+MPI+OpenMP)";
	const char errfmt[] = "*** file %s open error.\n";
	char str[BUFSIZ];
	double cpu[4];
	int ierr = 0;
	FILE *fp_in = NULL, *fp_out = NULL, *fp_log = NULL;

	// initialize MPI
	mpi_init(argc, argv);
	const int io = !commRank;

	// arguments
	runMode = 0;
	int nthread = 1;
	int ev = 0;
	int geom = 0;
	int prompt = 0;
	char fn_in[BUFSIZ] = "";
	args(argc, argv, &nthread, &ev, &geom, &prompt, fn_in);
	//printf("%d %d %d %d %d %s\n", runMode, nthread, ev, geom, prompt, fn_in); fflush(stdout);

	// set number of threads
#ifdef _OPENMP
	omp_set_num_threads(nthread);
#endif

	// input
	if (io) {
		if ((fp_in = fopen(fn_in, "r")) == NULL) {
			printf(errfmt, fn_in);
			fflush(stdout);
			ierr = 1;
		}
		if (!ierr) {
			ierr = input(fp_in, (!geom && (runMode < 2)));
			fclose(fp_in);
		}
	}
	comm_check(ierr, 0, prompt);

	// cpu time
	cpu[0] = comm_cputime();

	// === solver ===

	if ((runMode == 0) || (runMode == 1)) {
		// logo
		if (io) {
			sprintf(str, "<<< %s %s Ver.%d.%d.%d >>>", PROGRAM, prog, VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD);
		}

		if (!geom) {
			// open log file
			if (io) {
				if ((fp_log = fopen(FN_log, "w")) == NULL) {
					printf(errfmt, FN_log);
					fflush(stdout);
					ierr = 1;
				}
			}
			comm_check(ierr, 0, prompt);

			// monitor
			if (io) {
				monitor1(fp_log, str);
				sprintf(str, "Process = %d, Thread = %d", commSize, nthread);
				monitor1(fp_log, str);
			}
		}
		else {
			if (io) {
				printf("%s\n", str);
				fflush(stdout);
			}
		}

		// plot geometry 3d
		if (io) {
			plot3dGeom(ev);
		}

		// plot geometry only : exit
		if (geom) {
			mpi_close();
			exit(0);
		}

		// broadcast (MPI)
		if (commSize > 1) {
			comm_broadcast();
		}

		// setup
		setupSize();
		setupSizeNear();
		memalloc1();
		memalloc2();
		memalloc3();
		setup();
		alloc_farfield();

		// monitor
		if (io) {
			monitor2(fp_log, 0);
		}

		// solve
		cpu[1] = comm_cputime();
		solve(io, 1, fp_log);
		cpu[2] = comm_cputime();

		// output
		if (io) {
			// input imepedanece
			if ((NFeed > 0) && (NFreq1 > 0)) {
				zfeed();
				outputZfeed(fp_log);
			}

			// S-parameters
			if ((NPoint > 0) && (NFreq1 > 0)) {
				spara();
				outputSpara(fp_log);
			}

			// coupling
			if ((NFeed > 0) && (NPoint > 0) && (NFreq1 > 0)) {
				outputCoupling(fp_log);
			}

			// cross section
			if (IPlanewave && (NFreq2 > 0)) {
				outputCross(fp_log);
			}

			// output files
			monitor3(fp_log, ev);

			// write ofd.out
			if (runMode == 1) {
				if ((fp_out = fopen(FN_out, "wb")) == NULL) {
					printf(errfmt, FN_out);
					getchar();
					exit(1);
				}
				writeout(fp_out);
				fclose(fp_out);
			}
		}
	}

	// === post ===

	if (io && ((runMode == 0) || (runMode == 2))) {
		// read ofd.out
		if (runMode == 2) {
			if ((fp_out = fopen(FN_out, "rb")) == NULL) {
				printf(errfmt, FN_out);
				getchar();
				exit(1);
			}
			readout(fp_out);
			fclose(fp_out);
			alloc_farfield();
		}

		// post process
		post(ev);
	}

	// free
	if ((runMode == 0) || (runMode == 1)) {
		memfree1();
	}
	memfree3();

	// cpu time
	cpu[3] = comm_cputime();

	if (io && ((runMode == 0) || (runMode == 1))) {
		// cpu time
		monitor4(fp_log, cpu);

		// close log file
		if (fp_log != NULL) {
			fclose(fp_log);
		}
	}

	// finalize MPI
	mpi_close();

	// prompt
	if (io && prompt) getchar();

	return 0;
}

static void args(int argc, char *argv[],
	int *nthread, int *ev, int *geom, int *prompt, char fn_in[])
{
	const char usage[] = "Usage : mpiexec -n <process> ofd_mpi [-solver|-post] [-n <thread>] [-ev] [-geom] [-prompt] <datafile>";

	if (argc < 2) {
		if (commRank == 0) {
			printf("%s\n", usage);
			fflush(stdout);
		}
		mpi_close();
		exit(0);
	}

	while (--argc) {
		++argv;
		if      (!strcmp(*argv, "-solver")) {
			runMode = 1;
		}
		else if (!strcmp(*argv, "-post")) {
			runMode = 2;
		}
		else if (!strcmp(*argv, "-n")) {
			if (--argc) {
				*nthread = atoi(*++argv);
				if (*nthread < 1) *nthread = 1;
			}
			else {
				break;
			}
		}
		else if (!strcmp(*argv, "-ev")) {
			*ev = 1;
		}
		else if (!strcmp(*argv, "-geom")) {
			*geom = 1;
		}
		else if (!strcmp(*argv, "-prompt")) {
			*prompt = 1;
		}
		else if (!strcmp(*argv, "--help")) {
			if (commRank == 0) {
				printf("%s\n", usage);
				fflush(stdout);
			}
			mpi_close();
			exit(0);
		}
		else {
			strcpy(fn_in, *argv);
		}
	}
}
