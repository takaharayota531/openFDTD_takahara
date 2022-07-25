/*
OpenFDTD Version 2.7.5 (CUDA)

Copyright (C) EEM Inc. 2014-2022
URL : http://www.e-em.co.jp/OpenFDTD/
Mail : info@e-em.co.jp
*/

#define MAIN
#include "ofd.h"
#include "ofd_cuda.h"
#undef MAIN

#include "ofd_prototype.h"

static void args(int, char *[], int *, int *, int *, int *, char []);

int main(int argc, char *argv[])
{
	const char prog[] = "(CUDA)";
	const char errfmt[] = "*** file %s open error.\n";
	char str[BUFSIZ];
	double cpu[4];
	FILE *fp_in = NULL, *fp_out = NULL, *fp_log = NULL;

	// constant
	commSize = 1;
	commRank = 0;
	const int io = 1;

	// arguments
	runMode = 0;
	GPU = 1;
	UM = 0;
	int device = 0;
	int ev = 0;
	int geom = 0;
	int prompt = 0;
	char fn_in[BUFSIZ] = "";
	args(argc, argv, &device, &ev, &geom, &prompt, fn_in);

	// input
	if (io) {
		if ((fp_in = fopen(fn_in, "r")) == NULL) {
			printf(errfmt, fn_in);
			getchar();
			exit(1);
		}
		if (input(fp_in, (!geom && (runMode < 2)))) {
			getchar();
			exit(1);
		}
		fclose(fp_in);
	}

	// cpu time
	cpu[0] = cputime();

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
					getchar();
					exit(1);
				}
			}

			// monitor
			if (io) {
				monitor1(fp_log, str);
				info_gpu(fp_log, device, GPU, UM);
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
			exit(0);
		}

		// setup
		setupSize();
		setupSizeNear();
		memalloc1();
		memalloc2_gpu();
		memalloc3_gpu();
		setup();
		alloc_farfield();

		// monitor
		if (io) {
			monitor2(fp_log, GPU);
		}

		// solve
		cpu[1] = cputime();
		solve(io, 1, fp_log);
		cpu[2] = cputime();

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
	memfree3_gpu();

	// cpu time
	cpu[3] = cputime();

	if (io && ((runMode == 0) || (runMode == 1))) {
		// cpu time
		monitor4(fp_log, cpu);

		// close log file
		if (fp_log != NULL) {
			fclose(fp_log);
		}
	}

	// prompt
	if (io && prompt) getchar();

	return 0;
}

static void args(int argc, char *argv[],
	int *device, int *ev, int *geom, int *prompt, char fn_in[])
{
	const char usage[] = "Usage : ofd_cuda [-solver|-post] [-gpu|-cpu] [-hdm|-um] [-device <device>] [-ev] [-geom] [-prompt] <datafile>";

	if (argc < 2) {
		printf("%s\n", usage);
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
		else if (!strcmp(*argv, "-gpu")) {
			GPU = 1;
		}
		else if (!strcmp(*argv, "-cpu")) {
			GPU = 0;
		}
		else if (!strcmp(*argv, "-hdm")) {
			UM = 0;
		}
		else if (!strcmp(*argv, "-um")) {
			UM = 1;
		}
		else if (!strcmp(*argv, "-device")) {
			if (--argc) {
				*device = atoi(*++argv);
				if (*device < 0) *device = 0;
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
			printf("%s\n", usage);
			exit(0);
		}
		else {
			strcpy(fn_in, *argv);
		}
	}
}
