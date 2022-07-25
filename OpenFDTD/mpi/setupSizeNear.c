/*
setupSizeNear.c (MPI)

setup near field array size
*/

#include "ofd.h"
#include "ofd_mpi.h"

static void setupSizeNear1d(void)
{
	if (NNear1d <= 0) return;

	l_LNear1d = (int *)malloc(NNear1d * sizeof(int));

	for (int m = 0; m < NNear1d; m++) {
		int num = 0;
		if      (Near1d[m].dir == 'X') {
			//int j = Near1d[m].id1;
			//int k = Near1d[m].id2;
			for (int i = 0; i <= Nx; i++) {
				if (!iProc[i]) continue;
				num++;
			}
		}
		else if (Near1d[m].dir == 'Y') {
			//int k = Near1d[m].id1;
			int i = Near1d[m].id2;
			for (int j = 0; j <= Ny; j++) {
				if (!iProc[i]) continue;
				num++;
			}
		}
		else if (Near1d[m].dir == 'Z') {
			int i = Near1d[m].id1;
			//int j = Near1d[m].id2;
			for (int k = 0; k <= Nz; k++) {
				if (!iProc[i]) continue;
				num++;
			}
		}
		l_LNear1d[m] = num;
	}

	int sum = 0;
	for (int m = 0; m < NNear1d; m++) {
		sum += l_LNear1d[m];
	}
	Near1d_size = sum * NFreq2 * sizeof(d_complex_t);
}

static void setupSizeNear2d(void)
{
	if (NNear2d <= 0) return;

	l_LNear2d = (int *)malloc(NNear2d * sizeof(int));

	for (int m = 0; m < NNear2d; m++) {
		int num = 0;
		if      (Near2d[m].dir == 'X') {
			// Y-Z
			int i = Near2d[m].id0;
			for (int j = 0; j <= Ny; j++) {
			for (int k = 0; k <= Nz; k++) {
				if (!iProc[i]) continue;
				num++;
			}
			}
		}
		else if (Near2d[m].dir == 'Y') {
			// X-Z
			//int j = Near2d[m].id0;
			for (int i = 0; i <= Nx; i++) {
			for (int k = 0; k <= Nz; k++) {
				if (!iProc[i]) continue;
				num++;
			}
			}
		}
		else if (Near2d[m].dir == 'Z') {
			// X-Y
			//int k = Near2d[m].id0;
			for (int i = 0; i <= Nx; i++) {
			for (int j = 0; j <= Ny; j++) {
				if (!iProc[i]) continue;
				num++;
			}
			}
		}
		l_LNear2d[m] = num;
		//printf("%d %d %d %d\n", commSize, commRank, m, l_LNear2d[m]);
	}

	int sum = 0;
	for (int m = 0; m < NNear2d; m++) {
		sum += l_LNear2d[m];
	}
	Near2d_size = sum * NFreq2 * sizeof(d_complex_t);
}

void setupSizeNear(void)
{
	setupSizeNear1d();
	setupSizeNear2d();
}
