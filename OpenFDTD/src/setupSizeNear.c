/*
setupSizeNear.c

setup near field array size
*/

#include "ofd.h"

static void setupSizeNear1d(void)
{
	if (NNear1d <= 0) return;

	LNear1d = (int *)malloc(NNear1d * sizeof(int));

	for (int m = 0; m < NNear1d; m++) {
		int num = 0;
		if      (Near1d[m].dir == 'X') {
			num = Nx + 1;
		}
		else if (Near1d[m].dir == 'Y') {
			num = Ny + 1;
		}
		else if (Near1d[m].dir == 'Z') {
			num = Nz + 1;
		}
		LNear1d[m] = num;
	}

	int sum = 0;
	for (int m = 0; m < NNear1d; m++) {
		sum += LNear1d[m];
	}
	Near1d_size = sum * NFreq2 * sizeof(d_complex_t);
}

static void setupSizeNear2d(void)
{
	if (NNear2d <= 0) return;

	LNear2d = (int *)malloc(NNear2d * sizeof(int));

	for (int m = 0; m < NNear2d; m++) {
		int num = 0;
		if      (Near2d[m].dir == 'X') {
			num = (Ny + 1) * (Nz + 1);
		}
		else if (Near2d[m].dir == 'Y') {
			num = (Nz + 1) * (Nx + 1);
		}
		else if (Near2d[m].dir == 'Z') {
			num = (Nx + 1) * (Ny + 1);
		}
		LNear2d[m] = num;
		//printf("%d %d\n", m, LNear2d[m]);
	}

	int sum = 0;
	for (int m = 0; m < NNear2d; m++) {
		sum += LNear2d[m];
	}
	Near2d_size = sum * NFreq2 * sizeof(d_complex_t);
}

void setupSizeNear(void)
{
	setupSizeNear1d();
	setupSizeNear2d();
}
