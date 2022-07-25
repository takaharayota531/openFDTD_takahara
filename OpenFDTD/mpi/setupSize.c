/*
setupSize.c (MPI)

setup array size
*/

#include "ofd.h"
#include "ofd_mpi.h"
#include "ofd_prototype.h"

void setupSize(void)
{
	// too many prosess

	if (commSize > Nx) {
		if (commRank == 0) {
			fprintf(stderr, "*** too many processes = %d (limit = %d)\n", commSize, Nx);
			fflush(stderr);
		}
		comm_check(1, 0, 1);
	}

	// cells of each process

	int *ncell = (int *)malloc(commSize * sizeof(int));
	for (int ip = 0; ip < commSize; ip++) {
		ncell[ip] = Nx / commSize;
	}
	for (int ip = 0; ip < (Nx % commSize); ip++) {
		ncell[ip]++;
	}
	//for (ip = 0; ip < commSize; ip++) {printf("%d %d\n", ip, ncell[ip]); fflush(stdout);}

	// iMin...iMax, jMin...jMax, kMin...kMax

	int isum = 0;
	for (int ip = 0; ip < commRank; ip++) {
		isum += ncell[ip];
	}
	iMin = isum;
	iMax = iMin + ncell[commRank];

	jMin = 0;
	jMax = Ny;
	kMin = 0;
	kMax = Nz;
	//printf("%d %d %d %d %d %d %d %d\n", commSize, commRank, iMin, iMax, jMin, jMax, kMin, kMax); fflush(stdout);

	free(ncell);

	// array index

	const int lx = (iABC == 0) ? 1 : (iABC == 1) ? cPML.l : 0;
	const int ly = (iABC == 0) ? 1 : (iABC == 1) ? cPML.l : 0;
	const int lz = (iABC == 0) ? 1 : (iABC == 1) ? cPML.l : 0;

	Nk = 1;
	Nj = ((kMax - kMin) + (2 * lz) + 1);
	Ni = ((jMax - jMin) + (2 * ly) + 1) * Nj;
	N0 = -((iMin - lx) * Ni + (jMin - ly) * Nj + (kMin - lz) * Nk);  // offset

	NN = NA(iMax + lx, jMax + ly, kMax + lz) + 1;
	//printf("%d %d %d %d %d %d\n", commSize, commRank, Ni, Nj, N0, NN); fflush(stdout);
	//printf("%d %d %zd=0 %zd=%d\n", commSize, commRank, NA(iMin - lx, jMin - ly, kMin - lz), NN, (iMax - iMin + (2 * lx) + 1) * (jMax - jMin + (2 * ly) + 1) * (kMax - kMin + (2 * lz) + 1)); fflush(stdout);

	NEx = (Nx + 0) * (Ny + 1) * (Nz + 1);
	NEy = (Ny + 0) * (Nz + 1) * (Nx + 1);
	NEz = (Nz + 0) * (Nx + 1) * (Ny + 1);
	NHx = (Nx + 1) * (Ny + 2) * (Nz + 2);
	NHy = (Ny + 1) * (Nz + 2) * (Nx + 2);
	NHz = (Nz + 1) * (Nx + 2) * (Ny + 2);

	// MPI buffer

	int jmin_hy = -ly + 1;
	int jmin_hz = -ly + 0;
	int kmin_hy = -lz + 0;
	int kmin_hz = -lz + 1;

	int jmax_hy = Ny + ly;
	int jmax_hz = Ny + ly;
	int kmax_hy = Nz + lz;
	int kmax_hz = Nz + lz;

	// 0 : -X boundary - 1 (recv)
	// 1 : -X boundary (send)
	// 2 : +X boundary (send)
	// 3 : +X boundary + 1 (recv)

	Offset_Hy[0] = NA(iMin - 1, jmin_hy, kmin_hy);
	Offset_Hy[1] = NA(iMin,     jmin_hy, kmin_hy);
	Offset_Hy[2] = NA(iMax - 1, jmin_hy, kmin_hy);
	Offset_Hy[3] = NA(iMax,     jmin_hy, kmin_hy);

	Offset_Hz[0] = NA(iMin - 1, jmin_hz, kmin_hz);
	Offset_Hz[1] = NA(iMin,     jmin_hz, kmin_hz);
	Offset_Hz[2] = NA(iMax - 1, jmin_hz, kmin_hz);
	Offset_Hz[3] = NA(iMax,     jmin_hz, kmin_hz);

	Length_Hy = Nj * (jmax_hy - jmin_hy) + Nk * (kmax_hy - kmin_hy);
	Length_Hz = Nj * (jmax_hz - jmin_hz) + Nk * (kmax_hz - kmin_hz);

	size_t size = (Length_Hy + Length_Hz) * sizeof(real_t);
	sendBuf = (real_t *)malloc(size);
	recvBuf = (real_t *)malloc(size);

	// ABC
	numMurHx = numMurHy = numMurHz = 0;
	numPmlHx = numPmlHy = numPmlHz = 0;
	if      (iABC == 0) {
		setupMurHx(0);
		setupMurHy(0);
		setupMurHz(0);
		//printf("%zd %zd %zd\n", numMurHx, numMurHy, numMurHz);
	}
	else if (iABC == 1) {
		setupPmlEx(0);
		setupPmlEy(0);
		setupPmlEz(0);
		setupPmlHx(0);
		setupPmlHy(0);
		setupPmlHz(0);
		//printf("%zd %zd %zd\n", numPmlHx, numPmlHy, numPmlHz);
	}

	// my rank include index 'i' ?

	iProc = (int *)malloc((Nx + 1) * sizeof(int));
	for (int i = 0; i <= Nx; i++) {
		iProc[i] = ((i >= iMin) && (i < iMax)) || ((commRank == commSize - 1) && (i == Nx));
		//printf("%d %d %d %d\n", commSize, commRank, i, iProc[i]); fflush(stdout);
	}
}
