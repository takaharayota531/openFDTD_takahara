/*
setupSize.c

setup array size
*/

#include "ofd.h"
#include "ofd_prototype.h"

void setupSize(void)
{
	const int lx = (iABC == 0) ? 1 : (iABC == 1) ? cPML.l : 0;
	const int ly = (iABC == 0) ? 1 : (iABC == 1) ? cPML.l : 0;
	const int lz = (iABC == 0) ? 1 : (iABC == 1) ? cPML.l : 0;

	Nk = 1;
	Nj = (Nz + (2 * lz) + 1);
	Ni = (Ny + (2 * ly) + 1) * Nj;
	N0 = (lx * Ni) + (ly * Nj) + (lz * Nk);  // offset
	//printf("%d %d %d %d %d %d %d\n", Nx, Ny, Nz, Ni, Nj, Nk, N0);

	NN = NA(Nx + lx, Ny + ly, Nz + lz) + 1;
	//printf("%zd\n", NN);
	assert(NN > 0);
	//printf("%d=0 %d=%d\n", NA(-lx, -ly, -lz), NN, (Nx + (2 * lx) + 1) * (Ny + (2 * ly) + 1) * (Nz + (2 * lz) + 1));

	iMin = 0;
	iMax = Nx;
	jMin = 0;
	jMax = Ny;
	kMin = 0;
	kMax = Nz;

	NEx = (Nx + 0) * (Ny + 1) * (Nz + 1);
	NEy = (Ny + 0) * (Nz + 1) * (Nx + 1);
	NEz = (Nz + 0) * (Nx + 1) * (Ny + 1);
	NHx = (Nx + 1) * (Ny + 2) * (Nz + 2);
	NHy = (Ny + 1) * (Nz + 2) * (Nx + 2);
	NHz = (Nz + 1) * (Nx + 2) * (Ny + 2);
/*
	assert(NEX(Nx - 1, Ny,     Nz    ) == NEx - 1);
	assert(NEY(Nx,     Ny - 1, Nz    ) == NEy - 1);
	assert(NEZ(Nx,     Ny,     Nz - 1) == NEz - 1);
	assert(NHX(Nx,     Ny,     Nz    ) == NHx - 1);
	assert(NHY(Nx,     Ny,     Nz    ) == NHy - 1);
	assert(NHZ(Nx,     Ny,     Nz    ) == NHz - 1);
*/
	// ABC (array size)
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

	// dummy for MPI
	iProc = (int *)malloc((Nx + 1) * sizeof(int));
	for (int i = 0; i <= Nx; i++) {
		iProc[i] = 1;
	}
}
