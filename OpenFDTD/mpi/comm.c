/*
comm.c (MPI)

MPI routines
*/

#ifdef _MPI
#include <mpi.h>
#endif

#include "ofd.h"
#include "ofd_mpi.h"

// initialize
void mpi_init(int argc, char **argv)
{
#ifdef _MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &commSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
#else
	commSize = 1;
	commRank = 0;
	argc = argc;	// dummy
	argv = argv;	// dummy
#endif
}


// close
void mpi_close(void)
{
#ifdef _MPI
	MPI_Finalize();
#endif
}


// check error code
// mode = 0/1 : Bcast/Allreduce
void comm_check(int ierr, int mode, int prompt)
{
#ifdef _MPI
	if (commSize > 1) {
		if (mode == 0) {
			MPI_Bcast(&ierr, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
		else {
			int g_ierr;
			MPI_Allreduce(&ierr, &g_ierr, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
			ierr = g_ierr;
		}
	}
	if (ierr) {
		MPI_Finalize();
	}
#endif
	mode = mode;  // dummy
	if (ierr) {
		if (prompt && (commRank == 0)) {
			getchar();
		}
		exit(0);
	}
}


// gather string
void comm_string(const char *str, char *lstr)
{
#ifdef _MPI
	char buff[BUFSIZ];
	if (commRank == 0) {
		MPI_Status status;
		strcpy(lstr, str);
		for (int i = 1; i < commSize; i++) {
			MPI_Recv(buff, BUFSIZ, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
			strcat(lstr, "\n");
			strcat(lstr, buff);
		}
	}
	else {
		strcpy(buff, str);
		MPI_Send(buff, BUFSIZ, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	}
#else
	strcpy(lstr, str);
#endif
}


// get cpu time [sec]
double comm_cputime(void)
{
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	return MPI_Wtime();
#else
#ifdef _WIN32
	return (double)clock() / CLOCKS_PER_SEC;
#else
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return (ts.tv_sec + (ts.tv_nsec * 1e-9));
#endif  // _WIN32
#endif  // MPI
}


// broadcast input data
void comm_broadcast(void)
{
#ifdef _MPI
	int    *i_buf = NULL;
	double *d_buf = NULL;
	char   *c_buf = NULL;
	int    i_num = 0;
	int    d_num = 0;
	int    c_num = 0;

	// variables to buffers (root only)

	if (commRank == 0) {
		// number of data
		i_num = 21 + (1 * (int)NMaterial) + (2 * (int)NGeometry) + (2 * NNear1d) + NNear2d + NFar1d;
		d_num =  6 + (8 * (int)NMaterial) + (8 * (int)NGeometry)                                    + NFreq1 + NFreq2 + (Nx + 1) + (Ny + 1) + (Nz + 1) + Nx + Ny + Nz;
		c_num =                                                  + (1 * NNear1d) + NNear2d + NFar1d;
		if (NFeed > 0) {
			i_num += 3 * NFeed;
			d_num += 5 * NFeed;
			c_num += 1 * NFeed;
		}
		if (IPlanewave) {
			d_num += 13;
		}
		if (NPoint > 0) {
			i_num += 3 * (NPoint + 2);
			d_num += 3 * (NPoint + 2);
			c_num += 1 * (NPoint + 2);
		}
		if (NInductor > 0) {
			i_num += 3 * NInductor;
			d_num += 6 * NInductor;
			c_num += 1 * NInductor;
		}

		// alloc
		i_buf = (int *)   malloc(i_num * sizeof(int));
		d_buf = (double *)malloc(d_num * sizeof(double));
		c_buf = (char *)  malloc(c_num * sizeof(char));

		int i_id = 0;
		int d_id = 0;
		int c_id = 0;

		i_buf[i_id++] = Nx;
		i_buf[i_id++] = Ny;
		i_buf[i_id++] = Nz;
		i_buf[i_id++] = (int)NMaterial;
		i_buf[i_id++] = (int)NGeometry;
		i_buf[i_id++] = NFeed;
		i_buf[i_id++] = IPlanewave;
		i_buf[i_id++] = iABC;
		i_buf[i_id++] = cPML.l;
		i_buf[i_id++] = PBCx;
		i_buf[i_id++] = PBCy;
		i_buf[i_id++] = PBCz;
		i_buf[i_id++] = NFreq1;
		i_buf[i_id++] = NFreq2;
		i_buf[i_id++] = Solver.maxiter;
		i_buf[i_id++] = Solver.nout;
		i_buf[i_id++] = NPoint;
		i_buf[i_id++] = NInductor;
		i_buf[i_id++] = NNear1d;
		i_buf[i_id++] = NNear2d;
		i_buf[i_id++] = NFar1d;

		d_buf[d_id++] = rFeed;
		d_buf[d_id++] = cPML.m;
		d_buf[d_id++] = cPML.r0;
		d_buf[d_id++] = Solver.converg;
		d_buf[d_id++] = Dt;
		d_buf[d_id++] = Tw;

		for (int i = 0; i <= Nx; i++) {
			d_buf[d_id++] = Xn[i];
		}
		for (int j = 0; j <= Ny; j++) {
			d_buf[d_id++] = Yn[j];
		}
		for (int k = 0; k <= Nz; k++) {
			d_buf[d_id++] = Zn[k];
		}

		for (int i = 0; i < Nx; i++) {
			d_buf[d_id++] = Xc[i];
		}
		for (int j = 0; j < Ny; j++) {
			d_buf[d_id++] = Yc[j];
		}
		for (int k = 0; k < Nz; k++) {
			d_buf[d_id++] = Zc[k];
		}

		for (int n = 0; n < NMaterial; n++) {
			i_buf[i_id++] = Material[n].type;
			d_buf[d_id++] = Material[n].epsr;
			d_buf[d_id++] = Material[n].esgm;
			d_buf[d_id++] = Material[n].amur;
			d_buf[d_id++] = Material[n].msgm;
			d_buf[d_id++] = Material[n].einf;
			d_buf[d_id++] = Material[n].ae;
			d_buf[d_id++] = Material[n].be;
			d_buf[d_id++] = Material[n].ce;
		}

		for (int n = 0; n < NGeometry; n++) {
			i_buf[i_id++] = (int)Geometry[n].m;
			i_buf[i_id++] = Geometry[n].shape;
			for (int i = 0; i < 8; i++) {
				d_buf[d_id++] = Geometry[n].g[i];
			}
		}

		for (int n = 0; n < NFeed; n++) {
			c_buf[c_id++] = Feed[n].dir;
			i_buf[i_id++] = Feed[n].i;
			i_buf[i_id++] = Feed[n].j;
			i_buf[i_id++] = Feed[n].k;
			d_buf[d_id++] = Feed[n].volt;
			d_buf[d_id++] = Feed[n].delay;
			d_buf[d_id++] = Feed[n].dx;
			d_buf[d_id++] = Feed[n].dy;
			d_buf[d_id++] = Feed[n].dz;
		}

		if (IPlanewave) {
			for (int m = 0; m < 3; m++) {
				d_buf[d_id++] = Planewave.ei[m];
				d_buf[d_id++] = Planewave.hi[m];
				d_buf[d_id++] = Planewave.ri[m];
				d_buf[d_id++] = Planewave.r0[m];
			}
			d_buf[d_id++] = Planewave.ai;
		}

		if (NPoint > 0) {
			for (int n = 0; n < NPoint + 2; n++) {
				c_buf[c_id++] = Point[n].dir;
				i_buf[i_id++] = Point[n].i;
				i_buf[i_id++] = Point[n].j;
				i_buf[i_id++] = Point[n].k;
				d_buf[d_id++] = Point[n].dx;
				d_buf[d_id++] = Point[n].dy;
				d_buf[d_id++] = Point[n].dz;
			}
		}

		for (int n = 0; n < NInductor; n++) {
			c_buf[c_id++] = Inductor[n].dir;
			i_buf[i_id++] = Inductor[n].i;
			i_buf[i_id++] = Inductor[n].j;
			i_buf[i_id++] = Inductor[n].k;
			d_buf[d_id++] = Inductor[n].dx;
			d_buf[d_id++] = Inductor[n].dy;
			d_buf[d_id++] = Inductor[n].dz;
			d_buf[d_id++] = Inductor[n].fctr;
			d_buf[d_id++] = Inductor[n].e;
			d_buf[d_id++] = Inductor[n].esum;
		}

		for (int n = 0; n < NFreq1; n++) {
			d_buf[d_id++] = Freq1[n];
		}

		for (int n = 0; n < NFreq2; n++) {
			d_buf[d_id++] = Freq2[n];
		}

		for (int n = 0; n < NNear1d; n++) {
			c_buf[c_id++] = Near1d[n].dir;
			i_buf[i_id++] = Near1d[n].id1;
			i_buf[i_id++] = Near1d[n].id2;
		}

		for (int n = 0; n < NNear2d; n++) {
			c_buf[c_id++] = Near2d[n].dir;
			i_buf[i_id++] = Near2d[n].id0;
		}

		for (int n = 0; n < NFar1d; n++) {
			c_buf[c_id++] = Far1d[n].dir;
			i_buf[i_id++] = Far1d[n].div;
		}

		// check
		assert(i_id == i_num);
		assert(d_id == d_num);
		assert(c_id == c_num);
	}

	// broadcast (root to non-root)

	MPI_Bcast(&i_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&d_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&c_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//printf("%d %d %d %d %d\n", commSize, commRank, i_num, d_num, c_num); fflush(stdout);

	// alloc
	if (commRank > 0) {
		i_buf = (int *)   malloc(i_num * sizeof(int));
		d_buf = (double *)malloc(d_num * sizeof(double));
		c_buf = (char *)  malloc(c_num * sizeof(char));
	}

	MPI_Bcast(i_buf, i_num, MPI_INT,    0, MPI_COMM_WORLD);
	MPI_Bcast(d_buf, d_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(c_buf, c_num, MPI_CHAR,   0, MPI_COMM_WORLD);

	// buffers to variables (non-root)

	if (commRank > 0) {
		int i_id = 0;
		//int f_id = 0;
		int d_id = 0;
		int c_id = 0;

		Nx              = i_buf[i_id++];
		Ny              = i_buf[i_id++];
		Nz              = i_buf[i_id++];
		NMaterial       = i_buf[i_id++];
		NGeometry       = i_buf[i_id++];
		NFeed           = i_buf[i_id++];
		IPlanewave      = i_buf[i_id++];
		iABC            = i_buf[i_id++];
		cPML.l          = i_buf[i_id++];
		PBCx            = i_buf[i_id++];
		PBCy            = i_buf[i_id++];
		PBCz            = i_buf[i_id++];
		NFreq1          = i_buf[i_id++];
		NFreq2          = i_buf[i_id++];
		Solver.maxiter  = i_buf[i_id++];
		Solver.nout     = i_buf[i_id++];
		NPoint          = i_buf[i_id++];
		NInductor       = i_buf[i_id++];
		NNear1d         = i_buf[i_id++];
		NNear2d         = i_buf[i_id++];
		NFar1d          = i_buf[i_id++];

		rFeed           = d_buf[d_id++];
		cPML.m          = d_buf[d_id++];
		cPML.r0         = d_buf[d_id++];
		Solver.converg  = d_buf[d_id++];
		Dt              = d_buf[d_id++];
		Tw              = d_buf[d_id++];

		Xn = (double *)malloc((Nx + 1) * sizeof(double));
		Yn = (double *)malloc((Ny + 1) * sizeof(double));
		Zn = (double *)malloc((Nz + 1) * sizeof(double));
		for (int i = 0; i <= Nx; i++) {
			Xn[i] = d_buf[d_id++];
		}
		for (int j = 0; j <= Ny; j++) {
			Yn[j] = d_buf[d_id++];
		}
		for (int k = 0; k <= Nz; k++) {
			Zn[k] = d_buf[d_id++];
		}

		Xc = (double *)malloc(Nx * sizeof(double));
		Yc = (double *)malloc(Ny * sizeof(double));
		Zc = (double *)malloc(Nz * sizeof(double));
		for (int i = 0; i < Nx; i++) {
			Xc[i] = d_buf[d_id++];
		}
		for (int j = 0; j < Ny; j++) {
			Yc[j] = d_buf[d_id++];
		}
		for (int k = 0; k < Nz; k++) {
			Zc[k] = d_buf[d_id++];
		}

		if (NMaterial > 0) {
			Material = (material_t *)malloc(NMaterial * sizeof(material_t));
			for (int n = 0; n < NMaterial; n++) {
				Material[n].type = i_buf[i_id++];
				Material[n].epsr = d_buf[d_id++];
				Material[n].esgm = d_buf[d_id++];
				Material[n].amur = d_buf[d_id++];
				Material[n].msgm = d_buf[d_id++];
				Material[n].einf = d_buf[d_id++];
				Material[n].ae   = d_buf[d_id++];
				Material[n].be   = d_buf[d_id++];
				Material[n].ce   = d_buf[d_id++];
			}
		}

		if (NGeometry > 0){
			Geometry = (geometry_t *)malloc(NGeometry * sizeof(geometry_t));
			for (int n = 0; n < NGeometry; n++) {
				Geometry[n].m     = (id_t)i_buf[i_id++];
				Geometry[n].shape = i_buf[i_id++];
				for (int i = 0; i < 8; i++) {
					Geometry[n].g[i] = d_buf[d_id++];
				}
			}
		}

		if (NFeed > 0) {
			Feed = (feed_t *)malloc(NFeed * sizeof(feed_t));
			for (int n = 0; n < NFeed; n++) {
				Feed[n].dir   = c_buf[c_id++];
				Feed[n].i     = i_buf[i_id++];
				Feed[n].j     = i_buf[i_id++];
				Feed[n].k     = i_buf[i_id++];
				Feed[n].volt  = d_buf[d_id++];
				Feed[n].delay = d_buf[d_id++];
				Feed[n].dx    = d_buf[d_id++];
				Feed[n].dy    = d_buf[d_id++];
				Feed[n].dz    = d_buf[d_id++];
			}
		}

		if (IPlanewave) {
			for (int m = 0; m < 3; m++) {
				Planewave.ei[m] = d_buf[d_id++];
				Planewave.hi[m] = d_buf[d_id++];
				Planewave.ri[m] = d_buf[d_id++];
				Planewave.r0[m] = d_buf[d_id++];
			}
			Planewave.ai = d_buf[d_id++];
		}

		if (NPoint > 0) {
			Point = (point_t *)malloc((NPoint + 2) * sizeof(point_t));
			for (int n = 0; n < NPoint + 2; n++) {
				Point[n].dir = c_buf[c_id++];
				Point[n].i   = i_buf[i_id++];
				Point[n].j   = i_buf[i_id++];
				Point[n].k   = i_buf[i_id++];
				Point[n].dx  = d_buf[d_id++];
				Point[n].dy  = d_buf[d_id++];
				Point[n].dz  = d_buf[d_id++];
			}
		}

		if (NInductor > 0) {
			Inductor = (inductor_t *)malloc(NInductor * sizeof(inductor_t));
			for (int n = 0; n < NInductor; n++) {
				Inductor[n].dir  = c_buf[c_id++];
				Inductor[n].i    = i_buf[i_id++];
				Inductor[n].j    = i_buf[i_id++];
				Inductor[n].k    = i_buf[i_id++];
				Inductor[n].dx   = d_buf[d_id++];
				Inductor[n].dy   = d_buf[d_id++];
				Inductor[n].dz   = d_buf[d_id++];
				Inductor[n].fctr = d_buf[d_id++];
				Inductor[n].e    = d_buf[d_id++];
				Inductor[n].esum = d_buf[d_id++];
			}
		}

		if (NFreq1 > 0) {
			Freq1 = (double *)malloc(NFreq1 * sizeof(double));
			for (int n = 0; n < NFreq1; n++) {
				Freq1[n] = d_buf[d_id++];
			}
		}

		if (NFreq2 > 0) {
			Freq2 = (double *)malloc(NFreq2 * sizeof(double));
			for (int n = 0; n < NFreq2; n++) {
				Freq2[n] = d_buf[d_id++];
			}
		}

		if (NNear1d > 0) {
			Near1d = (near1d_t *)malloc(NNear1d * sizeof(near1d_t));
			for (int n = 0; n < NNear1d; n++) {
				Near1d[n].dir = c_buf[c_id++];
				Near1d[n].id1 = i_buf[i_id++];
				Near1d[n].id2 = i_buf[i_id++];
			}
		}

		if (NNear2d > 0) {
			Near2d = (near2d_t *)malloc(NNear2d * sizeof(near2d_t));
			for (int n = 0; n < NNear2d; n++) {
				Near2d[n].dir = c_buf[c_id++];
				Near2d[n].id0 = i_buf[i_id++];
			}
		}

		if (NFar1d > 0) {
			Far1d = (far1d_t *)malloc(NFar1d * sizeof(far1d_t));
			for (int n = 0; n < NFar1d; n++) {
				Far1d[n].dir = c_buf[c_id++];
				Far1d[n].div = i_buf[i_id++];
			}
		}

		// check
		assert(i_id == i_num);
		assert(d_id == d_num);
		assert(c_id == c_num);
	}

	// free
	free(i_buf);
	free(d_buf);
	free(c_buf);

	// debug
	//printf("%d %d %d %d\n", commSize, commRank, iSIMD, nThread);
	//printf("%d %d %d %d %d %d %d\n", Nx, Ny, Nz, NI, NJ, N0, NN);
	//printf("%d %d %d\n", NMaterial, NGeometry, NFeed);
	//printf("%d %d %e\n", Solver.maxiter, Solver.nout, Solver.converg);
	//for (int i = 0; i <= Nx; i++) printf("%d Xn[%d]=%.5f\n", commRank, i, Xn[i] * 1e3);
	//for (int j = 0; j <= Ny; j++) printf("%d Yn[%d]=%.5f\n", commRank, j, Yn[j] * 1e3);
	//for (int k = 0; k <= Nz; k++) printf("%d Zn[%d]=%.5f\n", commRank, k, Zn[k] * 1e3);
	//for (int n = 0; n < NMaterial; n++) printf("%d %d %e %e %e %e\n", commRank, n, Material[n].epsr, Material[n].esgm, Material[n].amur, Material[n].msgm);
	//for (int n = 0; n < NGeometry; n++) printf("%d %d %d %e %e %e %e %e %e\n", commRank, n, Geometry[n].m, Geometry[n].g[0], Geometry[n].g[1], Geometry[n].g[2], Geometry[n].g[3], Geometry[n].g[4], Geometry[n].g[5]);
	//for (int n = 0; n < NFeed; n++) printf("%d %d %c %d %d %d %e\n", commRank, n, Feed[n].dir, Feed[n].i, Feed[n].j, Feed[n].k, Feed[n].volt);
	//for (int n = 0; n < NFreq1; n++) printf("%d %d %e\n", commRank, n, Freq1[n]);
	//for (int n = 0; n < NFreq2; n++) printf("%d %d %e\n", commRank, n, Freq2[n]);
	//for (int n = 0; n < NPoint + 2; n++) printf("%d %d %c %d %d %d\n", commRank, n, Point[n].dir, Point[n].i, Point[n].j, Point[n].k);
	//for (int n = 0; n < NInductor; n++) printf("%d %d %c %d %d %d %e\n", commRank, n, Inductor[n].dir, Inductor[n].i, Inductor[n].j, Inductor[n].k, Inductor[n].fctr);
	//for (int n = 0; n < NNear1d; n++) printf("%d %d %c %d %d\n", commRank, n, Near1d[n].dir, Near1d[n].id1, Near1d[n].id2);
	//for (int n = 0; n < NNear2d; n++) printf("%d %d %c %d\n", commRank, n, Near2d[n].dir, Near2d[n].id0);
	fflush(stdout);

#endif
}


// share X-boundary Hy and Hz
void comm_boundary(void)
{
#ifdef _MPI
	MPI_Status status;
	const int tag = 0;
	const size_t size_hy = Length_Hy * sizeof(real_t);
	const size_t size_hz = Length_Hz * sizeof(real_t);
	const int count = (int)(Length_Hy + Length_Hz);

	// -X boundary
	if (commRank > 0) {
		// copy to buffer
		memcpy(sendBuf,             Hy + Offset_Hy[1], size_hy);
		memcpy(sendBuf + Length_Hy, Hz + Offset_Hz[1], size_hz);

		// MPI
		int dst = commRank - 1;
		MPI_Sendrecv(sendBuf, count, MPI_REAL_T, dst, tag,
		             recvBuf, count, MPI_REAL_T, dst, tag, MPI_COMM_WORLD, &status);

		// copy from buffer
		memcpy(Hy + Offset_Hy[0], recvBuf,             size_hy);
		memcpy(Hz + Offset_Hz[0], recvBuf + Length_Hy, size_hz);
	}

	// +X boundary
	if (commRank < commSize - 1) {
		// copy to buffer
		memcpy(sendBuf,             Hy + Offset_Hy[2], size_hy);
		memcpy(sendBuf + Length_Hy, Hz + Offset_Hz[2], size_hz);

		// MPI
		int dst = commRank + 1;
		MPI_Sendrecv(sendBuf, count, MPI_REAL_T, dst, tag,
		             recvBuf, count, MPI_REAL_T, dst, tag, MPI_COMM_WORLD, &status);

		// copy from buffer
		memcpy(Hy + Offset_Hy[3], recvBuf,             size_hy);
		memcpy(Hz + Offset_Hz[3], recvBuf + Length_Hy, size_hz);
	}
#endif
}


// PBC : copy H on X boundaries
void comm_pbcx(void)
{
#ifdef _MPI
	MPI_Status status;
	const int tag = 0;
	const size_t size_hy = Length_Hy * sizeof(real_t);
	const size_t size_hz = Length_Hz * sizeof(real_t);
	const int count = (int)(Length_Hy + Length_Hz);

	// -boundary
	if (commRank == 0) {
		// copy to buffer
		memcpy(sendBuf,             Hy + Offset_Hy[1], size_hy);
		memcpy(sendBuf + Length_Hy, Hz + Offset_Hz[1], size_hz);

		// MPI
		int dst = commSize - 1;
		MPI_Sendrecv(sendBuf, count, MPI_REAL_T, dst, tag,
		             recvBuf, count, MPI_REAL_T, dst, tag, MPI_COMM_WORLD, &status);

		// copy from buffer
		memcpy(Hy + Offset_Hy[0], recvBuf,             size_hy);
		memcpy(Hz + Offset_Hz[0], recvBuf + Length_Hy, size_hz);
	}

	// +boundary
	else if (commRank == commSize - 1) {
		// copy to buffer
		memcpy(sendBuf,             Hy + Offset_Hy[2], size_hy);
		memcpy(sendBuf + Length_Hy, Hz + Offset_Hz[2], size_hz);

		// MPI
		int dst = 0;
		MPI_Sendrecv(sendBuf, count, MPI_REAL_T, dst, tag,
		             recvBuf, count, MPI_REAL_T, dst, tag, MPI_COMM_WORLD, &status);

		// copy from buffer
		memcpy(Hy + Offset_Hy[3], recvBuf,             size_hy);
		memcpy(Hz + Offset_Hz[3], recvBuf + Length_Hy, size_hz);
	}
#endif
}


// allreduce average
void comm_average(double fsum[])
{
#ifdef _MPI
	double ftmp[2];

	MPI_Allreduce(fsum, ftmp, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	fsum[0] = ftmp[0];
	fsum[1] = ftmp[1];
#else
	fsum = fsum;	// dummy
#endif
}


// send feed waveform to root process
void comm_feed(void)
{
#ifdef _MPI
	MPI_Status status;

	int count = Solver.maxiter + 1;
	for (int n = 0; n < NFeed; n++) {
		// non-root only
		if      ((commRank == 0) && !iProc[Feed[n].i]) {
			MPI_Recv(&VFeed[n * count], count, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&IFeed[n * count], count, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		}
		else if ((commRank  > 0) &&  iProc[Feed[n].i]) {
			MPI_Send(&VFeed[n * count], count, MPI_DOUBLE, 0,              0, MPI_COMM_WORLD);
			MPI_Send(&IFeed[n * count], count, MPI_DOUBLE, 0,              0, MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
#endif
}


// send point waveform to root process
void comm_point(void)
{
#ifdef _MPI
	MPI_Status status;

	if (NPoint <= 0) return;

	int count = Solver.maxiter + 1;
	for (int n = 0; n < NPoint + 2; n++) {
		// non-root only
		if      ((commRank == 0) && !iProc[Point[n].i]) {
			MPI_Recv(&VPoint[n * count], count, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		}
		else if ((commRank  > 0) &&  iProc[Point[n].i]) {
			MPI_Send(&VPoint[n * count], count, MPI_DOUBLE, 0,              0, MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
#endif
}


// send near1d data to root process
void comm_near1d(void)
{
#ifdef _MPI
	int     id;
	int     g_num = 0;
	int64_t g_adr = 0;
	int     *offset = NULL, *count = NULL;
	double  *send = NULL, *recv = NULL;

	// alloc global array (root)
	if (commRank == 0) {
		int sum = 0;
		for (int m = 0; m < NNear1d; m++) {
			if      (Near1d[m].dir == 'X') {
				sum += Nx + 1;
			}
			else if (Near1d[m].dir == 'Y') {
				sum += Ny + 1;
			}
			else if (Near1d[m].dir == 'Z') {
				sum += Nz + 1;
			}
		}
		size_t g_size = sum * NFreq2 * sizeof(d_complex_t);
		Near1dEx = (d_complex_t *)malloc(g_size);
		Near1dEy = (d_complex_t *)malloc(g_size);
		Near1dEz = (d_complex_t *)malloc(g_size);
		Near1dHx = (d_complex_t *)malloc(g_size);
		Near1dHy = (d_complex_t *)malloc(g_size);
		Near1dHz = (d_complex_t *)malloc(g_size);
	}
	// alloc offset and count (root)
	if (commRank == 0) {
		offset = (int *)malloc(commSize * sizeof(int));
		count  = (int *)malloc(commSize * sizeof(int));
	}

	int64_t l_adr = 0;
	if (commRank == 0) {
		g_adr = 0;
	}
	for (int m = 0; m < NNear1d; m++) {
		// local number of data
		int l_num = l_LNear1d[m];

		// gather count to root
		// 6 = Ex/Ey/Ez/Hx/Hy/Hz components
		// 2 = sizeof(d_complex_t) / sizeof(double)
		int l_count = l_num * 6 * 2;
		MPI_Gather(&l_count, 1, MPI_INT, count, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// setup offset and count (root)
		if (commRank == 0) {
			offset[0] = 0;
			for (int iproc = 1; iproc < commSize; iproc++) {
				offset[iproc] = offset[iproc - 1] + count[iproc - 1];
			}
		}

		// send buffer (all process)
		size_t size = l_count * sizeof(double);
		send = (double *)malloc(size);

		// recv buffer (global array : root only)
		if (commRank == 0) {
			size = (offset[commSize - 1] + count[commSize - 1]) * sizeof(double);
			recv = (double *)malloc(size);
		}

		// global number of data
		if (commRank == 0) {
			g_num = 0;
			if      (Near1d[m].dir == 'X') {
				g_num = Nx + 1;
			}
			else if (Near1d[m].dir == 'Y') {
				g_num = Ny + 1;
			}
			else if (Near1d[m].dir == 'Z') {
				g_num = Nz + 1;
			}
		}

		// gather E and H to root
		for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
			id = 0;
			for (int n = 0; n < l_num; n++) {
				send[id++] = l_Near1dEx[l_adr].r;
				send[id++] = l_Near1dEx[l_adr].i;
				send[id++] = l_Near1dEy[l_adr].r;
				send[id++] = l_Near1dEy[l_adr].i;
				send[id++] = l_Near1dEz[l_adr].r;
				send[id++] = l_Near1dEz[l_adr].i;
				send[id++] = l_Near1dHx[l_adr].r;
				send[id++] = l_Near1dHx[l_adr].i;
				send[id++] = l_Near1dHy[l_adr].r;
				send[id++] = l_Near1dHy[l_adr].i;
				send[id++] = l_Near1dHz[l_adr].r;
				send[id++] = l_Near1dHz[l_adr].i;
				l_adr++;
			}

			MPI_Gatherv(send, id, MPI_DOUBLE, recv, count, offset, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			if (commRank == 0) {
				id = 0;
				for (int n = 0; n < g_num; n++) {
					Near1dEx[g_adr].r = recv[id++];
					Near1dEx[g_adr].i = recv[id++];
					Near1dEy[g_adr].r = recv[id++];
					Near1dEy[g_adr].i = recv[id++];
					Near1dEz[g_adr].r = recv[id++];
					Near1dEz[g_adr].i = recv[id++];
					Near1dHx[g_adr].r = recv[id++];
					Near1dHx[g_adr].i = recv[id++];
					Near1dHy[g_adr].r = recv[id++];
					Near1dHy[g_adr].i = recv[id++];
					Near1dHz[g_adr].r = recv[id++];
					Near1dHz[g_adr].i = recv[id++];
					g_adr++;
				}
			}
		}

		// free
		free(send);
		if (commRank == 0) {
			free(recv);
		}
	}

	// free
	if (commRank == 0) {
		free(offset);
		free(count);
	}
#endif
}


// send near2d data to root process
void comm_near2d(void)
{
#ifdef _MPI
	int     id;
	int     g_num = 0;
	int64_t g_adr = 0;
	int     *offset = NULL, *count = NULL;
	double  *send = NULL, *recv = NULL;

	// alloc global array (root)
	if (commRank == 0) {
		int sum = 0;
		for (int m = 0; m < NNear2d; m++) {
			if      (Near2d[m].dir == 'X') {
				sum += (Ny + 1) * (Nz + 1);
			}
			else if (Near2d[m].dir == 'Y') {
				sum += (Nz + 1) * (Nx + 1);
			}
			else if (Near2d[m].dir == 'Z') {
				sum += (Nx + 1) * (Ny + 1);
			}
		}
		size_t g_size = sum * NFreq2 * sizeof(d_complex_t);
		Near2dEx = (d_complex_t *)malloc(g_size);
		Near2dEy = (d_complex_t *)malloc(g_size);
		Near2dEz = (d_complex_t *)malloc(g_size);
		Near2dHx = (d_complex_t *)malloc(g_size);
		Near2dHy = (d_complex_t *)malloc(g_size);
		Near2dHz = (d_complex_t *)malloc(g_size);
	}

	// alloc offset and count (root)
	if (commRank == 0) {
		offset = (int *)malloc(commSize * sizeof(int));
		count  = (int *)malloc(commSize * sizeof(int));
	}

	int64_t l_adr = 0;
	if (commRank == 0) {
		g_adr = 0;
	}
	for (int m = 0; m < NNear2d; m++) {
		// local number of data
		int l_num = l_LNear2d[m];

		// gather count to root
		// 6 = Ex/Ey/Ez/Hx/Hy/Hz components
		// 2 = sizeof(d_complex_t) / sizeof(double)
		int l_count = l_num * 6 * 2;
		MPI_Gather(&l_count, 1, MPI_INT, count, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// setup offset and count (root)
		if (commRank == 0) {
			offset[0] = 0;
			for (int iproc = 1; iproc < commSize; iproc++) {
				offset[iproc] = offset[iproc - 1] + count[iproc - 1];
			}
		}

		// send buffer (all process)
		size_t size = l_count * sizeof(double);
		send = (double *)malloc(size);

		// recv buffer (global array : root only)
		if (commRank == 0) {
			size = (offset[commSize - 1] + count[commSize - 1]) * sizeof(double);
			recv = (double *)malloc(size);
		}

		// global number of data
		if (commRank == 0) {
			g_num = 0;
			if      (Near2d[m].dir == 'X') {
				g_num = (Ny + 1) * (Nz + 1);
			}
			else if (Near2d[m].dir == 'Y') {
				g_num = (Nz + 1) * (Nx + 1);
			}
			else if (Near2d[m].dir == 'Z') {
				g_num = (Nx + 1) * (Ny + 1);
			}
		}

		// gather E and H to root
		for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
			id = 0;
			for (int n = 0; n < l_num; n++) {
				send[id++] = l_Near2dEx[l_adr].r;
				send[id++] = l_Near2dEx[l_adr].i;
				send[id++] = l_Near2dEy[l_adr].r;
				send[id++] = l_Near2dEy[l_adr].i;
				send[id++] = l_Near2dEz[l_adr].r;
				send[id++] = l_Near2dEz[l_adr].i;
				send[id++] = l_Near2dHx[l_adr].r;
				send[id++] = l_Near2dHx[l_adr].i;
				send[id++] = l_Near2dHy[l_adr].r;
				send[id++] = l_Near2dHy[l_adr].i;
				send[id++] = l_Near2dHz[l_adr].r;
				send[id++] = l_Near2dHz[l_adr].i;
				l_adr++;
			}

			MPI_Gatherv(send, id, MPI_DOUBLE, recv, count, offset, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			if (commRank == 0) {
				id = 0;
				for (int n = 0; n < g_num; n++) {
					Near2dEx[g_adr].r = recv[id++];
					Near2dEx[g_adr].i = recv[id++];
					Near2dEy[g_adr].r = recv[id++];
					Near2dEy[g_adr].i = recv[id++];
					Near2dEz[g_adr].r = recv[id++];
					Near2dEz[g_adr].i = recv[id++];
					Near2dHx[g_adr].r = recv[id++];
					Near2dHx[g_adr].i = recv[id++];
					Near2dHy[g_adr].r = recv[id++];
					Near2dHy[g_adr].i = recv[id++];
					Near2dHz[g_adr].r = recv[id++];
					Near2dHz[g_adr].i = recv[id++];
					g_adr++;
				}
			}
		}

		// free
		free(send);
		if (commRank == 0) {
			free(recv);
		}
	}

	// free
	if (commRank == 0) {
		free(offset);
		free(count);
	}
#endif
}


// send near3d data to root process
void comm_near3d(void)
{
#ifdef _MPI
	MPI_Status status;
	int isend[12], irecv[12];

	if ((NN <= 0) || (NFreq2 <= 0)) return;

	// alloc
	if (commRank == 0) {
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
	}

	for (int ifreq = 0; ifreq < NFreq2; ifreq++) {

		if (commRank == 0) {

			// === self copy (0->0) ===

			// Ex
			for (int i = iMin; i <  iMax; i++) {
			for (int j = jMin; j <= jMax; j++) {
			for (int k = kMin; k <= kMax; k++) {
				const int64_t m = (ifreq * NEx) + NEX(i, j, k);
				const int64_t n = (ifreq * NN) + NA(i, j, k);
				cEx_r[m] = Ex_r[n];
				cEx_i[m] = Ex_i[n];
			}
			}
			}

			// Ey
			for (int i = iMin; i <= iMax; i++) {
			for (int j = jMin; j <  jMax; j++) {
			for (int k = kMin; k <= kMax; k++) {
				const int64_t m = (ifreq * NEy) + NEY(i, j, k);
				const int64_t n = (ifreq * NN) + NA(i, j, k);
				cEy_r[m] = Ey_r[n];
				cEy_i[m] = Ey_i[n];
			}
			}
			}

			// Ez
			for (int i = iMin; i <= iMax; i++) {
			for (int j = jMin; j <= jMax; j++) {
			for (int k = kMin; k <  kMax; k++) {
				const int64_t m = (ifreq * NEz) + NEZ(i, j, k);
				const int64_t n = (ifreq * NN) + NA(i, j, k);
				cEz_r[m] = Ez_r[n];
				cEz_i[m] = Ez_i[n];
			}
			}
			}

			// Hx
			for (int i = iMin - 0; i <= iMax; i++) {
			for (int j = jMin - 1; j <= jMax; j++) {
			for (int k = kMin - 1; k <= kMax; k++) {
				const int64_t m = (ifreq * NHx) + NHX(i, j, k);
				const int64_t n = (ifreq * NN) + NA(i, j, k);
				cHx_r[m] = Hx_r[n];
				cHx_i[m] = Hx_i[n];
			}
			}
			}

			// Hy
			for (int i = iMin - 1; i <= iMax; i++) {
			for (int j = jMin - 0; j <= jMax; j++) {
			for (int k = kMin - 1; k <= kMax; k++) {
				const int64_t m = (ifreq * NHy) + NHY(i, j, k);
				const int64_t n = (ifreq * NN) + NA(i, j, k);
				cHy_r[m] = Hy_r[n];
				cHy_i[m] = Hy_i[n];
			}
			}
			}

			// Hz
			for (int i = iMin - 1; i <= iMax; i++) {
			for (int j = jMin - 1; j <= jMax; j++) {
			for (int k = kMin - 0; k <= kMax; k++) {
				const int64_t m = (ifreq * NHz) + NHZ(i, j, k);
				const int64_t n = (ifreq * NN) + NA(i, j, k);
				cHz_r[m] = Hz_r[n];
				cHz_i[m] = Hz_i[n];
			}
			}
			}

			// === receive ===

			for (int irank = 1; irank < commSize; irank++) {

				// recv
				MPI_Recv(irecv, 12, MPI_INT, irank, 0, MPI_COMM_WORLD, &status);

				int imin = irecv[0];
				int imax = irecv[1];
				int jmin = irecv[2];
				int jmax = irecv[3];
				int kmin = irecv[4];
				int kmax = irecv[5];
				int nex = irecv[6];
				int ney = irecv[7];
				int nez = irecv[8];
				int nhx = irecv[9];
				int nhy = irecv[10];
				int nhz = irecv[11];

				real_t *exrecv = (real_t *)malloc(2 * nex * sizeof(real_t));
				real_t *eyrecv = (real_t *)malloc(2 * ney * sizeof(real_t));
				real_t *ezrecv = (real_t *)malloc(2 * nez * sizeof(real_t));
				real_t *hxrecv = (real_t *)malloc(2 * nhx * sizeof(real_t));
				real_t *hyrecv = (real_t *)malloc(2 * nhy * sizeof(real_t));
				real_t *hzrecv = (real_t *)malloc(2 * nhz * sizeof(real_t));

				MPI_Recv(exrecv, 2 * nex, MPI_REAL_T, irank, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(eyrecv, 2 * ney, MPI_REAL_T, irank, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(ezrecv, 2 * nez, MPI_REAL_T, irank, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(hxrecv, 2 * nhx, MPI_REAL_T, irank, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(hyrecv, 2 * nhy, MPI_REAL_T, irank, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(hzrecv, 2 * nhz, MPI_REAL_T, irank, 0, MPI_COMM_WORLD, &status);

				// === copy to global array ===

				// Ex
				int mex = 0;
				for (int i = imin; i <  imax; i++) {
				for (int j = jmin; j <= jmax; j++) {
				for (int k = kmin; k <= kmax; k++) {
					const int64_t m = (ifreq * NEx) + NEX(i, j, k);
					cEx_r[m] = exrecv[(2 * mex) + 0];
					cEx_i[m] = exrecv[(2 * mex) + 1];
					mex++;
				}
				}
				}
				//assert(mex == nex);

				// Ey
				int mey = 0;
				for (int i = imin; i <= imax; i++) {
				for (int j = jmin; j <  jmax; j++) {
				for (int k = kmin; k <= kmax; k++) {
					const int64_t m = (ifreq * NEy) + NEY(i, j, k);
					cEy_r[m] = eyrecv[(2 * mey) + 0];
					cEy_i[m] = eyrecv[(2 * mey) + 1];
					mey++;
				}
				}
				}
				//assert(mey == ney);

				// Ez
				int mez = 0;
				for (int i = imin; i <= imax; i++) {
				for (int j = jmin; j <= jmax; j++) {
				for (int k = kmin; k <  kmax; k++) {
					const int64_t m = (ifreq * NEz) + NEZ(i, j, k);
					cEz_r[m] = ezrecv[(2 * mez) + 0];
					cEz_i[m] = ezrecv[(2 * mez) + 1];
					mez++;
				}
				}
				}
				//assert(mez == nez);

				// Hx
				int mhx = 0;
				for (int i = imin - 0; i <= imax; i++) {
				for (int j = jmin - 1; j <= jmax; j++) {
				for (int k = kmin - 1; k <= kmax; k++) {
					const int64_t m = (ifreq * NHx) + NHX(i, j, k);
					cHx_r[m] = hxrecv[(2 * mhx) + 0];
					cHx_i[m] = hxrecv[(2 * mhx) + 1];
					mhx++;
				}
				}
				}
				//assert(mhx == nhx);

				// Hy
				int mhy = 0;
				for (int i = imin - 1; i <= imax; i++) {
				for (int j = jmin - 0; j <= jmax; j++) {
				for (int k = kmin - 1; k <= kmax; k++) {
					const int64_t m = (ifreq * NHy) + NHY(i, j, k);
					cHy_r[m] = hyrecv[(2 * mhy) + 0];
					cHy_i[m] = hyrecv[(2 * mhy) + 1];
					mhy++;
				}
				}
				}
				//assert(mhy == nhy);

				// Hz
				int mhz = 0;
				for (int i = imin - 1; i <= imax; i++) {
				for (int j = jmin - 1; j <= jmax; j++) {
				for (int k = kmin - 0; k <= kmax; k++) {
					const int64_t m = (ifreq * NHz) + NHZ(i, j, k);
					cHz_r[m] = hzrecv[(2 * mhz) + 0];
					cHz_i[m] = hzrecv[(2 * mhz) + 1];
					mhz++;
				}
				}
				}
				//assert(mhz == nhz);

				// free
				free(exrecv);
				free(eyrecv);
				free(ezrecv);
				free(hxrecv);
				free(hyrecv);
				free(hzrecv);
			}
		}

		else {
			// === send ===

			// alloc
			int nx = iMax - iMin;
			int ny = jMax - jMin;
			int nz = kMax - kMin;
			int nex = (nx + 0) * (ny + 1) * (nz + 1);
			int ney = (ny + 0) * (nz + 1) * (nx + 1);
			int nez = (nz + 0) * (nx + 1) * (ny + 1);
			int nhx = (nx + 1) * (ny + 2) * (nz + 2);
			int nhy = (ny + 1) * (nz + 2) * (nx + 2);
			int nhz = (nz + 1) * (nx + 2) * (ny + 2);
			real_t *exsend = (real_t *)malloc(2 * nex * sizeof(real_t));
			real_t *eysend = (real_t *)malloc(2 * ney * sizeof(real_t));
			real_t *ezsend = (real_t *)malloc(2 * nez * sizeof(real_t));
			real_t *hxsend = (real_t *)malloc(2 * nhx * sizeof(real_t));
			real_t *hysend = (real_t *)malloc(2 * nhy * sizeof(real_t));
			real_t *hzsend = (real_t *)malloc(2 * nhz * sizeof(real_t));

			// Ex
			int64_t mex = 0;
			for (int i = iMin; i <  iMax; i++) {
			for (int j = jMin; j <= jMax; j++) {
			for (int k = kMin; k <= kMax; k++) {
				int64_t n = (ifreq * NN) + NA(i, j, k);
				exsend[(2 * mex) + 0] = Ex_r[n];
				exsend[(2 * mex) + 1] = Ex_i[n];
				mex++;
			}
			}
			}
			//assert(mex == nex);

			// Ey
			int64_t mey = 0;
			for (int i = iMin; i <= iMax; i++) {
			for (int j = jMin; j <  jMax; j++) {
			for (int k = kMin; k <= kMax; k++) {
				int64_t n = (ifreq * NN) + NA(i, j, k);
				eysend[(2 * mey) + 0] = Ey_r[n];
				eysend[(2 * mey) + 1] = Ey_i[n];
				mey++;
			}
			}
			}
			//assert(mey == ney);

			// Ez
			int64_t mez = 0;
			for (int i = iMin; i <= iMax; i++) {
			for (int j = jMin; j <= jMax; j++) {
			for (int k = kMin; k <  kMax; k++) {
				int64_t n = (ifreq * NN) + NA(i, j, k);
				ezsend[(2 * mez) + 0] = Ez_r[n];
				ezsend[(2 * mez) + 1] = Ez_i[n];
				mez++;
			}
			}
			}
			//assert(mez == nez);

			// Hx
			int64_t mhx = 0;
			for (int i = iMin - 0; i <= iMax; i++) {
			for (int j = jMin - 1; j <= jMax; j++) {
			for (int k = kMin - 1; k <= kMax; k++) {
				int64_t n = (ifreq * NN) + NA(i, j, k);
				hxsend[(2 * mhx) + 0] = Hx_r[n];
				hxsend[(2 * mhx) + 1] = Hx_i[n];
				mhx++;
			}
			}
			}
			//assert(mhx == nhx);

			// Hy
			int64_t mhy = 0;
			for (int i = iMin - 1; i <= iMax; i++) {
			for (int j = jMin - 0; j <= jMax; j++) {
			for (int k = kMin - 1; k <= kMax; k++) {
				int64_t n = (ifreq * NN) + NA(i, j, k);
				hysend[(2 * mhy) + 0] = Hy_r[n];
				hysend[(2 * mhy) + 1] = Hy_i[n];
				mhy++;
			}
			}
			}
			//assert(mhy == nhy);

			// Hz
			int64_t mhz = 0;
			for (int i = iMin - 1; i <= iMax; i++) {
			for (int j = jMin - 1; j <= jMax; j++) {
			for (int k = kMin - 0; k <= kMax; k++) {
				int64_t n = (ifreq * NN) + NA(i, j, k);
				hzsend[(2 * mhz) + 0] = Hz_r[n];
				hzsend[(2 * mhz) + 1] = Hz_i[n];
				mhz++;
			}
			}
			}
			//assert(mhz == nhz);

			// send
			isend[0]  = iMin;
			isend[1]  = iMax;
			isend[2]  = jMin;
			isend[3]  = jMax;
			isend[4]  = kMin;
			isend[5]  = kMax;
			isend[6]  = nex;
			isend[7]  = ney;
			isend[8]  = nez;
			isend[9]  = nhx;
			isend[10] = nhy;
			isend[11] = nhz;
			MPI_Send(isend, 12, MPI_INT, 0, 0, MPI_COMM_WORLD);
			MPI_Send(exsend, 2 * nex, MPI_REAL_T, 0, 0, MPI_COMM_WORLD);
			MPI_Send(eysend, 2 * ney, MPI_REAL_T, 0, 0, MPI_COMM_WORLD);
			MPI_Send(ezsend, 2 * nez, MPI_REAL_T, 0, 0, MPI_COMM_WORLD);
			MPI_Send(hxsend, 2 * nhx, MPI_REAL_T, 0, 0, MPI_COMM_WORLD);
			MPI_Send(hysend, 2 * nhy, MPI_REAL_T, 0, 0, MPI_COMM_WORLD);
			MPI_Send(hzsend, 2 * nhz, MPI_REAL_T, 0, 0, MPI_COMM_WORLD);

			// free
			free(exsend);
			free(eysend);
			free(ezsend);
			free(hxsend);
			free(hysend);
			free(hzsend);
		}

	}
#endif
}
