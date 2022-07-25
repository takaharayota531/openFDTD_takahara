/*
comm.c (MPI)

OpenMOM + OpenFDTD
*/

#include <stdlib.h>
#include <stdio.h>
#ifdef _MPI
#include <mpi.h>
#endif

typedef struct {int xy, i, j, on;} segment_t;


// initialize
void mpi_init(int argc, char **argv, int *comm_size, int *comm_rank)
{
#ifdef _MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, comm_rank);
#endif
}


// close
void mpi_close(void)
{
#ifdef _MPI
	MPI_Finalize();
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
#endif  // _MPI
}


// communicate result
void comm_result(int comm_size, int comm_rank, double fminbest, int nseg, segment_t segbest[])
{
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Status status;

	if (!comm_rank) {
		// root
		double *comm_fmin = (double *)malloc(comm_size * sizeof(double));

		// (tag-1) receive fmin
		comm_fmin[0] = fminbest;
		for (int rank = 1; rank < comm_size; rank++) {
			MPI_Recv(&comm_fmin[rank], 1, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD, &status);
		}

		// find best process
		int imin = -1;
		double rmin = 1e10;
		for (int rank = 0; rank < comm_size; rank++) {
			//printf("%d %f\n", rank, comm_fmin[rank]);
			if (comm_fmin[rank] < rmin) {
				imin = rank;
				rmin = comm_fmin[rank];
			}
		}
		printf("fmin = %f\n", rmin); fflush(stdout);

		for (int rank = 1; rank < comm_size; rank++) {
			// (tag-2) send best flag=0/1
			int flag = (rank == imin) ? 1 : 0;
			MPI_Send(&flag, 1, MPI_INT, rank, 2, MPI_COMM_WORLD);

			// (tag-3) flag=1 : receive best segment
			if (flag) {
				int *recvbuf = (int *)malloc(4 * nseg * sizeof(int));
				MPI_Recv(recvbuf, 4 * nseg, MPI_INT, rank, 3, MPI_COMM_WORLD, &status);
				for (int iseg = 0; iseg < nseg; iseg++) {
					segbest[iseg].xy = recvbuf[(4 * iseg) + 0];
					segbest[iseg].i  = recvbuf[(4 * iseg) + 1];
					segbest[iseg].j  = recvbuf[(4 * iseg) + 2];
					segbest[iseg].on = recvbuf[(4 * iseg) + 3];
				}
				free(recvbuf);
			}
		}
		free(comm_fmin);
	}
	else {
		// non-root

		// (tag-1) send fmin
		MPI_Send(&fminbest, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

		// (tag-2) receive best flag=0/1
		int flag = 0;
		MPI_Recv(&flag, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);

		// (tag-3) flag=1 : send best segment
		if (flag) {
			int *sendbuf = (int *)malloc(4 * nseg * sizeof(int));
			for (int iseg = 0; iseg < nseg; iseg++) {
				sendbuf[(4 * iseg) + 0] = segbest[iseg].xy;
				sendbuf[(4 * iseg) + 1] = segbest[iseg].i;
				sendbuf[(4 * iseg) + 2] = segbest[iseg].j;
				sendbuf[(4 * iseg) + 3] = segbest[iseg].on;
			}
			MPI_Send(sendbuf, 4 * nseg, MPI_INT, 0, 3, MPI_COMM_WORLD);
			free(sendbuf);
		}
	}
#endif
}
