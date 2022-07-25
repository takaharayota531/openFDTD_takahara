/*
comm_cuda.cu (CUDA + MPI)
*/

#ifdef _MPI
#include <mpi.h>
#endif

#include "ofd.h"
#include "ofd_mpi.h"
#include "ofd_cuda.h"
#include "ofd_prototype.h"


// share boundary H : copy D2H + comm + copy H2D
void comm_cuda_boundary()
{
#ifdef _MPI
	MPI_Status status;
	const int tag = 0;
	const size_t size_hy = Length_Hy * sizeof(real_t);
	const size_t size_hz = Length_Hz * sizeof(real_t);
	const int count = (int)(Length_Hy + Length_Hz);

	// -X boundary
	if (commRank > 0) {
		// copy device to host
		cuda_memcpy(GPU, sendBuf,             Hy + Offset_Hy[1], size_hy, cudaMemcpyDeviceToHost);
		cuda_memcpy(GPU, sendBuf + Length_Hy, Hz + Offset_Hz[1], size_hz, cudaMemcpyDeviceToHost);

		// MPI
		MPI_Sendrecv(sendBuf, count, MPI_REAL_T, commRank - 1, tag,
		             recvBuf, count, MPI_REAL_T, commRank - 1, tag, MPI_COMM_WORLD, &status);

		// copy host to device
		cuda_memcpy(GPU, Hy + Offset_Hy[0], recvBuf,             size_hy, cudaMemcpyHostToDevice);
		cuda_memcpy(GPU, Hz + Offset_Hz[0], recvBuf + Length_Hy, size_hz, cudaMemcpyHostToDevice);
	}

	// +X boundary
	if (commRank < commSize - 1) {
		// copy device to host
		cuda_memcpy(GPU, sendBuf,             Hy + Offset_Hy[2], size_hy, cudaMemcpyDeviceToHost);
		cuda_memcpy(GPU, sendBuf + Length_Hy, Hz + Offset_Hz[2], size_hz, cudaMemcpyDeviceToHost);

		// MPI
		MPI_Sendrecv(sendBuf, count, MPI_REAL_T, commRank + 1, tag,
		             recvBuf, count, MPI_REAL_T, commRank + 1, tag, MPI_COMM_WORLD, &status);

		// copy host to device
		cuda_memcpy(GPU, Hy + Offset_Hy[3], recvBuf,             size_hy, cudaMemcpyHostToDevice);
		cuda_memcpy(GPU, Hz + Offset_Hz[3], recvBuf + Length_Hy, size_hz, cudaMemcpyHostToDevice);
	}
#endif
}


// PBC on +/- X boundaries : copy D2H + comm + copy H2D
void comm_cuda_pbcx()
{
#ifdef _MPI
	MPI_Status status;
	const int tag = 0;
	const size_t size_hy = Length_Hy * sizeof(real_t);
	const size_t size_hz = Length_Hz * sizeof(real_t);
	const int count = (int)(Length_Hy + Length_Hz);

	// -X boundary
	if (commRank == 0) {
		// copy to buffer
		cuda_memcpy(GPU, sendBuf,             Hy + Offset_Hy[1], size_hy, cudaMemcpyDeviceToHost);
		cuda_memcpy(GPU, sendBuf + Length_Hy, Hz + Offset_Hz[1], size_hz, cudaMemcpyDeviceToHost);

		// MPI
		int dst = commSize - 1;
		MPI_Sendrecv(sendBuf, count, MPI_REAL_T, dst, tag,
		             recvBuf, count, MPI_REAL_T, dst, tag, MPI_COMM_WORLD, &status);

		// copy from buffer
		cuda_memcpy(GPU, Hy + Offset_Hy[0], recvBuf,             size_hy, cudaMemcpyHostToDevice);
		cuda_memcpy(GPU, Hz + Offset_Hz[0], recvBuf + Length_Hy, size_hz, cudaMemcpyHostToDevice);
	}

	// +X boundary
	else if (commRank == commSize - 1) {
		// copy to buffer
		cuda_memcpy(GPU, sendBuf,             Hy + Offset_Hy[2], size_hy, cudaMemcpyDeviceToHost);
		cuda_memcpy(GPU, sendBuf + Length_Hy, Hz + Offset_Hz[2], size_hz, cudaMemcpyDeviceToHost);

		// MPI
		int dst = 0;
		MPI_Sendrecv(sendBuf, count, MPI_REAL_T, dst, tag,
		             recvBuf, count, MPI_REAL_T, dst, tag, MPI_COMM_WORLD, &status);

		// copy from buffer
		cuda_memcpy(GPU, Hy + Offset_Hy[3], recvBuf,             size_hy, cudaMemcpyHostToDevice);
		cuda_memcpy(GPU, Hz + Offset_Hz[3], recvBuf + Length_Hy, size_hz, cudaMemcpyHostToDevice);
	}
#endif
}
