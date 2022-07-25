// ofd_mpi.h
#ifndef _OFD_MPI_H_
#define _OFD_MPI_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MAIN
#define EXTERN
#else
#define EXTERN extern
#endif

#ifdef _DOUBLE
#define MPI_REAL_T MPI_DOUBLE
#else
#define MPI_REAL_T MPI_FLOAT
#endif

//EXTERN int       commSize, commRank;  // -> ofd.h

EXTERN real_t      *sendBuf, *recvBuf;
EXTERN int64_t     Offset_Hy[4], Offset_Hz[4];
EXTERN int64_t     Length_Hy, Length_Hz;

EXTERN int         *l_LNear1d;          // number of near1d data
EXTERN d_complex_t *l_Near1dEx, *l_Near1dEy, *l_Near1dEz;  // near1d E
EXTERN d_complex_t *l_Near1dHx, *l_Near1dHy, *l_Near1dHz;  // near1d H

EXTERN int         *l_LNear2d;          // number of near2d data
EXTERN d_complex_t *l_Near2dEx, *l_Near2dEy, *l_Near2dEz;  // near2d E
EXTERN d_complex_t *l_Near2dHx, *l_Near2dHy, *l_Near2dHz;  // near2d H

#ifdef __cplusplus
}
#endif

#endif		// _OFD_MPI_H_
