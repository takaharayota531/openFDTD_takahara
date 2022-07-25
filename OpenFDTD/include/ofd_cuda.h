// ofd_cuda.h
#ifndef _OFD_CUDA_H_
#define _OFD_CUDA_H_

#ifdef MAIN
#define EXTERN
#else
#define EXTERN extern
#endif

#define LA(p,i,j,k) ((i)*((p)->Ni)+(j)*((p)->Nj)+(k)*((p)->Nk)+((p)->N0))
#define CEIL(n,d) (((n) + (d) - 1) / (d))

// parameter
typedef struct {
	int64_t Ni, Nj, Nk, N0;
	int     Nx, Ny, Nz;
	int     iMin, iMax, jMin, jMax, kMin, kMax;
	int     NFeed, IPlanewave;
	real_t  ei[3], hi[3], r0[3], ri[3], ai, dt;
} param_t;

EXTERN int          GPU;
EXTERN int          UM;

// constant memory
EXTERN param_t h_Param;
__constant__ param_t d_Param;

// execution configuration
EXTERN dim3         updateBlock;
EXTERN int          dispersionBlock;
EXTERN dim3         sumGrid, sumBlock;
EXTERN int          murBlock;
EXTERN int          pmlBlock;
EXTERN int          pbcBlock;
EXTERN int          near1dBlock;
EXTERN dim3         near2dBlock;

// host memory
EXTERN real_t       *h_Xn, *h_Yn, *h_Zn;
EXTERN real_t       *h_Xc, *h_Yc, *h_Zc;

// device memory
EXTERN real_t       *d_Xn, *d_Yn, *d_Zn;
EXTERN real_t       *d_Xc, *d_Yc, *d_Zc;
EXTERN real_t       *d_RXn, *d_RYn, *d_RZn;
EXTERN real_t       *d_RXc, *d_RYc, *d_RZc;
EXTERN id_t         *d_iEx, *d_iEy, *d_iEz;
EXTERN id_t         *d_iHx, *d_iHy, *d_iHz;
EXTERN real_t       *d_C1, *d_C2, *d_C3, *d_C4;
EXTERN real_t       *d_D1, *d_D2, *d_D3, *d_D4;
EXTERN real_t       *d_DispersionEx, *d_DispersionEy, *d_DispersionEz;
EXTERN dispersion_t *d_mDispersionEx, *d_mDispersionEy, *d_mDispersionEz;
EXTERN mur_t        *d_fMurHx, *d_fMurHy, *d_fMurHz;
EXTERN pml_t        *d_fPmlEx, *d_fPmlEy, *d_fPmlEz;
EXTERN pml_t        *d_fPmlHx, *d_fPmlHy, *d_fPmlHz;
EXTERN real_t       *d_gPmlXn, *d_gPmlYn, *d_gPmlZn;
EXTERN real_t       *d_gPmlXc, *d_gPmlYc, *d_gPmlZc;
EXTERN real_t       *d_rPmlE, *d_rPmlH;
EXTERN feed_t       *d_Feed;
EXTERN double       *d_VFeed, *d_IFeed;
EXTERN inductor_t   *d_Inductor;
EXTERN point_t      *d_Point;
EXTERN double       *d_VPoint;
EXTERN real_t       *h_sumE, *h_sumH;
EXTERN real_t       *d_sumE, *d_sumH;
EXTERN d_complex_t  *d_Near1dEx, *d_Near1dEy, *d_Near1dEz;
EXTERN d_complex_t  *d_Near1dHx, *d_Near1dHy, *d_Near1dHz;
EXTERN d_complex_t  *d_Near2dEx, *d_Near2dEy, *d_Near2dEz;
EXTERN d_complex_t  *d_Near2dHx, *d_Near2dHy, *d_Near2dHz;
EXTERN real_t       *d_Ex_r, *d_Ex_i, *d_Ey_r, *d_Ey_i, *d_Ez_r, *d_Ez_i;
EXTERN real_t       *d_Hx_r, *d_Hx_i, *d_Hy_r, *d_Hy_i, *d_Hz_r, *d_Hz_i;

#endif		// _OFD_CUDA_H_
