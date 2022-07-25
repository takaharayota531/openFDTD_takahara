// ofd.h
#ifndef _OFD_H_
#define _OFD_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <limits.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define PROGRAM "OpenFDTD"
#define VERSION_MAJOR (2)
#define VERSION_MINOR (7)
#define VERSION_BUILD (5)

#ifdef MAIN
#define EXTERN
#else
#define EXTERN extern
#endif

#define FN_log      "ofd.log"
#define FN_out      "ofd.out"
#define FN_geom3d_0 "geom3d.htm"
#define FN_ev2d_0   "ev2d.htm"
#define FN_ev3d_0   "ev3d.htm"
#define FN_geom3d_1 "geom.ev3"
#define FN_ev2d_1   "ev.ev2"
#define FN_ev3d_1   "ev.ev3"
#define FN_far0d    "far0d.log"
#define FN_far1d    "far1d.log"
#define FN_far2d    "far2d.log"
#define FN_near1d   "near1d.log"
#define FN_near2d   "near2d.log"
#define FN_feed     "feed.log"
#define FN_point    "point.log"

#define PI     (4.0*atan(1.0))
#define C      (2.99792458e8)
#define MU0    (4*PI*1e-7)
#define EPS0   (1/(C*C*MU0))
#define ETA0   (C*MU0)
#define EPS    (1e-6)
#define EPS2   (EPS*EPS)
#define DTOR   (PI/180)
#define PEC    (1)

#define NA(i,j,k) ((i)*Ni+(j)*Nj+(k)*Nk+N0)

#define EX(i,j,k) Ex[NA(i,j,k)]
#define EY(i,j,k) Ey[NA(i,j,k)]
#define EZ(i,j,k) Ez[NA(i,j,k)]
#define HX(i,j,k) Hx[NA(i,j,k)]
#define HY(i,j,k) Hy[NA(i,j,k)]
#define HZ(i,j,k) Hz[NA(i,j,k)]

#define IEX(i,j,k) iEx[NA(i,j,k)]
#define IEY(i,j,k) iEy[NA(i,j,k)]
#define IEZ(i,j,k) iEz[NA(i,j,k)]
#define IHX(i,j,k) iHx[NA(i,j,k)]
#define IHY(i,j,k) iHy[NA(i,j,k)]
#define IHZ(i,j,k) iHz[NA(i,j,k)]

#define NEX(i,j,k) ((i)*(Ny+1)*(Nz+1)+(j)*(Nz+1)+(k))
#define NEY(i,j,k) ((j)*(Nz+1)*(Nx+1)+(k)*(Nx+1)+(i))
#define NEZ(i,j,k) ((k)*(Nx+1)*(Ny+1)+(i)*(Ny+1)+(j))
#define NHX(i,j,k) ((i)*(Ny+2)*(Nz+2)+(j+1)*(Nz+2)+(k+1))
#define NHY(i,j,k) ((j)*(Nz+2)*(Nx+2)+(k+1)*(Nx+2)+(i+1))
#define NHZ(i,j,k) ((k)*(Nx+2)*(Ny+2)+(i+1)*(Ny+2)+(j+1))

#define CEX_r(f,i,j,k) cEx_r[(f*NEx)+NEX(i,j,k)]
#define CEY_r(f,i,j,k) cEy_r[(f*NEy)+NEY(i,j,k)]
#define CEZ_r(f,i,j,k) cEz_r[(f*NEz)+NEZ(i,j,k)]
#define CHX_r(f,i,j,k) cHx_r[(f*NHx)+NHX(i,j,k)]
#define CHY_r(f,i,j,k) cHy_r[(f*NHy)+NHY(i,j,k)]
#define CHZ_r(f,i,j,k) cHz_r[(f*NHz)+NHZ(i,j,k)]
#define CEX_i(f,i,j,k) cEx_i[(f*NEx)+NEX(i,j,k)]
#define CEY_i(f,i,j,k) cEy_i[(f*NEy)+NEY(i,j,k)]
#define CEZ_i(f,i,j,k) cEz_i[(f*NEz)+NEZ(i,j,k)]
#define CHX_i(f,i,j,k) cHx_i[(f*NHx)+NHX(i,j,k)]
#define CHY_i(f,i,j,k) cHy_i[(f*NHy)+NHY(i,j,k)]
#define CHZ_i(f,i,j,k) cHz_i[(f*NHz)+NHZ(i,j,k)]

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define NINT(l,d) ((int)((l) / (d) + 0.5))

#if defined(_DOUBLE)
#define real_t double
#else
#define real_t float
#endif

#if defined(_ID64)
#define id_t int64_t
#define MAXMATERIAL ((int64_t)1 << 47)
#elif defined(_ID32) || defined(_VECTOR)
#define id_t int
#define MAXMATERIAL ((int64_t)1 << 31)
#elif defined(_ID16)
#define id_t unsigned short
#define MAXMATERIAL (1 << 16)
#else
#define id_t unsigned char
#define MAXMATERIAL (1 << 8)
#endif

typedef struct {double r, i;} d_complex_t;

typedef struct {
	id_t   m;                 // material id = 0,1,2,...
	int    shape;             // shape, 1:rectangle, 2:sphere, ...
	double g[8];              // geometry [m]
} geometry_t;                 // geometry

typedef struct {
	int    type;              // 1:normal, 2:dispersion
	double epsr, esgm;        // Eps-r, E-sigma[S/m]
	double amur, msgm;        // Mu-r, M-sigma[1/Sm]
	double einf, ae, be, ce;  // dispersion parameters
	real_t edisp[3];          // dispersion factors;
} material_t;                 // material

typedef struct {
	char   dir;               // direction : X/Y/Z
	int    i, j, k;           // position index
	double dx, dy, dz;        // cell size
	double volt;              // voltage [V]
	double delay;             // delay time [sec]
	double z0;                // feed impedance [ohm]
} feed_t;                     // feed

EXTERN struct {
	double theta, phi;        // direction
	double ei[3], hi[3];      // E and H unit vector
	double ri[3], r0[3], ai;  // incidence vector and factor
	int    pol;               // polarization : 1=V, 2=H
} Planewave;

typedef struct {
	char   dir;               // direction : X/Y/Z
	int    i, j, k;           // position index
	double dx, dy, dz;        // cell size
} point_t;                    // point

typedef struct {
	char   dir;               // direction : X/Y/Z
	int    i, j, k;           // position index
	double dx, dy, dz;        // cell size
	double fctr;              // scheme factor [1/(m*m)]
	double e, esum;           // scheme term
} inductor_t;                 // inductor

typedef struct {
	char   dir;               // direction : X/Y/Z/V/H
	int    div;               // division of angle 360 deg
	double angle;             // V/H constant angle [deg]
} far1d_t;                    // far1d field

typedef struct {
	int    divtheta, divphi;  // division in theta and phi
} far2d_t;                    // far2d field

typedef struct {
	char   cmp[3];            // component : E/Ex/Ey/Ez/H/Hx/Hy/Hz
	char   dir;               // direction : X/Y/Z
	double pos1, pos2;        // position
	int    id1, id2;          // node index
} near1d_t;                   // near1d field

typedef struct {
	char   cmp[3];            // component : E/Ex/Ey/Ez/H/Hx/Hy/Hz
	char   dir;               // direction : X/Y/Z
	double pos0;              // position
	int    id0;               // node index
} near2d_t;                   // near2d field

typedef struct {
	int    i, j, k;           // index
	int    i1, j1, k1;        // inner index
	real_t f, g;              // factor
} mur_t;                      // Mur

typedef struct {
	int    i, j, k;           // index
	id_t   m;                 // boundary material number
} pml_t;                      // PML

typedef struct {
	int     i, j, k;         // index
	real_t  f1, f2, f3;      // factor
} dispersion_t;              // dispersion

typedef struct {
	int    db;                // dB (0/1)
	int    user;              // 0/1 : auto/user
	double min, max;          // min, max
	int    div;               // division
} scale_t;                    // scale

EXTERN struct {
	int    l;                 // layer
	double m;                 // order
	double r0;                // reflection
} cPML;                       // PML

EXTERN struct {
	int    maxiter;           // max iteraions
	int    nout;              // output interval
	double converg;           // convergence
} Solver;                     // solver

EXTERN int          runMode;                 // 0=solver+post, 1=solver, 2=post
EXTERN int          commSize, commRank;      // MPI : number of processes, my rank

EXTERN char         Title[256];              // title

EXTERN int          Nx, Ny, Nz;              // number of cells
EXTERN int          iMin, iMax, jMin, jMax, kMin, kMax;  // non-MPI : 0, Nx, 0, Ny, 0, Nz
EXTERN int          *iProc;
EXTERN double       *Xn, *Yn, *Zn;           // node
EXTERN double       *Xc, *Yc, *Zc;           // cell center

EXTERN int64_t      NMaterial;               // number of materials
EXTERN material_t   *Material;               // material

EXTERN int64_t      NGeometry;               // number of geometries
EXTERN geometry_t   *Geometry;               // geometry

EXTERN int          NFeed;                   // number of feeds
EXTERN feed_t       *Feed;                   // feed
EXTERN double       rFeed;                   // internal resistor
 
EXTERN int          IPlanewave;              // 0/1 : plane wave ?

EXTERN int          NPoint;                  // number of points
EXTERN point_t      *Point;                  // point

EXTERN int          NInductor;               // number of inductors
EXTERN inductor_t   *Inductor;               // inductor

EXTERN int          iABC;                    // ABC: 0=Mur, 1=PML
EXTERN int          PBCx, PBCy, PBCz;        // PBC (0/1)

EXTERN real_t       *Ex, *Ey, *Ez;           // E
EXTERN real_t       *Hx, *Hy, *Hz;           // H

EXTERN id_t         *iEx, *iEy, *iEz;        // material ID of E
EXTERN id_t         *iHx, *iHy, *iHz;        // material ID of H
EXTERN real_t       *C1, *C2, *C3, *C4;      // E factor
EXTERN real_t       *D1, *D2, *D3, *D4;      // H factor

#ifdef _VECTOR
EXTERN real_t       *K1Ex, *K2Ex, *K3Ex, *K4Ex;
EXTERN real_t       *K1Ey, *K2Ey, *K3Ey, *K4Ey;
EXTERN real_t       *K1Ez, *K2Ez, *K3Ez, *K4Ez;
EXTERN real_t       *K1Hx, *K2Hx, *K3Hx, *K4Hx;
EXTERN real_t       *K1Hy, *K2Hy, *K3Hy, *K4Hy;
EXTERN real_t       *K1Hz, *K2Hz, *K3Hz, *K4Hz;
#endif

EXTERN int64_t      Ni, Nj, Nk, N0;          // array index
EXTERN int64_t      NN;                      // array size

EXTERN real_t       *RXn, *RYn, *RZn;        // (C * dt) / d (node)
EXTERN real_t       *RXc, *RYc, *RZc;        // (C * dt) / d (cell center)

EXTERN int64_t      numMurHx, numMurHy, numMurHz;
EXTERN mur_t        *fMurHx, *fMurHy, *fMurHz;

EXTERN real_t       *Exy, *Exz, *Eyz, *Eyx, *Ezx, *Ezy;
EXTERN real_t       *Hxy, *Hxz, *Hyz, *Hyx, *Hzx, *Hzy;
EXTERN int64_t      numPmlEx, numPmlEy, numPmlEz;
EXTERN int64_t      numPmlHx, numPmlHy, numPmlHz;
EXTERN pml_t        *fPmlEx, *fPmlEy, *fPmlEz;
EXTERN pml_t        *fPmlHx, *fPmlHy, *fPmlHz;
EXTERN real_t       *gPmlXn, *gPmlYn, *gPmlZn;
EXTERN real_t       *gPmlXc, *gPmlYc, *gPmlZc;
EXTERN real_t       *rPmlE, *rPmlH;

EXTERN real_t       *DispersionEx, *DispersionEy, *DispersionEz;
EXTERN dispersion_t *mDispersionEx, *mDispersionEy, *mDispersionEz;
EXTERN int64_t      numDispersionEx, numDispersionEy, numDispersionEz;

EXTERN double       Dt, Tw;                  // time step, pulse width [sec]
EXTERN int          NFreq1, NFreq2;          // number of frequencies
EXTERN double       *Freq1, *Freq2;          // frequency

EXTERN int          Ntime;                   // time steps

EXTERN size_t       Iter_size;               // size of Eiter/Hiter
EXTERN int          Niter;                   // number of E/H average of iteration
EXTERN double       *Eiter, *Hiter;          // E/H average of iteration

EXTERN size_t       Feed_size;               // size of Vfeed/Ifeed
EXTERN double       *VFeed, *IFeed;          // V and I waveform
EXTERN d_complex_t  *Zin;                    // input impedance
EXTERN double       *Ref;                    // reflection
EXTERN double       *Pin[2];                 // input feed power (for far field gain, matching loss)

EXTERN size_t       Point_size, Spara_size;  // size of EPoint, Spara
EXTERN double       *VPoint;                 // V waveform at points
EXTERN d_complex_t  *Spara;                  // S-parameter

EXTERN d_complex_t  *cEdft, *cHdft;          // DFT factor
EXTERN d_complex_t  *Fnorm;                  // normalization DFT factor

EXTERN int64_t      NEx, NEy, NEz, NHx, NHy, NHz;
EXTERN real_t       *Ex_r, *Ex_i, *Ey_r, *Ey_i, *Ez_r, *Ez_i;
EXTERN real_t       *Hx_r, *Hx_i, *Hy_r, *Hy_i, *Hz_r, *Hz_i;
EXTERN real_t       *cEx_r, *cEx_i, *cEy_r, *cEy_i, *cEz_r, *cEz_i;
EXTERN real_t       *cHx_r, *cHx_i, *cHy_r, *cHy_i, *cHz_r, *cHz_i;

EXTERN int64_t      NGline;                  // number of geometry lines
EXTERN double       (*Gline)[2][3];          // geometery lines
EXTERN id_t         *MGline;                 // material-id of geometry lines

// post

EXTERN int          MatchingLoss;            // include matching loss (0/1)

EXTERN int          Piter;                   // iteration (0/1)
EXTERN int          Pfeed;                   // feed (0/1)
EXTERN int          Ppoint;                  // point (0/1)

EXTERN int          IFreq[6];                // frequency char.s (0/1)
EXTERN scale_t      FreqScale[6];            // scale
EXTERN int          Freqdiv;                 // x-axis division

EXTERN int          IFar0d;                  // frequency char of far0d (0/1)
EXTERN double       Far0d[2];                // theta, phi [deg]
EXTERN scale_t      Far0dScale;              // scale

EXTERN int          NFar1d;                  // number of far1d
EXTERN far1d_t      *Far1d;                  // parameters
EXTERN int          Far1dStyle;              // style (0/1)
EXTERN int          Far1dComp[3];            // component (0/1)
EXTERN int          Far1dNorm;               // normalize (0/1)
EXTERN scale_t      Far1dScale;              // scale

EXTERN int          NFar2d;                  // number of far2d
EXTERN far2d_t      Far2d;                   // parameters
EXTERN int          Far2dComp[7];            // component (0/1)
EXTERN scale_t      Far2dScale;              // scale
EXTERN double       Far2dObj;                // size pf objects

EXTERN int          NNear1d;                 // number of near1d
EXTERN near1d_t     *Near1d;                 // parameters
EXTERN scale_t      Near1dScale;             // scale
EXTERN int          Near1dNoinc;             // exlude planewave incidence (0/1)
EXTERN int          *LNear1d;                // number of near1d data
EXTERN size_t       Near1d_size;             // size of near1d data
EXTERN d_complex_t  *Near1dEx, *Near1dEy, *Near1dEz;  // near1d field E
EXTERN d_complex_t  *Near1dHx, *Near1dHy, *Near1dHz;  // near1d field H

EXTERN int          NNear2d;                 // number of near2d
EXTERN near2d_t     *Near2d;                 // parameters
EXTERN int          Near2dDim[2];            // plot 2d/3d (0/1)
EXTERN int          Near2dFrame;             // animation frames
EXTERN scale_t      Near2dScale;             // scale
EXTERN int          Near2dContour;           // contour (0/1/2/3)
EXTERN int          Near2dObj;               // object (0/1)
EXTERN int          Near2dNoinc;             // exlude planewave incidence (0/1)
EXTERN int          Near2dIzoom;             // zoom ? (0/1)
EXTERN double       Near2dHzoom[2];          // zoom (horizontal)
EXTERN double       Near2dVzoom[2];          // zoom (vertical)
EXTERN int          *LNear2d;                // number of near2d data
EXTERN size_t       Near2d_size;             // size of near2d data
EXTERN d_complex_t  *Near2dEx, *Near2dEy, *Near2dEz;  // near2d field E
EXTERN d_complex_t  *Near2dHx, *Near2dHy, *Near2dHz;  // near2d field H

EXTERN int          Width2d, Height2d;       // window size (2D)
EXTERN int          Font2d;                  // font size (2D)
EXTERN int          Fontname2d;              // font name (0/1/2) (2D)
EXTERN int          Width3d, Height3d;       // window size (3D)
EXTERN int          Font3d;                  // font size (3D)
EXTERN double       Theta3d, Phi3d;          // view direction (3D)

#ifdef __cplusplus
}
#endif

#endif  // _OFD_H_
