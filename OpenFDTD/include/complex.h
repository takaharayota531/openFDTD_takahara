/*
complex.h
*/

#ifndef _COMPLEX_H_
#define _COMPLEX_H_

#if defined(__GNUC__) && !defined(__NEC__) && !defined(__FUJITSU) && !defined(__CLANG_FUJITSU)
//#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <math.h>

static inline d_complex_t d_complex(double r, double i)
{
	d_complex_t z;

	z.r = r;
	z.i = i;

	return z;
}

static inline d_complex_t d_add(d_complex_t a, d_complex_t b)
{
	return d_complex(a.r + b.r, a.i + b.i);
}

static inline d_complex_t d_sub(d_complex_t a, d_complex_t b)
{
	return d_complex(a.r - b.r, a.i - b.i);
}

static inline d_complex_t d_mul(d_complex_t a, d_complex_t b)
{
	return d_complex((a.r * b.r) - (a.i * b.i), (a.r * b.i) + (a.i * b.r));
}

static inline d_complex_t d_div(d_complex_t a, d_complex_t b)
{
	if ((fabs(b.r) <= 0) && (fabs(b.i) <= 0)) return d_complex(0, 0);
	return d_complex(((a.r * b.r) + (a.i * b.i)) / ((b.r * b.r) + (b.i * b.i)),
	                 ((a.i * b.r) - (a.r * b.i)) / ((b.r * b.r) + (b.i * b.i)));
}

static inline d_complex_t d_rmul(double r, d_complex_t z)
{
	return d_complex(r * z.r, r * z.i);
}

static inline d_complex_t d_add3(d_complex_t a, d_complex_t b, d_complex_t c)
{
	return d_complex(a.r + b.r + c.r, a.i + b.i + c.i);
}

static inline d_complex_t d_add4(d_complex_t a, d_complex_t b, d_complex_t c, d_complex_t d)
{
	return d_complex(a.r + b.r + c.r + d.r, a.i + b.i + c.i + d.i);
}

static inline double d_abs(d_complex_t z)
{
	return sqrt((z.r * z.r) + (z.i * z.i));
}

static inline double d_norm(d_complex_t z)
{
	return ((z.r * z.r) + (z.i * z.i));
}

static inline double d_rad(d_complex_t z)
{
	return atan2(z.i, z.r);
}

static inline double d_deg(d_complex_t z)
{
	return atan2(z.i, z.r) * 45.0 / atan(1.0);
}

static inline d_complex_t d_exp(double x)
{
	return d_complex(cos(x), sin(x));
}

static inline d_complex_t d_sqrt(d_complex_t z)
{
	return d_rmul(sqrt(d_abs(z)), d_exp(0.5 * atan2(z.i, z.r)));
}

static inline d_complex_t d_inv(d_complex_t z)
{
	return d_div(d_complex(1.0, 0.0), z);
}

#endif  // _COMPLEX_H_
