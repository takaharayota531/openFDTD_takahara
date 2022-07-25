/*
dftField.h

add E and H (complex)
*/

static inline void dftField(
	d_complex_t *cex, d_complex_t *cey, d_complex_t *cez, d_complex_t *chx, d_complex_t *chy, d_complex_t *chz,
	real_t ex, real_t ey, real_t ez, real_t hx, real_t hy, real_t hz,
	d_complex_t fe, d_complex_t fh)
{
	// E
	cex->r += fe.r * ex;
	cex->i += fe.i * ex;
	cey->r += fe.r * ey;
	cey->i += fe.i * ey;
	cez->r += fe.r * ez;
	cez->i += fe.i * ez;

	// H
	chx->r += fh.r * hx;
	chx->i += fh.i * hx;
	chy->r += fh.r * hy;
	chy->i += fh.i * hy;
	chz->r += fh.r * hz;
	chz->i += fh.i * hz;
}
