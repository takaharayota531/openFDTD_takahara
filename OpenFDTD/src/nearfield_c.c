/*
nearfield_c.c

get near field (complex)
*/

#include "ofd.h"
#include "complex.h"

static inline d_complex_t average2(d_complex_t c1, d_complex_t c2)
{
	return d_complex(
		(c1.r + c2.r) / 2,
		(c1.i + c2.i) / 2);
}

static inline d_complex_t average4(d_complex_t c1, d_complex_t c2, d_complex_t c3, d_complex_t c4)
{
	return d_complex(
		(c1.r + c2.r + c3.r + c4.r) / 4,
		(c1.i + c2.i + c3.i + c4.i) / 4);
}

// E at node
void NodeE_c(int ifreq, int i, int j, int k, d_complex_t *cex, d_complex_t *cey, d_complex_t *cez)
{
	d_complex_t c1, c2;

	if      (i <= 0) {
		i = 0;
		c1.r = CEX_r(ifreq, i + 0, j, k);
		c1.i = CEX_i(ifreq, i + 0, j, k);
		c2.r = CEX_r(ifreq, i + 1, j, k);
		c2.i = CEX_i(ifreq, i + 1, j, k);
		*cex = d_sub(d_rmul(1.5, c1), d_rmul(0.5, c2));
	}
	else if (i >= Nx) {
		i = Nx;
		c1.r = CEX_r(ifreq, i - 1, j, k);
		c1.i = CEX_i(ifreq, i - 1, j, k);
		c2.r = CEX_r(ifreq, i - 2, j, k);
		c2.i = CEX_i(ifreq, i - 2, j, k);
		*cex = d_sub(d_rmul(1.5, c1), d_rmul(0.5, c2));
	}
	else {
		c1.r = CEX_r(ifreq, i + 0, j, k);
		c1.i = CEX_i(ifreq, i + 0, j, k);
		c2.r = CEX_r(ifreq, i - 1, j, k);
		c2.i = CEX_i(ifreq, i - 1, j, k);
		*cex = average2(c1, c2);
	}

	if      (j <= 0) {
		j = 0;
		c1.r = CEY_r(ifreq, i, j + 0, k);
		c1.i = CEY_i(ifreq, i, j + 0, k);
		c2.r = CEY_r(ifreq, i, j + 1, k);
		c2.i = CEY_i(ifreq, i, j + 1, k);
		*cey = d_sub(d_rmul(1.5, c1), d_rmul(0.5, c2));
	}
	else if (j >= Ny) {
		j = Ny;
		c1.r = CEY_r(ifreq, i, j - 1, k);
		c1.i = CEY_i(ifreq, i, j - 1, k);
		c2.r = CEY_r(ifreq, i, j - 2, k);
		c2.i = CEY_i(ifreq, i, j - 2, k);
		*cey = d_sub(d_rmul(1.5, c1), d_rmul(0.5, c2));
	}
	else {
		c1.r = CEY_r(ifreq, i, j + 0, k);
		c1.i = CEY_i(ifreq, i, j + 0, k);
		c2.r = CEY_r(ifreq, i, j - 1, k);
		c2.i = CEY_i(ifreq, i, j - 1, k);
		*cey = average2(c1, c2);
	}

	if      (k <= 0) {
		k = 0;
		c1.r = CEZ_r(ifreq, i, j, k + 0);
		c1.i = CEZ_i(ifreq, i, j, k + 0);
		c2.r = CEZ_r(ifreq, i, j, k + 1);
		c2.i = CEZ_i(ifreq, i, j, k + 1);
		*cez = d_sub(d_rmul(1.5, c1), d_rmul(0.5, c2));
	}
	else if (k >= Nz) {
		k = Nz;
		c1.r = CEZ_r(ifreq, i, j, k - 1);
		c1.i = CEZ_i(ifreq, i, j, k - 1);
		c2.r = CEZ_r(ifreq, i, j, k - 2);
		c2.i = CEZ_i(ifreq, i, j, k - 2);
		*cez = d_sub(d_rmul(1.5, c1), d_rmul(0.5, c2));
	}
	else {
		c1.r = CEZ_r(ifreq, i, j, k + 0);
		c1.i = CEZ_i(ifreq, i, j, k + 0);
		c2.r = CEZ_r(ifreq, i, j, k - 1);
		c2.i = CEZ_i(ifreq, i, j, k - 1);
		*cez = average2(c1, c2);
	}
}

// H at node
void NodeH_c(int ifreq, int i, int j, int k, d_complex_t *chx, d_complex_t *chy, d_complex_t *chz)
{
	d_complex_t c1, c2, c3, c4;

	c1.r = CHX_r(ifreq, i,     j,     k    );
	c1.i = CHX_i(ifreq, i,     j,     k    );
	c2.r = CHX_r(ifreq, i,     j - 1, k    );
	c2.i = CHX_i(ifreq, i,     j - 1, k    );
	c3.r = CHX_r(ifreq, i,     j,     k - 1);
	c3.i = CHX_i(ifreq, i,     j,     k - 1);
	c4.r = CHX_r(ifreq, i,     j - 1, k - 1);
	c4.i = CHX_i(ifreq, i,     j - 1, k - 1);
	*chx = average4(c1, c2, c3, c4);

	c1.r = CHY_r(ifreq, i,     j,     k    );
	c1.i = CHY_i(ifreq, i,     j,     k    );
	c2.r = CHY_r(ifreq, i,     j,     k - 1);
	c2.i = CHY_i(ifreq, i,     j,     k - 1);
	c3.r = CHY_r(ifreq, i - 1, j,     k   );
	c3.i = CHY_i(ifreq, i - 1, j,     k   );
	c4.r = CHY_r(ifreq, i - 1, j,     k - 1);
	c4.i = CHY_i(ifreq, i - 1, j,     k - 1);
	*chy = average4(c1, c2, c3, c4);

	c1.r = CHZ_r(ifreq, i,     j,     k    );
	c1.i = CHZ_i(ifreq, i,     j,     k    );
	c2.r = CHZ_r(ifreq, i - 1, j,     k    );
	c2.i = CHZ_i(ifreq, i - 1, j,     k    );
	c3.r = CHZ_r(ifreq, i,     j - 1, k    );
	c3.i = CHZ_i(ifreq, i,     j - 1, k    );
	c4.r = CHZ_r(ifreq, i - 1, j - 1, k    );
	c4.i = CHZ_i(ifreq, i - 1, j - 1, k    );
	*chz = average4(c1, c2, c3, c4);
}
