/*
dftNear2d.cu (=../src/dftNear2d.c)

DFT of near-2d
*/

#include "ofd.h"
#include "ofd_prototype.h"

void dftNear2d(int itime, int *lng,
	d_complex_t *cex, d_complex_t *cey, d_complex_t *cez,
	d_complex_t *chx, d_complex_t *chy, d_complex_t *chz)
{
	if ((NNear2d <= 0) || (NFreq2 <= 0)) return;

	int64_t adr1 = 0;
	int64_t adr2 = NFreq2 * itime;
	for (int m = 0; m < NNear2d; m++) {
		for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
			if      (Near2d[m].dir == 'X') {
				dftNear2dX(Near2d[m].id0, adr1, adr2 + ifreq,
					cex, cey, cez, chx, chy, chz);
			}
			else if (Near2d[m].dir == 'Y') {
				dftNear2dY(Near2d[m].id0, adr1, adr2 + ifreq,
					cex, cey, cez, chx, chy, chz);
			}
			else if (Near2d[m].dir == 'Z') {
				dftNear2dZ(Near2d[m].id0, adr1, adr2 + ifreq,
					cex, cey, cez, chx, chy, chz);
			}
			adr1 += lng[m];
		}
	}
}
