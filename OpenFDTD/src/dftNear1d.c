/*
dftNear1d.c

DFT of near-1d
*/

#include "ofd.h"
#include "ofd_prototype.h"

void dftNear1d(int itime, int *lng,
	d_complex_t *cex, d_complex_t *cey, d_complex_t *cez,
	d_complex_t *chx, d_complex_t *chy, d_complex_t *chz)
{
	if ((NNear1d <= 0) || (NFreq2 <= 0)) return;

	int64_t adr1 = 0;
	int64_t adr2 = NFreq2 * itime;
	for (int m = 0; m < NNear1d; m++) {
		for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
			if      (Near1d[m].dir == 'X') {
				dftNear1dX(Near1d[m].id1, Near1d[m].id2, adr1, adr2 + ifreq,
					cex, cey, cez, chx, chy, chz);
			}
			else if (Near1d[m].dir == 'Y') {
				dftNear1dY(Near1d[m].id1, Near1d[m].id2, adr1, adr2 + ifreq,
					cex, cey, cez, chx, chy, chz);
			}
			else if (Near1d[m].dir == 'Z') {
				dftNear1dZ(Near1d[m].id1, Near1d[m].id2, adr1, adr2 + ifreq,
					cex, cey, cez, chx, chy, chz);
			}
			adr1 += lng[m];
		}
	}
}
