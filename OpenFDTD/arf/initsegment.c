/*
initsegment.c

OpenMOM + OpenFDTD
*/

#include <stdint.h>
#include <assert.h>

typedef struct {int xy, i, j, on;} segment_t;

extern double urand(int, int64_t *);

// set initial segment
void initsegment(int nx, int ny, int randtype, int64_t *seed, double ratio, int nseg, segment_t seg[])
{
	int iseg = 0;
	for (int i = 0; i <  nx; i++) {
	for (int j = 0; j <= ny; j++) {
		seg[iseg].xy = 1;
		seg[iseg].i = i;
		seg[iseg].j = j;
		seg[iseg].on = (urand(randtype, seed) < ratio);
		iseg++;
	}
	}
	for (int i = 0; i <= nx; i++) {
	for (int j = 0; j <  ny; j++) {
		seg[iseg].xy = 2;
		seg[iseg].i = i;
		seg[iseg].j = j;
		seg[iseg].on = (urand(randtype, seed) < ratio);
		iseg++;
	}
	}
	assert(iseg == nseg);
}
