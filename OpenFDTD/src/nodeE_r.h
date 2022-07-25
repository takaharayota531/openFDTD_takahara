/*
nodeE_r.h

E on node (real)
*/

static inline void nodeE_r(int i, int j, int k, real_t *ex, real_t *ey, real_t *ez)
{
	if      (i <= iMin) {
		i = iMin;
		*ex = (EX(i + 0, j, k) * 3 - EX(i + 1, j, k) * 1) / 2;
	}
	else if (i >= iMax) {
		i = iMax;
		*ex = (EX(i - 1, j, k) * 3 - EX(i - 2, j, k) * 1) / 2;
	}
	else {
		*ex = (EX(i, j, k) + EX(i - 1, j, k)) / 2;
	}

	if      (j <= jMin) {
		j = jMin;
		*ey = (EY(i, j + 0, k) * 3 - EY(i, j + 1, k) * 1) / 2;
	}
	else if (j >= jMax) {
		j = jMax;
		*ey = (EY(i, j - 1, k) * 3 - EY(i, j - 2, k) * 1) / 2;
	}
	else {
		*ey = (EY(i, j, k) + EY(i, j - 1, k)) / 2;
	}

	if      (k <= kMin) {
		k = kMin;
		*ez = (EZ(i, j, k + 0) * 3 - EZ(i, j, k + 1) * 1) / 2;
	}
	else if (k >= kMax) {
		k = kMax;
		*ez = (EZ(i, j, k - 1) * 3 - EZ(i, j, k - 2) * 1) / 2;
	}
	else {
		*ez = (EZ(i, j, k) + EZ(i, j, k - 1)) / 2;
	}
}
