/*
nodeH_r.h

H on node (real)
*/

static inline void nodeH_r(int i, int j, int k, real_t *hx, real_t *hy, real_t *hz)
{
	*hx = (HX(i, j, k) + HX(i,     j - 1, k    ) + HX(i,     j,     k - 1) + HX(i,     j - 1, k - 1)) / 4;
	*hy = (HY(i, j, k) + HY(i,     j,     k - 1) + HY(i - 1, j,     k    ) + HY(i - 1, j,     k - 1)) / 4;
	*hz = (HZ(i, j, k) + HZ(i - 1, j,     k    ) + HZ(i,     j - 1, k    ) + HZ(i - 1, j - 1, k    )) / 4;
}
