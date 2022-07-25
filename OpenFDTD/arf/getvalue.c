/*
getvalue.c
*/

#include "ofd.h"
#include "ofd_prototype.h"
#include "complex.h"

// (static) far field
// e[0-6] = Eabs/Etheta/Ephi/Emajor/Eminor/RHCP/LHCP
static void efar(int ifreq, double theta, double phi, double e[])
{
	assert(ifreq >= 0);
	d_complex_t etheta, ephi;
	double ffctr = farfactor(ifreq);
	farfield(ifreq, theta, phi, ffctr, &etheta, &ephi);
	farComponent(etheta, ephi, e);
}


// input impedance
d_complex_t zin(int ifeed, int ifreq)
{
	assert(ifeed >= 0);
	assert(ifreq >= 0);
	return Zin[(ifeed * NFreq1) + ifreq];
}


// input impedance difference
double zindiff(int ifeed, int ifreq, double z0)
{
	assert(ifeed >= 0);
	assert(ifreq >= 0);
	assert(z0 > 0);
	const d_complex_t z = zin(ifeed, ifreq);
	return d_abs(d_sub(z, d_complex(z0, 0)));
}


// reflection
double reflection(int ifeed, int ifreq, int db, double z0)
{
	assert(ifeed >= 0);
	assert(ifreq >= 0);
	assert(z0 > 0);
	const d_complex_t z = zin(ifeed, ifreq);
	double ret = d_abs(d_div(d_sub(z, d_complex(z0, 0)), d_add(z, d_complex(z0, 0))));
	if (db) {
		ret = 20 * log10(MAX(ret, 1e-2));
	}
	return ret;
}


// VSWR
double vswr(int ifeed, int ifreq, double z0)
{
	assert(ifeed >= 0);
	assert(ifreq >= 0);
	assert(z0 > 0);
	const double gamma = reflection(ifeed, ifreq, 0, z0);
	return (fabs(1 - gamma) > EPS) ? (1 + gamma) / (1 - gamma) : 1000;
}


// gain, component = 0-6
double gain(int ifreq, double theta, double phi, int component, int db)
{
	assert(ifreq >= 0);
	assert((component >= 0) && (component <= 6));
	double e[7];
	efar(ifreq, theta, phi, e);
	double ret = e[component];
	if (db) {
		ret = 20 * log10(MAX(ret, EPS));
	}
	return ret;
}


double getvalue(void)
{
	double f = 0;

	int ifeed = 0;
	int ifreq = 0;
	double theta = 0;
	double phi = 0;
	const int pol = 1;
	const int db = 1;
	const double z0 = 50;

	// gain
	f -= gain(ifreq, theta, phi, pol, db);
	//f -= gain(ifreq, 45,   0, 1, 0);
	//f += 0.02 * gain(ifreq, 30, 180, 0, 0);
	//f += 0.02 * gain(ifreq, 45, 180, 0, 0);

/*
	// gain : frequency average
	for (int ifreq = 0; ifreq < NFrequency; ifreq++) {
		f -= gain(ifreq, theta, phi, pol, db) / NFrequency;
	}
*/

	// input impedanece
	f += 0.03 * zindiff(ifeed, ifreq, z0);

/*
	// input impedanece : frequency average
	for (ifreq = 0; ifreq < NFrequency; ifreq++) {
		f += 0.03 * zindiff(ifeed, ifreq) / NFrequency;
	}
*/
	return f;
}
