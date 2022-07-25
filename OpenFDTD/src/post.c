/*
post.c

post process
*/

#include "ofd.h"
#include "complex.h"
#include "ev.h"
#include "ofd_prototype.h"

void post(int ev)
{
	ev2d_init(Width2d, Height2d);
	ev2d_setFont((Fontname2d == 0) ? "sansserif" : (Fontname2d == 1) ? "serif" : "monospace");

	ev3d_init(Width3d, Height3d);

	if (Piter) {
		plot2dIter();
	}

	if (Pfeed) {
		plot2dFeed();
	}

	if (Ppoint) {
		plot2dPoint();
	}

	if (NFreq1) {
		plot2dFreq();
	}

	if (NFreq2) {

		// setup near field
		if (runMode == 2) {
			if (NNear1d) {
				calcNear1d();
			}
			if (NNear2d || IFar0d || NFar1d || NFar2d) {
				calcNear2d(0);
				calcNear2d(1);
			}
		}

		if (IFar0d) {
			outputFar0d();
		}

		if (NFar1d) {
			outputFar1d();
		}

		if (NFar2d) {
			outputFar2d();
		}

		if (NNear1d) {
			outputNear1d();
		}

		if (NNear2d) {
			outputNear2d();
		}

	}

	ev2d_file(ev, (ev ? FN_ev2d_1 : FN_ev2d_0));
	ev2d_output();

	if (!ev) ev3d_setAngle(Theta3d, Phi3d);
	ev3d_file(ev, (ev ? FN_ev3d_1 : FN_ev3d_0), 0);
	ev3d_output();
}
