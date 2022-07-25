/*
input.c

input data
*/

#include "ofd.h"
#include "ofd_prototype.h"

#define MAXTOKEN 1000

int input(FILE *fp, int check)
{
	int    ntoken, ngeom, nline;
	int    version = 0;
	int    nxr = 0, nyr = 0, nzr = 0;
	int    *dxr = NULL, *dyr = NULL, *dzr = NULL;
	double *xr = NULL, *yr = NULL, *zr = NULL;
	double *xfeed = NULL, *yfeed = NULL, *zfeed = NULL;
	double *xpoint = NULL, *ypoint = NULL, *zpoint = NULL;
	int    nload = 0;
	double *xload = NULL, *yload = NULL, *zload = NULL, *pload = NULL;
	char   *dload = NULL, *cload = NULL;
	char   prog[BUFSIZ];
	char   strline[BUFSIZ], strkey[BUFSIZ], strsave[BUFSIZ];
	char   strprop[BUFSIZ];
	char   *token[MAXTOKEN];
	const int array_inc = 10000;		// reduce malloc times
	const char sep[] = " \t";			// separator
	const char errfmt1[] = "*** too many %s data #%d\n";
	const char errfmt2[] = "*** invalid %s data\n";
	const char errfmt3[] = "*** invalid %s data #%d\n";

	// initialize

	NMaterial = 2;  // air + PEC
	Material = (material_t *)malloc(NMaterial * sizeof(material_t));
	for (int64_t m = 0; m < NMaterial; m++) {
		Material[m].type = 1;
		Material[m].epsr = 1;
		Material[m].esgm = 0;
		Material[m].amur = 1;
		Material[m].msgm = 0;
	}

	NGeometry = 0;

	NFeed = 0;
	rFeed = 0;

	IPlanewave = 0;

	NPoint = 0;

	NInductor = 0;

	iABC = 0;  // Mur-1st
	PBCx = PBCy = PBCz = 0;

	Dt = 0;
	Tw = 0;

	Solver.maxiter = 3000;
	Solver.nout = 50;
	Solver.converg = 1e-3;

	NFreq1 =
	NFreq2 = 0;

	MatchingLoss = 0;

	Piter =
	Pfeed =
	Ppoint = 0;

	IFreq[0] =
	IFreq[1] =
	IFreq[2] =
	IFreq[3] =
	IFreq[4] =
	IFreq[5] = 0;
	Freqdiv = 10;

	IFar0d =
	NFar1d =
	NFar2d =
	NNear1d =
	NNear2d = 0;

	Far1dScale.db = 1;       // dB
	Far1dScale.user = 0;     // auto scale
	Far1dStyle = 0;
	Far1dNorm = 0;
	Far1dComp[0] = 1;
	Far1dComp[1] = 0;
	Far1dComp[2] = 0;

	Far2dScale.db = 1;       // dB
	Far2dScale.user = 0;     // auto scale
	Far2dComp[0] = 1;
	Far2dComp[1] = 0;
	Far2dComp[2] = 0;
	Far2dComp[3] = 0;
	Far2dComp[4] = 0;
	Far2dComp[5] = 0;
	Far2dComp[6] = 0;
	Far2dObj = 0.5;

	Near1dScale.db = 0;      // V/m
	Near1dScale.user = 0;    // auto scale
	Near1dNoinc = 0;

	Near2dDim[0] = Near2dDim[1] = 1;
	Near2dFrame = 0;
	Near2dScale.db = 0;      // V/m
	Near2dScale.user = 0;    // auto scale
	Near2dObj = 1;
	Near2dNoinc = 0;
	Near2dIzoom = 0;

	Width2d    = 750;
	Height2d   = 500;
	Font2d     = 13;
	Fontname2d = 0;

	Width3d    = 600;
	Height3d   = 600;
	Font3d     = 12;
	Theta3d    = 60;
	Phi3d      = 30;

	// read

	nline = 0;
	while (fgets(strline, sizeof(strline), fp) != NULL) {
		// skip a empty line
		if (strlen(strline) <= 1) continue;

		// skip a comment line
		if (strline[0] == '#') continue;

		// delete "\n"
		//printf("%zd\n", strlen(strline));
		if (strstr(strline, "\r\n") != NULL) {
			strline[strlen(strline) - 2] = '\0';
		}
		else if ((strstr(strline, "\r") != NULL) || (strstr(strline, "\n") != NULL)) {
			strline[strlen(strline) - 1] = '\0';
		}
		//printf("%zd\n", strlen(strline));

		// "end" -> break
		if (!strncmp(strline, "end", 3)) break;

		// save "strline"
		strcpy(strsave, strline);

		// token ("strline" is destroyed)
		ntoken = tokenize(strline, sep, token, MAXTOKEN);
		//for (int i = 0; i < ntoken; i++) printf("%d %s\n", i, token[i]);

		// check number of data and "=" (exclude header)
		if ((nline > 0) && ((ntoken < 3) || strcmp(token[1], "="))) continue;

		// keyword
		strcpy(strkey, token[0]);

		// input
		if      (nline == 0) {
			strcpy(prog, strkey);
			if (strcmp(prog, "OpenFDTD") && strcmp(prog, "OpenTHFD")) {
				printf("%s\n", "*** not OpenFDTD/OpenTHFD data");
				return 1;
			}
			if (ntoken < 3) {
				printf("%s\n", "*** no version data");
				return 1;
			}
			version = (10 * atoi(token[1])) + atoi(token[2]);
			nline++;
		}
		else if (!strcmp(strkey, "title")) {
			strcpy(Title, strchr(strsave, '=') + 2);
		}
		else if (!strcmp(strkey, "xmesh")) {
			if ((ntoken < 5) || (ntoken % 2 == 0)) {
				printf(errfmt2, strkey);
				return 1;
			}
			nxr = (ntoken - 3) / 2;
			xr = (double *)malloc((nxr + 1) * sizeof(double));
			dxr = (int *)malloc(nxr * sizeof(int));
			sscanf(token[2], "%lf", &xr[0]);
			for (int i = 0; i < nxr; i++) {
				sscanf(token[2 * i + 3], "%d", &dxr[i]);
				sscanf(token[2 * i + 4], "%lf", &xr[i + 1]);
			}
		}
		else if (!strcmp(strkey, "ymesh")) {
			if ((ntoken < 5) || (ntoken % 2 == 0)) {
				printf(errfmt2, strkey);
				return 1;
			}
			nyr = (ntoken - 3) / 2;
			yr = (double *)malloc((nyr + 1) * sizeof(double));
			dyr = (int *)malloc(nyr * sizeof(int));
			sscanf(token[2], "%lf", &yr[0]);
			for (int j = 0; j < nyr; j++) {
				sscanf(token[2 * j + 3], "%d", &dyr[j]);
				sscanf(token[2 * j + 4], "%lf", &yr[j + 1]);
			}
		}
		else if (!strcmp(strkey, "zmesh")) {
			if ((ntoken < 5) || (ntoken % 2 == 0)) {
				printf(errfmt2, strkey);
				return 1;
			}
			nzr = (ntoken - 3) / 2;
			zr = (double *)malloc((nzr + 1) * sizeof(double));
			dzr = (int *)malloc(nzr * sizeof(int));
			sscanf(token[2], "%lf", &zr[0]);
			for (int k = 0; k < nzr; k++) {
				sscanf(token[2 * k + 3], "%d", &dzr[k]);
				sscanf(token[2 * k + 4], "%lf", &zr[k + 1]);
			}
		}
		else if (!strcmp(strkey, "material")) {
			if (ntoken < 7) {
				printf(errfmt3, strkey, (int)NMaterial - 1);
				return 1;
			}
			if (NMaterial % array_inc == 2) {   // 2 : initial set (air + PEC)
				Material = (material_t *)realloc(Material, (NMaterial + array_inc) * sizeof(material_t));
			}

			if (NMaterial >= MAXMATERIAL) {
				printf(errfmt1, strkey, (int)NMaterial - 1);
				return 1;
			}

			int type = 1;
			double epsr = 1, esgm = 0, amur = 1, msgm = 0;
			double einf = 0, ae = 0, be = 0, ce = 0;
			if (!strcmp(prog, "OpenFDTD") && (version < 22)) {
				type = 1;
				epsr = atof(token[2]);
				esgm = atof(token[3]);
				amur = atof(token[4]);
				msgm = atof(token[5]);
			}
			else if (!strcmp(token[2], "1")) {
				type = 1;
				epsr = atof(token[3]);
				esgm = atof(token[4]);
				amur = atof(token[5]);
				msgm = atof(token[6]);
			}
			else if (!strcmp(token[2], "2")) {
				type = 2;
				einf = atof(token[3]);
				ae   = atof(token[4]);
				be   = atof(token[5]);
				ce   = atof(token[6]);
				epsr = 1;
				esgm = 0;
				amur = 1;
				msgm = 0;
			}
			if ((type == 1) && ((epsr <= 0) || (esgm < 0) || (amur <= 0) || (msgm < 0))) {
				printf(errfmt3, strkey, (int)NMaterial - 1);
				return 1;
			}
			else if ((type == 2) && (einf <= 0)) {
				printf(errfmt3, strkey, (int)NMaterial - 1);
				return 1;
			}
			Material[NMaterial].type = type;
			Material[NMaterial].epsr = epsr;
			Material[NMaterial].esgm = esgm;
			Material[NMaterial].amur = amur;
			Material[NMaterial].msgm = msgm;
			Material[NMaterial].einf = einf;
			Material[NMaterial].ae   = ae;
			Material[NMaterial].be   = be;
			Material[NMaterial].ce   = ce;
			NMaterial++;
			//printf("%d\n", NMaterial);
		}
		else if (!strcmp(strkey, "geometry")) {
			if (ntoken < 4) {
				printf(errfmt3, strkey, (int)NGeometry + 1);
				return 1;
			}
			if (NGeometry % array_inc == 0) {
				Geometry = (geometry_t *)realloc(Geometry, (NGeometry + array_inc) * sizeof(geometry_t));
			}
			Geometry[NGeometry].m     = (id_t)atoi(token[2]);
			Geometry[NGeometry].shape = atoi(token[3]);
			switch (Geometry[NGeometry].shape) {
				case 1:
				case 2:
				case 11:
				case 12:
				case 13:
					ngeom = 6;
					break;
				case 31:
				case 32:
				case 33:
				case 41:
				case 42:
				case 43:
				case 51:
				case 52:
				case 53:
					ngeom = 8;
					break;
				default:
					ngeom = 0;
					break;
			}
			if (ntoken < 4 + ngeom) {
				printf(errfmt3, strkey, (int)NGeometry + 1);
				return 1;
			}
			for (int n = 0; n < ngeom; n++) {
				Geometry[NGeometry].g[n] = atof(token[4 + n]);
			}
			NGeometry++;
		}
		else if (!strcmp(strkey, "name")) {
			;
		}
		else if (!strcmp(strkey, "feed")) {
			if (ntoken > 8) {
				Feed  = (feed_t *)realloc(Feed,  (NFeed + 1) * sizeof(feed_t));
				xfeed = (double *)realloc(xfeed, (NFeed + 1) * sizeof(double));
				yfeed = (double *)realloc(yfeed, (NFeed + 1) * sizeof(double));
				zfeed = (double *)realloc(zfeed, (NFeed + 1) * sizeof(double));
				Feed[NFeed].dir   = (char)toupper((int)token[2][0]);
				xfeed[NFeed]      = atof(token[3]);
				yfeed[NFeed]      = atof(token[4]);
				zfeed[NFeed]      = atof(token[5]);
				Feed[NFeed].volt  = atof(token[6]);
				Feed[NFeed].delay = atof(token[7]);
				Feed[NFeed].z0    = atof(token[8]);
				NFeed++;
			}
			else {
				printf(errfmt3, strkey, NFeed + 1);
				return 1;
			}
		}
		else if (!strcmp(strkey, "planewave")) {
			if (ntoken < 5) {
				printf(errfmt2, strkey);
				return 1;
			}
			IPlanewave = 1;
			Planewave.theta = atof(token[2]);
			Planewave.phi   = atof(token[3]);
			Planewave.pol   = atoi(token[4]);
		}
		else if (!strcmp(strkey, "point")) {
			if (((NPoint == 0) && (ntoken < 7)) || (ntoken < 6)) {
				printf(errfmt3, strkey, NPoint + 1);
				return 1;
			}
			Point  = (point_t *)realloc(Point,  (NPoint + 1) * sizeof(point_t));
			xpoint = (double *) realloc(xpoint, (NPoint + 1) * sizeof(double));
			ypoint = (double *) realloc(ypoint, (NPoint + 1) * sizeof(double));
			zpoint = (double *) realloc(zpoint, (NPoint + 1) * sizeof(double));
			Point[NPoint].dir = (char)toupper((int)token[2][0]);
			xpoint[NPoint]    = atof(token[3]);
			ypoint[NPoint]    = atof(token[4]);
			zpoint[NPoint]    = atof(token[5]);
			if (NPoint == 0) {
				strcpy(strprop, token[6]);		// propagation on port #1
			}
			NPoint++;
		}
		else if (!strcmp(strkey, "load")) {
			if (ntoken < 8) {
				printf(errfmt3, strkey, nload + 1);
				return 1;
			}
			dload = (char   *)realloc(dload, (nload + 1) * sizeof(char));
			xload = (double *)realloc(xload, (nload + 1) * sizeof(double));
			yload = (double *)realloc(yload, (nload + 1) * sizeof(double));
			zload = (double *)realloc(zload, (nload + 1) * sizeof(double));
			cload = (char   *)realloc(cload, (nload + 1) * sizeof(char));
			pload = (double *)realloc(pload, (nload + 1) * sizeof(double));
			dload[nload] = (char)toupper((int)token[2][0]);
			xload[nload] = atof(token[3]);
			yload[nload] = atof(token[4]);
			zload[nload] = atof(token[5]);
			cload[nload] = (char)toupper((int)token[6][0]);
			pload[nload] = atof(token[7]);
			nload++;
		}
		else if (!strcmp(strkey, "rfeed")) {
			if (ntoken > 2) {
				rFeed = atof(token[2]);
			}
		}
		else if (!strcmp(strkey, "abc")) {
			if      ((ntoken >= 3) && !strncmp(token[2], "0", 1)) {
				iABC = 0;
			}
			else if ((ntoken >= 6) && !strncmp(token[2], "1", 1)) {
				iABC = 1;
				cPML.l = atoi(token[3]);
				cPML.m = atof(token[4]);
				cPML.r0 = atof(token[5]);
			}
			else {
				printf(errfmt2, strkey);
				return 1;
			}
		}
		else if (!strcmp(strkey, "pbc")) {
			if      (ntoken >= 5) {
				PBCx = atoi(token[2]);
				PBCy = atoi(token[3]);
				PBCz = atoi(token[4]);
			}
			else {
				printf(errfmt2, strkey);
				return 1;
			}
		}
		else if (!strcmp(strkey, "frequency1")) {
			if (ntoken > 4) {
				double f0 = atof(token[2]);
				double f1 = atof(token[3]);
				int fdiv = atoi(token[4]);
				if (fdiv < 0) {
					printf(errfmt2, strkey);
					return 1;
				}
				double df = (fdiv > 0) ? (f1 - f0) / fdiv : 0;
				NFreq1 = fdiv + 1;
				Freq1 = (double *)malloc(NFreq1 * sizeof(double));
				for (int n = 0; n < NFreq1; n++) {
					Freq1[n] = f0 + (n * df);
				}
			}
		}
		else if (!strcmp(strkey, "frequency2")) {
			if (ntoken > 4) {
				double f0 = atof(token[2]);
				double f1 = atof(token[3]);
				int fdiv = atoi(token[4]);
				if (fdiv < 0) {
					printf(errfmt2, strkey);
					return 1;
				}
				double df = (fdiv > 0) ? (f1 - f0) / fdiv : 0;
				NFreq2 = fdiv + 1;
				Freq2 = (double *)malloc(NFreq2 * sizeof(double));
				for (int n = 0; n < NFreq2; n++) {
					Freq2[n] = f0 + (n * df);
				}
			}
		}
		else if (!strcmp(prog, "OpenTHFD") && !strcmp(strkey, "frequency")) {
			if (ntoken > 4) {
				double f0 = atof(token[2]);
				double f1 = atof(token[3]);
				int fdiv = atoi(token[4]);
				if (fdiv < 0) {
					printf(errfmt2, strkey);
					return 1;
				}
				double df = (fdiv > 0) ? (f1 - f0) / fdiv : 0;
				NFreq1 = NFreq2 = fdiv + 1;
				Freq1 = (double *)malloc(NFreq1 * sizeof(double));
				Freq2 = (double *)malloc(NFreq2 * sizeof(double));
				for (int n = 0; n < NFreq1; n++) {
					Freq1[n] = Freq2[n] = f0 + (n * df);
				}
			}
		}
		else if (!strcmp(strkey, "solver")) {
			if (ntoken > 4) {
				Solver.maxiter = atoi(token[2]);
				Solver.nout    = atoi(token[3]);
				Solver.converg = atof(token[4]);
				if (Solver.maxiter % Solver.nout != 0) {
					Solver.maxiter = (Solver.maxiter / Solver.nout + 1) * Solver.nout;
				}
			}
		}
		else if (!strcmp(strkey, "timestep")) {
			if (ntoken > 2) {
				Dt = atof(token[2]);
			}
		}
		else if (!strcmp(strkey, "pulsewidth")) {
			if (ntoken > 2) {
				Tw = atof(token[2]);
			}
		}

		// === post ===

		else if (!strcmp(strkey, "matchingloss")) {
			if (ntoken > 2) {
				MatchingLoss = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "plotiter")) {
			if (ntoken > 2) {
				Piter = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "plotfeed")) {
			if (ntoken > 2) {
				Pfeed = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "plotpoint")) {
			if (ntoken > 2) {
				Ppoint = atoi(token[2]);
			}
		}
		/* version <= 2.5.5
		else if (!strcmp(strkey, "plotfreq")) {
			if (ntoken > 7) {
				for (int n = 0; n < 6; n++) {
					IFreq[n] = atoi(token[2 + n]);
				}
			}
		}*/
		else if (!strcmp(strkey, "plotsmith")) {
			if (ntoken > 2) {
				IFreq[0] = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "plotzin"     ) ||
		         !strcmp(strkey, "plotyin"     ) ||
		         !strcmp(strkey, "plotref"     ) ||
		         !strcmp(strkey, "plotspara"   ) ||
		         !strcmp(strkey, "plotcoupling")) {
			const int id = !strcmp(strkey, "plotzin"     ) ? 1
			             : !strcmp(strkey, "plotyin"     ) ? 2
			             : !strcmp(strkey, "plotref"     ) ? 3
			             : !strcmp(strkey, "plotspara"   ) ? 4
			             : !strcmp(strkey, "plotcoupling") ? 5 : 0;
			if ((ntoken > 2) && !strcmp(token[2], "1")) {
				IFreq[id] = 1;
				FreqScale[id].user = 0;
			}
			else if ((ntoken > 5) && !strcmp(token[2], "2")) {
				IFreq[id] = 1;
				FreqScale[id].user = 1;
				FreqScale[id].min = atof(token[3]);
				FreqScale[id].max = atof(token[4]);
				FreqScale[id].div = atoi(token[5]);
			}
		}
		else if (!strcmp(strkey, "freqdiv")) {
			if (ntoken > 2) {
				Freqdiv = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "plotfar0d")) {
			if ((ntoken > 4) && !strcmp(token[4], "1")) {
				IFar0d = 1;
				Far0d[0] = atof(token[2]);
				Far0d[1] = atof(token[3]);
				Far0dScale.user = 0;
			}
			else if ((ntoken > 7) && !strcmp(token[4], "2")) {
				IFar0d = 1;
				Far0d[0] = atof(token[2]);
				Far0d[1] = atof(token[3]);
				Far0dScale.user = 1;
				Far0dScale.min = atof(token[5]);
				Far0dScale.max = atof(token[6]);
				Far0dScale.div = atoi(token[7]);
			}
		}
		/*
		else if (!strcmp(strkey, "plotfar0d")) {
			if (ntoken > 3) {
				Far0d = (double (*)[2])realloc(Far0d, (NFar0d + 1) * 2 * sizeof(double));
				Far0d[NFar0d][0] = atof(token[2]);
				Far0d[NFar0d][1] = atof(token[3]);
				NFar0d++;
			}
		}*/
		else if (!strcmp(strkey, "plotfar1d")) {
			char dir = (char)toupper((int)token[2][0]);
			if ((dir != 'X') && (dir != 'Y') && (dir != 'Z') &&
			    (dir != 'V') && (dir != 'H')) {
				printf(errfmt3, strkey, NFar1d + 1);
				return 1;
			}
			if ((((dir == 'X') || (dir == 'Y') || (dir == 'Z')) && (ntoken < 4)) ||
			    (((dir == 'V') || (dir == 'H')) && (ntoken < 5))) {
				printf(errfmt3, strkey, NFar1d + 1);
				return 1;
			}
			Far1d = (far1d_t *)realloc(Far1d, (NFar1d + 1) * sizeof(far1d_t));
			Far1d[NFar1d].dir = dir;
			Far1d[NFar1d].div = atoi(token[3]);
			if ((dir == 'V') || (dir == 'H')) {
				Far1d[NFar1d].angle = atof(token[4]);
			}
			NFar1d++;
		}
		else if (!strcmp(strkey, "plotfar2d")) {
			if (ntoken > 3) {
				Far2d.divtheta = atoi(token[2]);
				Far2d.divphi   = atoi(token[3]);
				NFar2d = 1;
			}
		}
		else if (!strcmp(strkey, "plotnear1d")) {
			if ((ntoken < 6) || (strlen(token[2]) > 2) || (strlen(token[3]) > 1)) {
				printf(errfmt3, strkey, NNear1d + 1);
				return 1;
			}
			Near1d = (near1d_t *)realloc(Near1d, (NNear1d + 1) * sizeof(near1d_t));
			//pos1d1 = (double *)realloc(pos1d1, (NNear1d + 1) * sizeof(double));
			//pos1d2 = (double *)realloc(pos1d2, (NNear1d + 1) * sizeof(double));
			strcpy(Near1d[NNear1d].cmp, token[2]);
			Near1d[NNear1d].dir = (char)toupper((int)token[3][0]);
			Near1d[NNear1d].pos1 = atof(token[4]);
			Near1d[NNear1d].pos2 = atof(token[5]);
			NNear1d++;
		}
		else if (!strcmp(strkey, "plotnear2d")) {
			if ((ntoken < 5) || (strlen(token[2]) > 2) || (strlen(token[3]) > 1)) {
				printf(errfmt3, strkey, NNear2d + 1);
				return 1;
			}
			Near2d = (near2d_t *)realloc(Near2d, (NNear2d + 1) * sizeof(near2d_t));
			//pos2d0 = (double *)realloc(pos2d0, (NNear2d + 1) * sizeof(double));
			strcpy(Near2d[NNear2d].cmp, token[2]);
			Near2d[NNear2d].dir = (char)toupper((int)token[3][0]);
			Near2d[NNear2d].pos0 = atof(token[4]);
			NNear2d++;
		}
		else if (!strcmp(strkey, "far1dcomponent")) {
			if (ntoken > 4) {
				for (int n = 0; n < 3; n++) {
					Far1dComp[n] = atoi(token[2 + n]);
				}
			}
		}
		else if (!strcmp(strkey, "far1dstyle")) {
			if (ntoken > 2) {
				Far1dStyle = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "far1ddb")) {
			if (ntoken > 2) {
				Far1dScale.db = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "far1dnorm")) {
			if (ntoken > 2) {
				Far1dNorm = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "far1dscale")) {
			if (ntoken > 4) {
				Far1dScale.user = 1;
				Far1dScale.min = atof(token[2]);
				Far1dScale.max = atof(token[3]);
				Far1dScale.div = atoi(token[4]);
			}
		}
		else if (!strcmp(strkey, "far2dcomponent")) {
			if (ntoken > 8) {
				for (int n = 0; n < 7; n++) {
					Far2dComp[n] = atoi(token[2 + n]);
				}
			}
		}
		else if (!strcmp(strkey, "far2ddb")) {
			if (ntoken > 2) {
				Far2dScale.db = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "far2dscale")) {
			if (ntoken > 3) {
				Far2dScale.user = 1;
				Far2dScale.min  = atof(token[2]);
				Far2dScale.max  = atof(token[3]);
			}
		}
		else if (!strcmp(strkey, "far2dobj")) {
			if (ntoken > 2) {
				Far2dObj = atof(token[2]);
			}
		}
		else if (!strcmp(strkey, "near1ddb")) {
			if (ntoken > 2) {
				Near1dScale.db = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "near1dscale")) {
			if (ntoken > 4) {
				Near1dScale.user = 1;
				Near1dScale.min = atof(token[2]);
				Near1dScale.max = atof(token[3]);
				Near1dScale.div = atoi(token[4]);
			}
		}
		else if (!strcmp(strkey, "near1dnoinc")) {
			if (ntoken > 2) {
				Near1dNoinc = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "near2ddim")) {
			if (ntoken > 3) {
				Near2dDim[0] = atoi(token[2]);
				Near2dDim[1] = atoi(token[3]);
			}
		}
		else if (!strcmp(strkey, "near2dframe")) {
			if (ntoken > 2) {
				Near2dFrame = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "near2ddb")) {
			if (ntoken > 2) {
				Near2dScale.db = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "near2dscale")) {
			if (ntoken > 3) {
				Near2dScale.user = 1;
				Near2dScale.min = atof(token[2]);
				Near2dScale.max = atof(token[3]);
			}
		}
		else if (!strcmp(strkey, "near2dcontour")) {
			if (ntoken > 2) {
				Near2dContour = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "near2dobj")) {
			if (ntoken > 2) {
				Near2dObj = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "near2dnoinc")) {
			if (ntoken > 2) {
				Near2dNoinc = atoi(token[2]);
			}
		}
		else if (!strcmp(strkey, "near2dzoom")) {
			if (ntoken > 5) {
				Near2dIzoom = 1;
				Near2dHzoom[0] = MIN(atof(token[2]), atof(token[3]));
				Near2dHzoom[1] = MAX(atof(token[2]), atof(token[3]));
				Near2dVzoom[0] = MIN(atof(token[4]), atof(token[5]));
				Near2dVzoom[1] = MAX(atof(token[4]), atof(token[5]));
			}
		}
		else if (!strcmp(strkey, "window2d")) {
			if (ntoken > 5) {
				Width2d    = atoi(token[2]);
				Height2d   = atoi(token[3]);
				Font2d     = atoi(token[4]);
				Fontname2d = atoi(token[5]);
			}
		}
		else if (!strcmp(strkey, "window3d")) {
			if (ntoken > 6) {
				Width3d  = atoi(token[2]);
				Height3d = atoi(token[3]);
				Font3d   = atoi(token[4]);
				Theta3d  = atof(token[5]);
				Phi3d    = atof(token[6]);
			}
		}
	}
/*
	// debug
	//printf("title = %s\n", Title);
	//printf("xmesh = %e", xr[0]); for (int i = 0; i < nxr; i++) printf(" %d %e", dxr[i], xr[i + 1]); printf("\n");
	//printf("ymesh = %e", yr[0]); for (int j = 0; j < nyr; j++) printf(" %d %e", dyr[j], yr[j + 1]); printf("\n");
	//printf("zmesh = %e", zr[0]); for (int k = 0; k < nzr; k++) printf(" %d %e", dzr[k], zr[k + 1]); printf("\n");
	//for (int n = 0; n < NMaterial; n++) if (Material[n].type == 1) printf("material = %d %.3e %.3e %.3e %.3e\n", Material[n].epsr, Material[n].esgm, Material[n].amur, Material[n].msgm);
	//for (int n = 0; n < NMaterial; n++) if (Material[n].type == 2) printf("material = %.3e %.3e %.3e %.3e\n", Material[n].einf, Material[n].ae, Material[n].be, Material[n].ce);
	//for (int n = 0; n < NGeometry; n++) printf("geometry = %d %.3e %.3e %.3e %.3e %.3e %.3e\n", Geometry[n].m, Geometry[n].g[0], Geometry[n].g[1], Geometry[n].g[2], Geometry[n].g[3], Geometry[n].g[4], Geometry[n].g[5]);
	//for (int n = 0; n < NFeed; n++) printf("feed = %c %e %e %e %e\n", Feed[n].dir, xfeed[n], yfeed[n], zfeed[n], Feed[n].volt);
	//for (int n = 0; n < NFreq1; n++) printf("frequency1 = %e\n", Freq1[n]);
	//for (int n = 0; n < NFreq2; n++) printf("frequency2 = %e\n", Freq2[n]);
	//printf("%d %d %e %e\n", iABC, cPML.l, cPML.m, cPML.r0);
	//printf("solver = %d %d %e\n", Solver.maxiter, Solver.nout, Solver.converg);
	//for (int n = 0; n < NNear2d; n++) printf("near2d = %c %f\n", Near2d[n].dir, pos2d[n]);
	//for (int n = 0; n < NFar; n++) printf("far = %c %d\n", Far[n].dir, Far[n].div);
*/
	// error check

	if (check) {
		if (nxr <= 0) {
			printf("%s\n", "*** no xmesh data");
			return 1;
		}
		if (nyr <= 0) {
			printf("%s\n", "*** no ymesh data");
			return 1;
		}
		if (nzr <= 0) {
			printf("%s\n", "*** no zmesh data");
			return 1;
		}
		for (int i = 0; i < nxr; i++) {
			if ((xr[i] >= xr[i + 1]) || (dxr[i] <= 0)) {
				printf("%s\n", "*** invalid xmesh data");
				return 1;
			}
		}
		for (int j = 0; j < nyr; j++) {
			if ((yr[j] >= yr[j + 1]) || (dyr[j] <= 0)) {
				printf("%s\n", "*** invalid ymesh data");
				return 1;
			}
		}
		for (int k = 0; k < nzr; k++) {
			if ((zr[k] >= zr[k + 1]) || (dzr[k] <= 0)) {
				printf("%s\n", "*** invalid zmesh data");
				return 1;
			}
		}
		if (!NFeed && !IPlanewave) {
			printf("%s\n", "*** no source");
			return 1;
		}
		if (NFeed && IPlanewave) {
			printf("%s\n", "*** feed and planewave");
			return 1;
		}
		if ((Solver.maxiter <= 0) || (Solver.nout <= 0)) {
			printf("%s\n", "*** invalid solver data");
			return 1;
		}
	}

	for (int64_t n = 0; n < NGeometry; n++) {
		if (Geometry[n].m >= NMaterial) {
			printf("%zd %d %zd\n", n, (int)Geometry[n].m, NMaterial);
			printf("*** invalid material id of geometry data #%zd\n", n + 1);
			return 1;
		}
	}
	for (int n = 0; n < NFeed; n++) {
		if ((Feed[n].dir != 'X') && (Feed[n].dir != 'Y') && (Feed[n].dir != 'Z')) {
			printf("%s #%d\n", "*** invalid feed direction", n + 1);
			return 1;
		}
	}
	for (int n = 0; n < NPoint; n++) {
		if ((Point[n].dir != 'X') && (Point[n].dir != 'Y') && (Point[n].dir != 'Z')) {
			printf("%s #%d\n", "*** invalid point direction", n + 1);
			return 1;
		}
	}
	for (int n = 0; n < nload; n++) {
		if (((dload[n] != 'X') && (dload[n] != 'Y') && (dload[n] != 'Z')) ||
		    ((cload[n] != 'R') && (cload[n] != 'C') && (cload[n] != 'L'))) {
			printf("*** invalid load parameter #%d\n", n + 1);
			return 1;
		}
	}
	for (int n = 0; n < NFar1d; n++) {
		if (Far1d[n].div < 2) {
			printf("%s #%d\n", "*** invalid far1d division", n + 1);
			return 1;
		}
	}
	if (NFar2d) {
		if ((Far2d.divtheta <= 0) || (Far2d.divphi <= 0)) {
			printf("%s\n", "*** invalid far2d division");
			return 1;
		}
	}
/*
	if ((NNear1d || NNear2d) && (NFreq2 <= 0)) {
		printf("%d %d %d\n", NNear1d, NNear2d, NFreq2);
		printf("%s\n", "*** no near field frequency");
		return 1;
	}
*/
	for (int n = 0; n < NNear1d; n++) {
		if ((Near1d[n].dir != 'X') && (Near1d[n].dir != 'Y') && (Near1d[n].dir != 'Z')) {
			printf("%s #%d\n", "*** invalid near1d direction", n + 1);
			return 1;
		}
	}
	for (int n = 0; n < NNear2d; n++) {
		if ((Near2d[n].dir != 'X') && (Near2d[n].dir != 'Y') && (Near2d[n].dir != 'Z')) {
			printf("%s #%d\n", "*** invalid near2d direction", n + 1);
			return 1;
		}
	}
/*
	if ((IFar0d || NFar1d || NFar2d) && (NFreq2 <= 0)) {
		printf("%s\n", "*** no far field frequency");
		return 1;
	}
*/

	// warnings
/*
	if ((NFreq1 <= 0) && (NFreq2 <= 0)) {
		printf("%s\n", "*** no frequency data");
	}
*/
	// PBC -> Mur-1st
	if ((iABC == 1) && (PBCx || PBCy || PBCz)) {
		printf("%s\n", "*** warning : PBC -> Mur-1st");
		iABC = 0;
	}

	// number of cells
	setup_cells(nxr, nyr, nzr, dxr, dyr, dzr);

	// node
	setup_node(nxr, nyr, nzr, xr, yr, zr, dxr, dyr, dzr);

	// cell center
	setup_center();

	// feed
	if (NFeed) {
		setup_feed(xfeed, yfeed, zfeed);
	}

	// plane wave
	if (IPlanewave) {
		setup_planewave();
	}

	// point
	if (NPoint) {
		setup_point(xpoint, ypoint, zpoint, strprop);
	}

	// load
	if (nload > 0) {
		setup_load(nload, dload, xload, yload, zload, cload, pload, array_inc);
	}

	// near1d
	setup_near1d();

	// near2d
	setup_near2d();

	// fit geometry without thickness
	fitgeometry();

	// free
	free(xr);
	free(yr);
	free(zr);
	free(dxr);
	free(dyr);
	free(dzr);
	if (NFeed) {
		free(xfeed);
		free(yfeed);
		free(zfeed);
	}
	if (NPoint) {
		free(xpoint);
		free(ypoint);
		free(zpoint);
	}

	return 0;
}
