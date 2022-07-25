PROGRAM = ../ofd.exe

OMPOPT  = /openmp

OBJS    =           \
Main.obj            \
input.obj           \
input2.obj          \
monitor.obj         \
memallocfree1.obj   \
memallocfree2.obj   \
memallocfree3.obj   \
setup.obj           \
setup_vector.obj    \
setupId.obj         \
setupSize.obj       \
setupSizeNear.obj   \
setupDispersion.obj \
setupNear.obj       \
setupMurHx.obj      \
setupMurHy.obj      \
setupMurHz.obj      \
setupPmlEx.obj      \
setupPmlEy.obj      \
setupPmlEz.obj      \
setupPmlHx.obj      \
setupPmlHy.obj      \
setupPmlHz.obj      \
setupPml.obj        \
solve.obj           \
initfield.obj       \
updateEx.obj        \
updateEy.obj        \
updateEz.obj        \
updateHx.obj        \
updateHy.obj        \
updateHz.obj        \
dispersionEx.obj    \
dispersionEy.obj    \
dispersionEz.obj    \
vfeed.obj           \
efeed.obj           \
eload.obj           \
vpoint.obj          \
murHx.obj           \
murHy.obj           \
murHz.obj           \
pmlEx.obj           \
pmlEy.obj           \
pmlEz.obj           \
pmlHx.obj           \
pmlHy.obj           \
pmlHz.obj           \
pbcx.obj            \
pbcy.obj            \
pbcz.obj            \
dftNear1d.obj       \
dftNear1dX.obj      \
dftNear1dY.obj      \
dftNear1dZ.obj      \
dftNear2d.obj       \
dftNear2dX.obj      \
dftNear2dY.obj      \
dftNear2dZ.obj      \
dftNear3d.obj       \
calcNear3d.obj      \
nearfield_c.obj     \
average.obj         \
zfeed.obj           \
spara.obj           \
coupling.obj        \
outputCross.obj     \
plot3dGeom.obj      \
ingeometry.obj      \
geomlines.obj       \
writeout.obj        \
readout.obj         \
post.obj            \
farfield.obj        \
planewave.obj       \
outputFar0d.obj     \
outputFar1d.obj     \
outputFar2d.obj     \
outputNear1d.obj    \
outputNear2d.obj    \
calcFar1d.obj       \
calcFar2d.obj       \
calcNear1d.obj      \
calcNear2d.obj      \
plot2dIter.obj      \
plot2dFeed.obj      \
plot2dPoint.obj     \
plot2dFreq.obj      \
plot2dFreq0.obj     \
plot2dFar0d0.obj    \
plot2dFar1d0.obj    \
plot3dFar2d.obj     \
plot3dFar2d0.obj    \
plot2dNear1d0.obj   \
plot2dNear2d0.obj   \
plot3dNear2d0.obj   \
utils.obj           \
cputime.obj         \
ev2dlib.obj         \
ev2d.obj            \
ev3d.obj

CC      = cl.exe
CFLAGS  = /O2 /nologo /I../include /W4 /wd4996
CFLAGS  = $(CFLAGS) $(OMPOPT)
#CFLAGS  = $(CFLAGS) /D_VECTOR /D__NEC__
#CFLAGS  = $(CFLAGS) /D_DOUBLE
#CFLAGS  = $(CFLAGS) /D_ID64
LDFLAGS = /O2 /nologo
LIBS    =

$(PROGRAM): $(OBJS)
	@echo "Loading $(PROGRAM) ... "
	@$(CC) $(LDFLAGS) /Fe$(PROGRAM) $(OBJS) $(LIBS)

.c.obj:
	$(CC) /c $(CFLAGS) $<

clean:
	del *.obj
