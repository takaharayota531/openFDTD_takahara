PROGRAM = ../ofd_mpi.exe

OMPOPT  = /openmp
MPIOPT  = /D_MPI

OBJS    =          \
Main.obj           \
setupSize.obj      \
setupSizeNear.obj  \
memallocfree3.obj  \
solve.obj          \
comm.obj           \
../src/updateEx.obj        \
../src/updateEy.obj        \
../src/updateEz.obj        \
../src/updateHx.obj        \
../src/updateHy.obj        \
../src/updateHz.obj        \
../src/dispersionEx.obj    \
../src/dispersionEy.obj    \
../src/dispersionEz.obj    \
../src/calcFar1d.obj       \
../src/calcFar2d.obj       \
../src/input.obj           \
../src/input2.obj          \
../src/monitor.obj         \
../src/memallocfree1.obj   \
../src/memallocfree2.obj   \
../src/setup.obj           \
../src/setup_vector.obj    \
../src/setupId.obj         \
../src/setupDispersion.obj \
../src/setupNear.obj       \
../src/setupMurHx.obj      \
../src/setupMurHy.obj      \
../src/setupMurHz.obj      \
../src/setupPmlEx.obj      \
../src/setupPmlEy.obj      \
../src/setupPmlEz.obj      \
../src/setupPmlHx.obj      \
../src/setupPmlHy.obj      \
../src/setupPmlHz.obj      \
../src/setupPml.obj        \
../src/initfield.obj       \
../src/vfeed.obj           \
../src/efeed.obj           \
../src/eload.obj           \
../src/vpoint.obj          \
../src/average.obj         \
../src/murHx.obj           \
../src/murHy.obj           \
../src/murHz.obj           \
../src/pmlEx.obj           \
../src/pmlEy.obj           \
../src/pmlEz.obj           \
../src/pmlHx.obj           \
../src/pmlHy.obj           \
../src/pmlHz.obj           \
../src/pbcx.obj            \
../src/pbcy.obj            \
../src/pbcz.obj            \
../src/dftNear1d.obj       \
../src/dftNear1dX.obj      \
../src/dftNear1dY.obj      \
../src/dftNear1dZ.obj      \
../src/dftNear2d.obj       \
../src/dftNear2dX.obj      \
../src/dftNear2dY.obj      \
../src/dftNear2dZ.obj      \
../src/dftNear3d.obj       \
../src/calcNear3d.obj      \
../src/nearfield_c.obj     \
../src/zfeed.obj           \
../src/spara.obj           \
../src/coupling.obj        \
../src/outputCross.obj     \
../src/plot3dGeom.obj      \
../src/ingeometry.obj      \
../src/geomlines.obj       \
../src/writeout.obj        \
../src/readout.obj         \
../src/post.obj            \
../src/farfield.obj        \
../src/planewave.obj       \
../src/outputFar0d.obj     \
../src/outputFar1d.obj     \
../src/outputFar2d.obj     \
../src/outputNear1d.obj    \
../src/outputNear2d.obj    \
../src/calcNear1d.obj      \
../src/calcNear2d.obj      \
../src/plot2dIter.obj      \
../src/plot2dFeed.obj      \
../src/plot2dPoint.obj     \
../src/plot2dFreq.obj      \
../src/plot2dFreq0.obj     \
../src/plot2dFar0d0.obj    \
../src/plot2dFar1d0.obj    \
../src/plot3dFar2d.obj     \
../src/plot3dFar2d0.obj    \
../src/plot2dNear1d0.obj   \
../src/plot2dNear2d0.obj   \
../src/plot3dNear2d0.obj   \
../src/utils.obj           \
../src/ev2dlib.obj         \
../src/ev2d.obj            \
../src/ev3d.obj

CC      = cl.exe
CFLAGS  = /O2 /nologo /I../include /W4 /wd4996
#CFLAGS  = $(CFLAGS) /D_VECTOR /D__NEC__
#CFLAGS  = $(CFLAGS) /D_DOUBLE
#CFLAGS  = $(CFLAGS) /D_ID32
LDFLAGS = /O2 /nologo
LIBS    = msmpi.lib

$(PROGRAM): $(OBJS)
	@echo "Loading $(PROGRAM) ... "
	@$(CC) $(LDFLAGS) /Fe$(PROGRAM) $(OBJS) $(LIBS)

Main.obj:
	$(CC) /c $(CFLAGS) $(OMPOPT) $?
comm.obj:
	$(CC) /c $(CFLAGS) $(MPIOPT) $?

.c.obj:
	$(CC) /c $(CFLAGS) $<

clean:
	del *.obj
