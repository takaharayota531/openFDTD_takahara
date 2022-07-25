PROGRAM = arf.exe

OMPOPT  = /openmp
MPIOPT  = /D_MPI

OBJS    =           \
Main.obj            \
initsegment.obj     \
setgeometry.obj     \
getvalue.obj        \
outputofd.obj       \
comm.obj            \
urand.obj           \
../src/input2.obj          \
../src/monitor.obj         \
../src/memallocfree1.obj   \
../src/memallocfree2.obj   \
../src/memallocfree3.obj   \
../src/setup.obj           \
../src/setupSize.obj       \
../src/setupSizeNear.obj   \
../src/setupId.obj         \
../src/setupDispersion.obj \
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
../src/setupNear.obj       \
../src/solve.obj           \
../src/initfield.obj       \
../src/updateEx.obj        \
../src/updateEy.obj        \
../src/updateEz.obj        \
../src/updateHx.obj        \
../src/updateHy.obj        \
../src/updateHz.obj        \
../src/dispersionEx.obj    \
../src/dispersionEy.obj    \
../src/dispersionEz.obj    \
../src/vfeed.obj           \
../src/efeed.obj           \
../src/eload.obj           \
../src/vpoint.obj          \
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
../src/calcNear2d.obj      \
../src/calcNear3d.obj      \
../src/nearfield_c.obj     \
../src/average.obj         \
../src/zfeed.obj           \
../src/spara.obj           \
../src/coupling.obj        \
../src/ingeometry.obj      \
../src/farfield.obj        \
../src/planewave.obj       \
../src/utils.obj

CC      = cl.exe
CFLAGS  = /O2 /nologo /I../include /W4 /wd4996
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
