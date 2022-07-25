PROGRAM = ..\ofd_cuda.exe
TRASH   = ..\ofd_cuda.exp ..\ofd_cuda.lib

OBJS    =             \
Main.obj              \
info_gpu.obj          \
check_gpu.obj         \
cuda_memory.obj       \
setup_gpu.obj         \
memallocfree2_gpu.obj \
memallocfree3_gpu.obj \
solve.obj             \
initfield.obj         \
updateEx.obj          \
updateEy.obj          \
updateEz.obj          \
updateHx.obj          \
updateHy.obj          \
updateHz.obj          \
dispersionEx.obj      \
dispersionEy.obj      \
dispersionEz.obj      \
efeed.obj             \
eload.obj             \
vpoint.obj            \
murH.obj              \
pmlEx.obj             \
pmlEy.obj             \
pmlEz.obj             \
pmlHx.obj             \
pmlHy.obj             \
pmlHz.obj             \
pbcx.obj              \
pbcy.obj              \
pbcz.obj              \
dftNear1d.obj         \
dftNear1dX.obj        \
dftNear1dY.obj        \
dftNear1dZ.obj        \
dftNear2d.obj         \
dftNear2dX.obj        \
dftNear2dY.obj        \
dftNear2dZ.obj        \
dftNear3d.obj         \
average.obj           \
../src/input.obj             \
../src/input2.obj            \
../src/monitor.obj           \
../src/memallocfree1.obj     \
setup.obj             \
../src/setupSize.obj         \
../src/setupSizeNear.obj     \
setupId.obj           \
../src/setupDispersion.obj   \
../src/setupNear.obj         \
../src/setupMurHx.obj        \
../src/setupMurHy.obj        \
../src/setupMurHz.obj        \
../src/setupPmlEx.obj        \
../src/setupPmlEy.obj        \
../src/setupPmlEz.obj        \
../src/setupPmlHx.obj        \
../src/setupPmlHy.obj        \
../src/setupPmlHz.obj        \
../src/setupPml.obj          \
../src/vfeed.obj             \
../src/calcNear3d.obj        \
../src/nearfield_c.obj       \
../src/zfeed.obj             \
../src/spara.obj             \
../src/coupling.obj          \
../src/outputCross.obj       \
../src/plot3dGeom.obj        \
../src/ingeometry.obj        \
../src/geomlines.obj         \
../src/writeout.obj          \
../src/readout.obj           \
../src/post.obj              \
../src/farfield.obj          \
../src/planewave.obj         \
../src/outputFar0d.obj       \
../src/outputFar1d.obj       \
../src/outputFar2d.obj       \
../src/outputNear1d.obj      \
../src/outputNear2d.obj      \
calcFar1d.obj         \
calcFar2d.obj         \
../src/calcNear1d.obj        \
../src/calcNear2d.obj        \
../src/plot2dIter.obj        \
../src/plot2dFeed.obj        \
../src/plot2dPoint.obj       \
../src/plot2dFreq.obj        \
../src/plot2dFreq0.obj       \
../src/plot2dFar0d0.obj      \
../src/plot2dFar1d0.obj      \
../src/plot3dFar2d.obj       \
../src/plot3dFar2d0.obj      \
../src/plot2dNear1d0.obj     \
../src/plot2dNear2d0.obj     \
../src/plot3dNear2d0.obj     \
../src/utils.obj             \
../src/cputime.obj           \
../src/ev2dlib.obj           \
../src/ev2d.obj              \
../src/ev3d.obj

CC      = nvcc.exe
CFLAGS  = -O2 -I../include -Xcompiler "/wd4819 /W4"
#CFLAGS  = $(CFLAGS) -D_DOUBLE
#CFLAGS  = $(CFLAGS) -D_ID32
LDFLAGS = -O2
LIBS    =

$(PROGRAM): $(OBJS)
	@echo "Loading $(PROGRAM) ... "
	@$(CC) -o $(PROGRAM) $(LDFLAGS) $(OBJS) $(LIBS)
	del $(TRASH)

.SUFFIXES: .cu

setup.obj: ../src/setup.c
	$(CC) -c $(CFLAGS) $?
setupId.obj: ../src/setupId.c
	$(CC) -c $(CFLAGS) $?
calcFar1d.obj: ../src/calcFar1d.c
	$(CC) -c $(CFLAGS) $?
calcFar2d.obj: ../src/calcFar2d.c
	$(CC) -c $(CFLAGS) $?

.cu.obj:
	$(CC) -c $(CFLAGS) $<

clean:
	del *.obj
