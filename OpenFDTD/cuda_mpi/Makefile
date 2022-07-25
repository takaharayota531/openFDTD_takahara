PROGRAM = ..\ofd_cuda_mpi.exe

TRASH   = ..\ofd_cuda_mpi.exp ..\ofd_cuda_mpi.lib
MPIOPT  = -D_MPI -I../../../../progra~2/micros~1/MPI/Include

OBJS    =             \
Main.obj              \
memallocfree3_gpu.obj \
solve.obj             \
comm_cuda.obj         \
info_gpu_mpi.obj      \
rank2device.obj       \
../cuda/check_gpu.obj         \
../cuda/cuda_memory.obj       \
../cuda/setup_gpu.obj         \
../cuda/memallocfree2_gpu.obj \
../cuda/initfield.obj         \
../cuda/updateEx.obj          \
../cuda/updateEy.obj          \
../cuda/updateEz.obj          \
../cuda/updateHx.obj          \
../cuda/updateHy.obj          \
../cuda/updateHz.obj          \
../cuda/dispersionEx.obj      \
../cuda/dispersionEy.obj      \
../cuda/dispersionEz.obj      \
../cuda/murH.obj              \
../cuda/pmlEx.obj             \
../cuda/pmlEy.obj             \
../cuda/pmlEz.obj             \
../cuda/pmlHx.obj             \
../cuda/pmlHy.obj             \
../cuda/pmlHz.obj             \
../cuda/pbcx.obj              \
../cuda/pbcy.obj              \
../cuda/pbcz.obj              \
../cuda/efeed.obj             \
../cuda/eload.obj             \
../cuda/vpoint.obj            \
../cuda/dftNear1d.obj         \
../cuda/dftNear1dX.obj        \
../cuda/dftNear1dY.obj        \
../cuda/dftNear1dZ.obj        \
../cuda/dftNear2d.obj         \
../cuda/dftNear2dX.obj        \
../cuda/dftNear2dY.obj        \
../cuda/dftNear2dZ.obj        \
../cuda/dftNear3d.obj         \
../cuda/average.obj           \
../src/input.obj             \
../src/input2.obj            \
../src/monitor.obj           \
../src/memallocfree1.obj     \
../cuda/setup.obj             \
../cuda/setupId.obj           \
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
../src/calcNear3d.obj        \
../src/nearfield_c.obj       \
../src/vfeed.obj             \
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
../cuda/calcFar1d.obj         \
../cuda/calcFar2d.obj         \
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
../src/ev2dlib.obj           \
../src/ev2d.obj              \
../src/ev3d.obj              \
../mpi/setupSize.obj         \
../mpi/setupSizeNear.obj     \
../mpi/comm.obj

CC      = nvcc.exe
CFLAGS  = -O2 -I../include -Xcompiler "/wd4819 /W4"
#CFLAGS  = $(CFLAGS) -D_VECTOR
#CFLAGS  = $(CFLAGS) -D_DOUBLE
#CFLAGS  = $(CFLAGS) -D_ID32
LDFLAGS = -O2
LIBS    = msmpi.lib

$(PROGRAM): $(OBJS)
	@echo "Loading $(PROGRAM) ... "
	@$(CC) -o $(PROGRAM) $(LDFLAGS) $(OBJS) $(LIBS)
	del $(TRASH)

.SUFFIXES: .cu

solve.obj:
	$(CC) -c $(CFLAGS) $(MPIOPT) $?
comm_cuda.obj:
	$(CC) -c $(CFLAGS) $(MPIOPT) $?

.cu.obj:
	$(CC) -c $(CFLAGS) $<

clean:
	del *.obj
