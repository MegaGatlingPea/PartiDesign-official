NAME1=DAlphaBall
DIR=.

CPU=gcc
NAMEFUL1=$(DIR)/$(NAME1).$(CPU)
LIBRARIES=-lgmp -lm -lgfortran
LD=$(CC) # use c compiler as frontend for linker

.c.o :
	$(CC) -c $(CFLAGS) -O3 -fcommon $(INCS) $<

.f.o :
	$(FC) -c $(FFLAGS) -O3 -x f77-cpp-input -fsecond-underscore $<

OBJECTS1 = \
$(NAME1).o \
sos_minor_gmp.o alf_tools_gmp.o binding.o \
adjust.o alfcx.o alfcx_tools.o cavballs.o delcx.o truncate_real.o \
measure_tools.o surface_tools.o surface_only.o surface_deriv_coord.o \
volume_tools.o volume_only.o volume_deriv_coord.o \
vector.o write_simplices.o

$(NAMEFUL1) : $(OBJECTS1)
	$(LD) -o $(NAMEFUL1) $(LDFLAGS) $(OBJECTS1) $(LIBS) $(LIBRARIES)

all: $(OBJECTS1)
	$(LD) -o $(NAMEFUL1) $(LDFLAGS) $(OBJECTS1) $(LIBS) $(LIBRARIES)

clean:
	rm -f *.o $(NAMEFUL1)

$(OBJECTS) : defines.h gmpvar.h
