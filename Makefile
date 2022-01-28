.SUFFIXES: .c .cpp

CC=g++
# linux
#CFLAGS=-I. -O3 -fopenmp
CFLAGS=-I.
# MinGW
#CFLAGS=-I. -O3 -fno-strict-aliasing -fopenmp -lpthread
#CFLAGS=-I. -g -fno-strict-aliasing
LD=g++
# linux
#LFLAGS=-O3 -fopenmp
LFLAGS=-O3
# MinGW
#LFLAGS=-g -fno-strict-aliasing -Wl,--large-address-aware 
#LFLAGS=-O3 -fno-strict-aliasing -Wl,--large-address-aware -fopenmp -lpthread
LIBS= 

%.o:	%.cpp
	$(CC) -c -o $@ $(CFLAGS) $<

%.o:	%.c
	gcc -c -o $@ $(CFLAGS) $<

%.o:	%.f
	g77 -c -o $@ $(CFLAGS) $<

target=ssvm_tool ssvm_tool_bin
all: $(target)

# ssvm
SSVM= joint_ssvm.o latent_ssvm.o ssvm.o ssvm_train.o
SSVM_BIN= joint_ssvm_bin.o latent_ssvm_bin.o ssvm_bin.o ssvm_train.o

# general feature
ssvm_tool: ssvm_tool.o $(SSVM) ssvm_cmdline.o getopt.o getopt1.o
	$(LD) -o $@ ssvm_tool.o $(SSVM) \
	ssvm_cmdline.o getopt.o getopt1.o \
	$(LIBS) $(LFLAGS)

# binary feature
ssvm_tool_bin: ssvm_tool.o $(SSVM_BIN) ssvm_cmdline.o getopt.o getopt1.o
	$(LD) -o $@ ssvm_tool.o $(SSVM_BIN) \
	ssvm_cmdline.o getopt.o getopt1.o \
	$(LIBS) $(LFLAGS)

ssvm_bin.o:	ssvm.cpp ssvm.hpp
	$(CC) -c -o $@ -D BINARY_FEATURE $(CFLAGS) ssvm.cpp

latent_ssvm_bin.o:	latent_ssvm.cpp latent_ssvm.hpp
	$(CC) -c -o $@ -D BINARY_FEATURE $(CFLAGS) latent_ssvm.cpp

joint_ssvm_bin.o:	joint_ssvm.cpp joint_ssvm.hpp
	$(CC) -c -o $@ -D BINARY_FEATURE $(CFLAGS) joint_ssvm.cpp

install:
	cp ssvm_tool*.exe /usr/local/bin

backup: *.cpp *.hpp *.h *.c
	cp *.[ch]pp backup
	cp *.[chf] backup

clean:
	rm *.o *.a *.lib *.dll *.def *.exp *.so

