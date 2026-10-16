CC = gcc
AR = ar -r
RANLIB = ranlib
MACROS = -DNO_ATOMIC #-DNICS_INT64 #-DNO_EXTENSION
ARCH = 
CFLAGS = -c -g -O2 -Wall -Wextra $(ARCH) $(MACROS)
LFLAGS = 
