@echo off
echo Building KLT library and examples...

REM Set compiler
set CC=gcc

REM Set flags
set CFLAGS=-DNDEBUG -I../include

REM Create object files
echo Compiling source files...
%CC% -c %CFLAGS% ../src/convolve.c
%CC% -c %CFLAGS% ../src/error.c
%CC% -c %CFLAGS% ../src/pnmio.c
%CC% -c %CFLAGS% ../src/pyramid.c
%CC% -c %CFLAGS% ../src/selectGoodFeatures.c
%CC% -c %CFLAGS% ../src/storeFeatures.c
%CC% -c %CFLAGS% ../src/trackFeatures.c
%CC% -c %CFLAGS% ../src/klt.c
%CC% -c %CFLAGS% ../src/klt_util.c
%CC% -c %CFLAGS% ../src/writeFeatures.c

REM Create library
echo Creating library...
ar ruv libklt.a *.o
del *.o

REM Build example3 (our main target)
echo Building example3...
%CC% -O3 %CFLAGS% -o example3 ../src/example3.c -L. -lklt -lm

echo Build complete!
echo.
echo To run example3, execute: example3.exe
echo Output files will be created in ../output/
