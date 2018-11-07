#!/usr/bin/env bash
set -e

rm aclocal.m4
rm m4/libtool.m4
rm m4/ltoptions.m4
rm m4/ltsugar.m4
rm m4/ltversion.m4
rm m4/lt\~obsolete.m4

# Create the configuration script
bash ./scripts/bootstrap.sh 

export CXX=`which mpic++` 
export CC=`which mpicc`
export F77=`which mpifort` 
export FC=`which mpifort`

#configure and make
./configure --with-googletest
make
make check
