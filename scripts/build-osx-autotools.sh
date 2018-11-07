#!/usr/bin/env bash
set -e

rm -fr aclocal.m4
rm -fr m4/libtool.m4
rm -fr m4/ltoptions.m4
rm -fr m4/ltsugar.m4
rm -fr m4/ltversion.m4
rm -fr m4/lt\~obsolete.m4

# Create the configuration script
bash ./scripts/bootstrap.sh 

export UMUQ_CXX=`which mpic++` 
export UMUQ_CC=`which mpicc`
export UMUQ_FC=`which mpifort`

#configure and make
./configure CC=${UMUQ_CC} CXX=${UMUQ_CXX} FC=${UMUQ_FC} --with-googletest
bash ./scripts/bootstrap.sh 
make
make check
