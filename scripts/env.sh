#!/usr/bin/env bash

export CXX=`which mpic++`
export CC=`which mpicc`
if [ -n "$(type -t mpifort)" ]; then
    export F77=`which mpifort`
    export FC=`which mpifort`
elif [ -n "$(type -t mpif90)" ]; then
    export F77=`which mpif90`
    export FC=`which mpif90`
fi
