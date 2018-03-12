#!/usr/bin/env bash
set -e


cd external 
cd torc_lite  
./configure CC=mpicc F77=mpif77
make
sudo make install

cd ../


# Create the configuration script
autoreconf -i


# Run in a subdirectory to keep the sources clean
mkdir build || true
cd build
../configure
make
make check
