#!/usr/bin/env bash
set -e

# Create the configuration script
autoreconf -i

# Run in a subdirectory to keep the sources clean
mkdir build || true
cd build
../configure
make
make check
