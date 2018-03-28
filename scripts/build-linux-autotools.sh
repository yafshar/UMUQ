#!/usr/bin/env bash
set -e

# Create the configuration script
autoreconf -i
#configure and make
./configure --with-googletest
make
make check
