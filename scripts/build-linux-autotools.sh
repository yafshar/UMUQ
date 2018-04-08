#!/usr/bin/env bash
set -e

aclocal
# Create the configuration script
autoreconf -i
#configure and make
./configure --with-googletest
make
make check
