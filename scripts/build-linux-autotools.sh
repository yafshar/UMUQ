#!/usr/bin/env bash
set -e

# Create the configuration script
autoreconf -i
#configure and make
./configure
make
make check
