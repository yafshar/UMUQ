#!/usr/bin/env bash
set -e

# Create the configuration script
autoreconf -i
../configure
make
make check
