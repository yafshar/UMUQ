#!/usr/bin/env bash
set -e

# --------------------------------------------------------------------------
# bootstrapping utility for autotools
# -------------------------------------------------------------------------- 

aclocal

echo "Bootstrapping ..."

# Create the configuration script
autoreconf -f -i
automake --foreign --add-missing

