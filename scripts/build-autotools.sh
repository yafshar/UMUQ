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

# Configure and make
./configure --with-googletest
make
make check
