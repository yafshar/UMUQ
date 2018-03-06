#!/usr/bin/env bash
set -e

echo PATH=${PATH}

echo "Compiler configuration:"
echo CXX=${CXX}
echo CC=${CC}
echo F77=${F77}
echo FC=${FC}

echo CXXFLAGS=${CXXFLAGS}
echo CFLAGS=${CFLAGS}
echo LDFLAGS=${LDFLAGS}

echo "C++ compiler version:"
${CXX} --version || echo "${CXX} does not seem to support the --version flag"
${CXX} -v || echo "${CXX} does not seem to support the -v flag"

echo "C compiler version:"
${CC} --version || echo "${CXX} does not seem to support the --version flag"
${CC} -v || echo "${CXX} does not seem to support the -v flag"
