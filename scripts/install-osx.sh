#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	brew update;
	brew install cmake || brew upgrade cmake ;
	brew install autoconf;
	brew install automake;
	brew tap homebrew/versions;
	brew install gcc@5;
	brew install mpich;
	brew install flann;
	brew update;
fi


