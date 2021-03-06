<img src="./docs/umuq.png?raw=true" width="100">

[![Build Status](https://travis-ci.com/yafshar/UMUQ.svg?token=aY1dW9PfH9SMySdB6Pzy&branch=master)](https://travis-ci.com/yafshar/UMUQ)
[![License](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](LICENSE)

Welcome to **UMUQ**, University of Michigan's Uncertainty Quantification framework!

Introduction
------------

<span style="font-size:300%; color:red; font-weight: 900;">!WORK IN PROGRESS!</span>

Purpose :

<table>
  <tr>
    <td colspan="2"> Multivariate normal distribution  &nbsp; &nbsp;
    <img src="https://latex.codecogs.com/svg.latex?&space;f=0.1\mathcal{N}\left(\mu_1,\sigma^2\right)+0.9\mathcal{N}\left(\mu_2,\sigma^2\right)" title="f=0.1\mathcal{N}\left(\mu_1,\sigma^2\right)+0.9\mathcal{N}\left(\mu_2,\sigma^2\right)" />
    </td>
</td>
  </tr>
  <tr>
    <td> <img src="./docs/two_Gaussian.png?raw=true" width="400" height="400"> </td>
    <td> <img src="./docs/two_Gaussian.gif?raw=true" width="400" height="400"> </td>
  </tr>
</table>

<table>
  <tr>
    <td colspan="2"> Rosenbrock function  &nbsp; &nbsp; <img src="https://latex.codecogs.com/svg.latex?&space;f=\left(1-x\right)^2+100\left(y-x^2\right)^2" title="f=\left(1-x\right)^2+100\left(y-x^2\right)^2" /> </td>
</td>
  </tr>
  <tr>
    <td> <img src="./docs/Rosenbrock.png?raw=true" width="400" height="400"> </td>
    <td> <img src="./docs/Rosenbrock.gif?raw=true" width="400" height="400"> </td>
  </tr>
</table>

Initial release of UMUQ source program.

<span style="font-size:300%; color:red; font-weight: 900;">!WORK IN PROGRESS!</span>

Getting the code
------------

You can download the latest version from [here](https://github.com/yafshar/UMUQ).<br>
The very latest version is always available via 'github' by invoking one of the following:

````bash

## For the traditional ssh-based Git interaction:
$ git clone git@github.com:yafshar/UMUQ.git

## For HTTP-based Git interaction
$ git clone https://github.com/yafshar/UMUQ.git

````

Prerequisites
------------

If you wish to compile UMUQ from source, you will require the following components (these are **not** necessary for running the statically linked binary):

1. **Torc**; _latest version_ (<https://github.com/yafshar/torc_lite-1>)

   A tasking library that allows to write platform-independent code.
   UMUQ uses [torc](https://github.com/yafshar/torc_lite-1) which is forked from the original work of (TORC tasking library), and contains changes towards the original library.

2. **Eigen**; _at least 3.3.2_ release (<http://eigen.tuxfamily.org>)

   Eigen forms the core mathematics library of UMUQ, with all its linear algebra routines.

3. **FLANN**; _latest version 1.8_ (<http://www.cs.ubc.ca/research/flann>)

   FLANN is a library for performing fast approximate nearest neighbor searches in high dimensional spaces.

4. **Google Test** (optional); _at least 1.8.0_ (<https://github.com/google/googletest>)

   Google's framework for writing and using C++ test & mock classes.

Furthermore, you will require a compiler that can handle **C++0x** (which includes all **c++14** compilers with Variable templates feature). 
(GCC >= 5, Clang >= 3.4, MSVC >= 19.0, EDG eccp >= 4.11, Intel C++ >= 17.0, IBM XLC++ >= 13.1.2, Sun/Oracle C++ >= 5.15, Cray >= 8.6, and Portland Group (PGI) >= 17.4)

UMUQ has been successfully compiled with GCC 5.0 and GCC 8.0 on Gentoo/Debian and compiled with GCC 6.0 and Apple LLVM version 9.0.0 (clang-900.0.37) on macOS.

If you wish to do development, you will require parts of the extended GNU toolchain (the infamous Autotools):

1. **Autoconf**; latest 2.69 release (<http://www.gnu.org/software/autoconf>)

   GNU Autoconf produces the ./configure script from configure.ac.

2. **Automake**; latest 1.14 release (<http://www.gnu.org/software/automake>)

   GNU Automake produces the Makefile.in precursor, that is processed with ./configure to yield the final Makefile.

3. **Libtool**; latest 2.4.2 release (<http://www.gnu.org/software/libtool>)

   GNU Libtool is required as a dependency in configure.

UMUQ is tested and developed on UNIX-like systems, hence supporting Microsoft Windows is not a goal at the moment.

Installation
------------

If you do not have a `configure` script in the top level directory, run `bootstrap` to generate a configure script using `autotools`.

Before compiling, you must run the `configure` script. To run, type `./configure`. Additional options may be provided if desired. Run `./configure --help` for details.

After successfully running `configure`, type `make` to build the `UMUQ` library.

Then type `make install` to install it in the directory previously specified by the `--prefix` option of the `configure` script.

Documentation
-------------

`UMUQ` documentation is available [here](https://yafshar.github.io/UMUQ/).

Licenses
------------

UMUQ is LGPL 3.0 licensed. The [LICENSE](https://github.com/yafshar/UMUQ/blob/master/LICENSE) file contains the LGPL 3.0 text. Also see these links:<br>
    [LGPL3](https://www.gnu.org/licenses/lgpl-3.0.en.html)<br>
    [GPL-FAQ](https://www.gnu.org/licenses/gpl-faq.html)

Packages in [external](https://github.com/yafshar/UMUQ/tree/master/external) folder contain third-party code under:<br>
[MPL2](https://www.mozilla.org/en-US/MPL/2.0),<br>
[GPL3](https://www.gnu.org/licenses/gpl-3.0.html),<br>
[GPL2](https://www.gnu.org/licenses/gpl-2.0.html),<br>
[BSD](https://github.com/eigenteam/eigen-git-mirror/blob/72741bba73c97bdd1a16896ad3eed6934ea4ccb6/COPYING.BSD) and<br>
[BSD3](https://github.com/mariusmuja/flann/blob/f3a17cd3f94a0e9dd8f6a55bce11536c50d4fb24/COPYING) licenses.

Contact us
------------

If something is not working as you think it should or would like it to, please get in touch with us! Further, if you have an algorithm or any idea that you would want to try using the UMUQ, please get in touch with us, we would be glad to help!

[![Join the chat at https://gitter.im/UMUQ](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/UMUQ/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Contributing
------------

Contributions are very welcome.  If you wish to contribute, please take a few moments to review the [branching model](http://nvie.com/posts/a-successful-git-branching-model/) `UMUQ` utilizes.

Citing UMUQ
-------

Please add the following citation to any paper, technical report or article describing the use of the `UMUQ` library:

```bibtex

```

Contributors
------------

UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>

Computational Aerosciences Laboratory<br>
University of Michigan, Ann Arbor
