# UMUQ
[![Build Status](https://travis-ci.com/yafshar/UMUQ.svg?token=aY1dW9PfH9SMySdB6Pzy&branch=master)](https://travis-ci.com/yafshar/UMUQ)

Welcome to **UMUQ**, University of Michigan's Uncertainty Quantification framework!

Introduction
------------

Purpose :

<table>
  <tr>
    <td colspan="2"> Multivariate normal distribution  &nbsp; &nbsp; <img src="https://latex.codecogs.com/svg.latex?&space;f=0.1\mathcal{N}\left(\mu_1,\sigma^2\right)+0.9\mathcal{N}\left(\mu_2,\sigma^2\right)" title="f=0.1\mathcal{N}\left(\mu_1,\sigma^2\right)+0.9\mathcal{N}\left(\mu_2,\sigma^2\right)" /> </td>
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

Getting the code
------------
You can download the latest version from [here](https://github.com/yafshar/UMUQ).
The very latest version is always available via 'github' by invoking one of the following:
````
## For the traditional ssh-based Git interaction:
$ git clone git@github.com:yafshar/UMUQ.git

## For HTTP-based Git interaction
$ git clone https://github.com/yafshar/UMUQ.git
````

Licenses
------------
UMUQ is LGPL 3.0 licensed. See these links:    
    [LGPL3](https://www.gnu.org/licenses/lgpl-3.0.en.html)    
    [GPL-FAQ](https://www.gnu.org/licenses/gpl-faq.html)

The [LICENSE](https://github.com/yafshar/UMUQ/blob/master/LICENSE) file contains the LGPL 3.0 text.

Packages in [external](https://github.com/yafshar/UMUQ/tree/master/external) folder contain 
third-party code under 
[MPL2](https://github.com/yafshar/UMUQ/tree/master/external/COPYING.MPL2), [GPL](https://github.com/yafshar/UMUQ/tree/master/external/COPYING.GPL), 
[BSD](https://github.com/yafshar/UMUQ/tree/master/external/COPYING.BSD) and [BSD3](https://github.com/yafshar/UMUQ/tree/master/external/COPYING.BSD3) licenses.

Contributors
------------
UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>

Computational Aerosciences Laboratory  
University of Michigan, Ann Arbor 
