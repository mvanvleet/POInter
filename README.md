# POInter: Parameter Optimization for Intermolecular Force Fields


<img src="http://www.clipartpal.com/_thumbs/pd/animal/dog/pointer_3.png" width="300" >



Introduction
------------
POInter is an open-source python package for intermolecular force field parameter optimization. 
It has primarily been designed to optimize parameters for the MASTIFF force field, which 
uses a Slater-type functional form to describe short-range effects. In addition, 
POInter can be used to optimize intermolecular potentialss whose short-range effects are 
described by one of the following functional forms:

* Lennard-Jones 12-6 Potentials
* Born-Mayer Potentials
* Stone/Misquitta form: exp(-ar - p))

POInter can be called directly from the command line, or can be executed from within a Python script.


Getting Started
---------------
POInter is a relatively new (and constantly updating) code. 
Email mvanvleet at chem dot wisc dot edu
with any questions and to ask for the most recent (stable) version of the code.

Documentation can be found in this repository's wiki.

Issues with the code (bugs, unclear documentation, suggestions for improvement)
should be reported with the issue tracker.


Dependencies
------------
To run properly, POInter requires the following python packages:

* Numpy
* Scipy
* Sympy
* [Cloudpickle](https://github.com/cloudpipe/cloudpickle) (optional, but speeds up multipole calculations)



