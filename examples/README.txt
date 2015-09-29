Provided here is list of example input scripts to run the force field fitting
program in one of two ways, either as a module via a python script or directly through the
command line. The following commands are equivalent (assuming fit_ff_parameters.py is
somewhere in your PATH):

$ ./run_ff_fitting.py

$ fit_ff_parameters.py h2o_methane.sapt h2o_methane.param coeffs.out -sb --aij geometric_mean --bij geometric_mean

-----------------

The chosen example shows how anisotropic coefficients, hard constraints, and
exponent optimization can all be employed simultaneously. 

The .sapt file contains dimer geometries (in bohr) as well as an energy
decomposition (in mH) for each component.

The .param file should *hopefully* be somewhat self-explanatory, but a few
notes are in order:

1. Hard constraints for Aparams are listed first. For isotropic atomtypes,
only one value needs to be included for each atomtype. For anisotropic
atomtypes, the isotropic coefficient should be listed first, followed by all
anisotropic coefficients in the same order as listed in the next section (LIST
OF ANISOTROPIC ATOMTYPES). All values are in a.u.

2. Anisotropic atomtypes are listed second; notation is as in the example
file.

3. Coordinate axes for each anisotropic atomtype are listed next. This is
potentially the most confusing section of the .param file, and I welcome
suggestions for improvement. Currently, the anisotropic atom index (indexing
from 0) is listed first, followed by which axis (z or x) is being described,
and finally a list of atomic indices defining the direction vector for the
axis. This vector is also defined by atomic indices; the first index (often
the anisotropic atom itself) lists the start of the vector, and the endpoint
of the vector is defined as the midpoint of all subsequently listed atoms.
    As a concrete example, to define a coordinate system for oxygen in water
(iO = 0, iH1 = 1, iH2 = 2), we set the z-axis as the bisecting vector between
oxygen and the two hydrogen atoms, and the x-axis as the vector between oxygen
and one of the hydrogens, giving us the following input lines:
0  z  0  1  2  
0  x  0  1  
Note that, while our x-axis is not currently orthogonal to the z-axis, the
program will project the x-axis into the plane defined by the z-axis so as to
make an orthogonal coordinate system.  Also note that, if the included
anisotropy terms only depend on the choice of z-axis, such as with our H2O
atomtype, an x-axis does not need to be specified. 

4. Exponents and Cn Coefficients are listed in the next two sections.

5. A multipole file (containing all relevant moments) is required for each
monomer. These multipole files use the same formatting as does ORIENT (only
the Q00 = ___ formatting is currently accepted, though this could be easily
changed if necessary).

6. Drude charges are required for each atom; setting charges to zero is
equivalent to a non-polarizable model.

7. Lastly, two parameters related to the weighting function are listed. We use
a Fermi-Dirac weighting function; mu and kt are the standard values given in
this formula.

