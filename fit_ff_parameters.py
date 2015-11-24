#!/share/apps/python
__version__ = '1.0.1'

# Standard Packages
import numpy as np
import sys
import os
from scipy.optimize import minimize
from copy import copy
import sympy as sp
from sympy.utilities import lambdify
from sympy.utilities.iterables import flatten
import itertools

# Local modules
import functional_forms 
from multipoles import Multipoles

# Numpy error message settings
#np.seterr(all='raise')

####################################################################################################    
####################################################################################################    
class FitFFParameters:
    '''Given two input files, one containing geometries and decomposed QM
    energies for a series of dimer configurations (energy_file) and the other
    containing hard constraints and settings (param_file), fit unknown force
    field parameters using a least squares fitting procedure, and output the
    results to output_file.

    Terminology
    -----------
    QM - Quantum mechanical (usually SAPT)
    FF - Force field

    Usage
    -----
    $ fit_ff_parameters.py [.sapt file] [parameter file] [output file]
    (for other fitting options, run ./fit_ff_parameters.py -h)

    Attributes
    ----------
    energy_file : str
        File containing geometries and a QM energy decomposition for a series
        of dimer configurations; see example for details and formatting.
    param_file : str
        File containing hard constraints (exponents, multipoles, drude
        charges, dispersion coefficients, and optionally pre-factors) and
        optimization options (weighting function, anisotropies); see example
        for details and formatting.
    output_file : str, optional
        Output file. Default 'coeffs.out'.
    slater_correction : bool, optional.
        If True, multiplies short-range exponential by a polynomial pre-factor
        so as to model the short-range potential as an overlap of Slater
        orbitals. Default True. Note that this option should always be used if
        exponents are derived from atomic densities using the ISA procedure.
    fit_bii : bool, optional
        If True, optimizes Bii parameters. Default False (i.e., fixed Bii
        parameters).
    aij_combination_rule : str, optional
        String to set Aij combination rule. Default 'geometric_mean'; can also
        be set to 'saptff', or 'waldman-hagler5'.
    bij_combination_rule : str, optional
        String to set Bij combination rule. Default 'geometric_mean'; can also
        be set to 'saptff', 'arithmetic_mean', or 'waldman-hagler5'.
    functional_form : str, optional
        Short-range potential can be modeled using either a Born-Mayer
        functional form (E_sr = Aexp(-br) ) or a Stone functional form (E_sr =
        Kexp(-b(r - a)) ). Default 'born-mayer'; can also be set to 'stone'.
        Either functional form can be used with or without a Slater correction
        (see slater_correction, above).

    Methods
    -------
    fit_ff_parameters
        Fit force field parameters to each component of the energy
        decomposition. This routine is automatically called upon
        initialization of this module, and serves as a high-level function to
        call all necessary subroutines in order to optimize parameters.

    Known Issues
    ------------

    Units
    -----
    Atomic units are assumed throughout this module (exception is for energies
    read in from energy_file, which are read in as mH and then converted to
    a.u.).

    '''
    def __init__(self,energy_file,param_file,
                output_file='coeffs.out',
                slater_correction=True,
                fit_bii=False,
                aij_combination_rule='geometric',
                bij_combination_rule='geometric_mean',
                cij_combination_rule='geometric',
                functional_form='born-mayer'):

        '''Initilialize input variables and run the main fitting code.
        '''

        ###########################################################################
        ################ User-Defined Class Variables #############################
        # User-Defined Class Variables, below, change frequently enough that
        # they are read in each time this class is instantiated. Defaults are
        # given for some of the variables.
        self.energy_file=energy_file
        self.param_file=param_file
        self.output_file=output_file

        # If set to true, computes the density overlap as the overlap of
        # Slaters rather than as a simple Born-Mayer potential
        self.slater_correction = slater_correction
        # If set to true, fits scale factors to each exponent (i.e., optimizes
        # exponents)
        self.fit_bii = fit_bii

        # Set combination rules for pre-factors and exponents. Options for
        # each are as follows:
        #   aij: 'saptff', 'waldman-hagler5', 'geometric_mean' (same as #   saptff)
        #   bij: 'saptff', 'waldman-hagler5', 'geometric_mean', 'arithmetic_mean'
        self.aij_combination_rule = aij_combination_rule
        self.bij_combination_rule = bij_combination_rule
        self.cij_combination_rule = cij_combination_rule

        # Functional form can be chosen to either be the Born-Mayer or Stone
        # potentials; see Stone's book for more details.
        #self.functional_form = 'stone'
        #self.functional_form = 'born-mayer'
        self.functional_form = functional_form
        ###########################################################################
        ###########################################################################


        ###########################################################################
        ################## Program-Defined Class Variables ########################
        # Program-Defined Class Variable defaults, below, can be redefined in
        # the .param file as necessary, but should be left unchanged in most
        # cases.

        # ----------------------------------------------------------------------
        # General Variables; Sets names and types of energy components that
        # can be fit.
        # ----------------------------------------------------------------------

        # Turn on/off verbose printing
        self.verbose = False

        # Number of energy components:
        self.ncomponents = 7
            # components are, in order, exchange (0),
            #                           electrostatics (1),
            #                           induction (2),
            #                           dhf (3),
            #                           dispersion (4),
            #                           residual error (5),
            #                           total energy (6)


        # Names of Energy Components (ordering same as self.ncomponents)
        self.energy_component_names = ['Exchange',
                                      'Electrostatics',
                                      'Induction',
                                      'Dhf', 
                                      'Dispersion', 
                                      'Residual Energy',
                                      'Total_Energy']

        # Names of Output Files (ordering same as self.ncomponents)
        self.energy_component_file = ['exchange.dat',
                                      'electrostatics.dat',
                                      'induction.dat',
                                      'dhf.dat',
                                      'dispersion.dat',
                                      'residual_energy.dat',
                                      'total_energy.dat']

        # ----------------------------------------------------------------------
        # ATOMTYPE Variables; options for how to read in and recognize
        # atomtypes
        # ----------------------------------------------------------------------

        # Set ignorecase to true if atomtypes are not case sensitive
        self.ignorecase = False

        # The flag tells the program how to deal with atom where multiple
        # exponent values are read in for the same atomtype. Options are
        # 'average_exponents' or 'raise_error'
        self.handle_parameter_conflicts = 'average_exponents'

        # ----------------------------------------------------------------------
        # Paramter Optimization Variables; controls weighting functions,
        # paramter defaults, and constraints (both hard and soft)
        # ----------------------------------------------------------------------

        # Weighed RMSE cutoff in Ha; only points with total interaction energies
        # below this cutoff will be included in the weighted RMSE fit
        self.weighted_rmse_cutoff = 0.0

        # Parameters related to drude oscillators (in a.u.)
        self.springcon = 0.1
        self.thole_param = 2.0

        # If scaling exponents, determine whether or not to scale all
        # exponents separately, or just scale the exchange exponent (and set
        # all subsequent fits to use this exchange-scaled exponent)
        self.only_scale_exchange_bii = True

        # If scaling exponents, determine whether or not to put harmonic
        # constraints on scaled bii coefficients
        self.harmonic_constraints = True

        # Constrain optimization to positive values of the pre-factor and
        # exponent. This flag does not effect the sign of any anisotropic
        # corrections. Changing this boolean can have some effect on the
        # convergence properties of the fitting procedure, and may be worth
        # changing to achieve optimal fits.
        self.constrain_ab_positive = True

        # If set to true, fits a final A parameter to errors in the total
        # energy, in an effort to reduce systematic errors in the total energy
        self.fit_residual_errors = False
        #self.fit_residual_errors = True

        # ----------------------------------------------------------------------
        # Functional Form Variables; controls options related to the
        # mathematical form of the force field itself
        # ----------------------------------------------------------------------

        # Number of parameters per isotropic atomtype
        self.n_isotropic_params = 1

        # If a radial correction is being employed, choose whether or not this
        # correction should correspond to the exact Slater overlap correction
        # or a more approximate form, which is only formally exact for bi=bj.
        self.exact_radial_correction = False
        #self.exact_radial_correction = True

        # Choose damping method for electrostatic interactions. Currently
        # accepted options are 'None' and 'Tang-Toennies'
        self.electrostatic_damping_type = 'None'

        # When fitting anisotropic parameters, choose whether or not to fit
        # anisotropic dispersion.
        self.fit_anisotropic_dispersion = True

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # The following variables should be changed rarely, if ever, and are
        # primarily included for debugging purposes:
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        # Get source for drude oscillator energies. Can either be read in or
        # computed on the fly. Currently three computation engines exist, two
        # gradient optimizers (one is limited to computing rank0 field terms)
        # and a finite-differences module (can do up to rank2, but is super
        # slow). Aside from debugging, the multipole-gradient code should
        # always be used.
        # Options: 'read', 'pointcharge-gradient', 'multipole-gradient', 'finite-differences'
        self.drude_method='multipole-gradient'
        # If drude method is 'read', specify a filename containing the drude
        # energies:
        self.drude_file = 'edrudes.dat'

        # The multipolar component of the interaction energy can be computed
        # using the Orient program. In this case, the location of the energy
        # file (1 energy per line in the same order as the .sapt file) must be
        # specified.
        self.read_multipole_energy_from_orient = False
        self.orient_multipolar_energy_file = 'orient_multipolar_energy.dat'


        ###########################################################################


        ###########################################################################
        ###########################################################################


        self.fit_ff_parameters()

        return
####################################################################################################    
####################################################################################################    




####################################################################################################    
    def fit_ff_parameters(self):
        '''Fit force field parameters to all components of the QM energy
        decomposition.

        This is the main routine for this module.

        Parameters
        ----------
        None

        Returns
        ------
        None, though optimized FF parameters will have been produced and
        output to output_file.

        '''
        # Read in input files:
        self.read_energy_file()
        self.read_param_file()
        # Initialize arrays, variables, and constants that will later be used
        # in parameter fitting, and which serve to keep track of which atomtypes
        # are isotropic/anisotropic and to be fit/constrained.
        self.initialize_parameters()

        # Fit parameters for the exchange, electrostatics, induction, and dhf
        # energy components and output results to output files:
        ff_energy = np.zeros_like(self.qm_energy[6])

        # Fit exchange pre-factors (and potentially exponents, depending on
        # value of self.fit_bii)
        self.component = 0 
        ff_energy += self.fit_component_parameters()

        # Once exponents are set, can compute the drude oscillator energy that
        # will later be needed for induction and dhf components
        self.get_drude_oscillator_energy()
                
        # Fit electrostatic, induction, and dhf pre-factors
        for i in range(1,4): 
            self.component = i
            ff_energy += self.fit_component_parameters()

        # Compute the dispersion energy and output to file; parameters may not
        # be fit here
        self.component = 4
        if self.fit_anisotropic_dispersion and self.anisotropic_atomtypes:
            # Subtract one free parameter per atomtype; a0 is constrained to
            # be the input (isotropic) cn coefficient
            self.n_isotropic_params -= 1
            ff_energy += self.fit_component_parameters()
            self.n_isotropic_params += 1
        else:
            ff_energy += self.calc_dispersion_energy()

        # If fitting the residual errors, compute this energy and output to
        # file
        if self.fit_residual_errors:
            self.component = 5
            # Changing weighting here!
            eff_mu_tmp = self.eff_mu
            self.eff_mu = np.amin(self.qm_energy[6])*0.5
            print self.eff_mu
            self.qm_energy[5] = self.qm_energy[6] - ff_energy
            ff_energy += self.fit_component_parameters()
            self.eff_mu = eff_mu_tmp

        # Sum up all energy components and output to the total energy file 
        self.component = 6
        qm_energy = self.qm_energy[self.component]
        weight = functional_forms.weight(qm_energy, self.eff_mu, self.eff_kt)
        self.lsq_error =  np.sum(weight*(ff_energy - qm_energy)**2)
        self.write_energy_file(ff_energy)
        self.rms_error = self.calc_rmse(ff_energy)
        self.weighted_rms_error = self.calc_rmse(ff_energy, cutoff=self.weighted_rmse_cutoff)
        self.weighted_absolute_error = self.calc_mae(ff_energy, cutoff=self.weighted_rmse_cutoff)
        self.write_output_file(ff_energy)

        print 'Finished fitting all force field parameters.' 

        return
####################################################################################################    


####################################################################################################    
    def read_energy_file(self):
        '''Read in contents of the qm energy file, creating arrays
        to store the qm energy and xyz coordinates of each data point in the
        file.

        Parameters
        ----------
        None, aside from implicit dependence on energy_file

        Returns
        -------
        None, though initializes values for the following class variables:
        natoms[1,2] : int
            Number of atoms in each monomer.
        atoms[1,2] : list
            Atomtype names for each atom in monomer.
        ndatpts : int
            Number of dimer configurations to fit.
        xyz[1,2] : 3darray (ndatpts x natoms[1,2] x 3)
            Cartesian positions for each atom in each monomer.
        qm_energy : 2darray (ncomponents x ndatpts)
            QM energies for each component (exchange etc.) and each dimer
            configuration.
        r12 : ndarray
            Interatomic distances between all atom pairs i and j in
            monomers 1 and 2, respectively.
        atomtypes : list
            List of all unique atomtypes read in through the QM energy file.

        '''
        print 'Reading in information from the QM Energy file.'

        try:
            with open(self.energy_file,'r') as f:
                lines = [line.split() for line in f.readlines()]

            # Number of atoms for each monomer
            self.natoms1 = int(lines[0][0])
            self.natoms2 = int(lines[self.natoms1+1][0])

        except ValueError:
            print 'Error in reading the QM energy file.'
            print 'Did you switch the order of the parameter and energy files?\n'
            raise
        else:
            # Obtain element names from energy file
            self.atoms1 = [ lines[i][0] for i in xrange(1,self.natoms1+1)]
            self.atoms2 = [ lines[i][0] for i in xrange(self.natoms1+2,self.natoms1+self.natoms2+2)]

            if self.ignorecase:
                self.atoms1 = [ atom.upper() for atom in self.atoms1 ]
                self.atoms2 = [ atom.upper() for atom in self.atoms2 ]

            # Obtain geometry arrays from energy_file
            nlines = len(lines)
            self.ndatpts = lines.count([]) # count number of blank lines
            self.xyz1 = np.zeros((self.ndatpts,self.natoms1,3))
            self.xyz2 = np.zeros((self.ndatpts,self.natoms2,3))
            self.qm_energy = [ [] for i in xrange(self.ncomponents)]
            for i in xrange(self.ndatpts):
                # Monomer 1 geometry array:
                for j in xrange(self.natoms1):
                    k = i*nlines/self.ndatpts+j+1
                    self.xyz1[i,j,:] = np.array([float(lines[k][l]) for l in xrange(1,4)])
                # Monomer 2 geometry array:
                for j in xrange(self.natoms2):
                    k = i*nlines/self.ndatpts+j+self.natoms1+2
                    self.xyz2[i,j,:] = np.array([float(lines[k][l]) for l in xrange(1,4)])

                # QM Energy array:
                j = i*nlines/self.ndatpts+self.natoms1+self.natoms2+2

                self.qm_energy[0].append(float(lines[j+1][1])) # exchange 
                self.qm_energy[1].append(float(lines[j][1])) # electrostatics
                self.qm_energy[2].append(float(lines[j+4][1])+\
                                          float(lines[j+5][1])) # induction
                self.qm_energy[3].append(float(lines[j+17][1])) # dhf
                self.qm_energy[4].append(float(lines[j+7][1])+\
                                          float(lines[j+9][1])) # dispersion
                self.qm_energy[6].append(float(lines[j+12][1])) # E1tot+E2tot

            self.qm_energy = np.array([np.array(i) for i in self.qm_energy])

            # Use xyz1 and xyz2 arrays to compute the r array
            self.r12 = (self.xyz1[:,:,np.newaxis,:] - self.xyz2[:,np.newaxis,:,:])**2 
            self.r12 = np.sqrt(np.sum(self.r12,axis=-1))
            self.r12 = np.swapaxes(np.swapaxes(self.r12,0,2),0,1)


            # Add dhf energy to E1tot+E2tot to get the total interaction
            # energy:
            self.qm_energy[6] = self.qm_energy[3] + self.qm_energy[6]

            # Convert QM energies to Hartree from mH
            self.qm_energy /= 1000

            # Construct a list of all atoms present in the qm energy file
            self.atomtypes = set()
            for xyz in self.atoms1+self.atoms2:
                self.atomtypes.add(xyz)
            self.atomtypes = list(self.atomtypes)

        return
####################################################################################################    


####################################################################################################    
    def read_param_file(self):
        '''Read in hard constraints and parametesr from param_file.
        
        Parameters
        ----------
        None, aside from implicit dependence on param_file

        Returns
        -------
        None, though initializes values for the following class variables:

        Aparams : list of lists (ncomponents x nparams)
            Aij parameters for each energy component; nparams is variable and
            depends on how many anisotropic parameters have been read in for
            each atomtype.

        anisotropic_atomtypes : list
            Names of all atomtypes to be treated anisotropically.
        anisotropic_axes[1,2] : list of lists
            Axis specification (in terms of monomer indices) for each
            anisotropic atom. Will later be converted to an axis specification
            in terms of Cartesian coordinates.
        anisotropic_symmetries : dict
            For each aisotropic atom, lists which spherical harmonic terms
            should be fit.
        exponents[1,2] : list
            Bii parameters for each atom in each monomer.
        Cparams[1,2] : list of lists
            Cii parameters for each atom in each monomer.
        drude_charges[1,2] : list
            Drude charges for each atom in each monomer.
        fixed_atomtypes : list
            Atomtypes with hard constraints for A parameters.
        multipole_file[1,2]
            Name input file containing multipole moments for each monomer.
        eff_mu : float
            Weighting parameter; mu value in Fermi-Dirac function.
        eff_kt
            Weighting parameter; kt value in Fermi-Dirac function.

        '''
        print 'Reading in information from the parameter file.'

        # Initialize arrays for A,B,C parameters as well as charges
        self.Aparams = [ [] for i in xrange(self.ncomponents) ] # 4 components; exch, elst, ind, dhf
        self.anisotropic_atomtypes = []
        self.anisotropic_symmetries = {}
        self.exponents1 = []
        self.exponents2 = []
        self.Cparams1 = []
        self.Cparams2 = []
        self.drude_charges1 = []
        self.drude_charges2 = []

        self.Cparams = {}

        # Initialize list of all hard constraints
        self.fixed_atomtypes = {}

        with open(self.param_file,'r') as f:
            # Read in any changes to default parameters:
            f.readline()
            line = f.readline().split()
            while len(line) > 0:
                settyp = type(getattr(self,line[0]))
                if settyp == bool and line[1].lower() == 'false':
                    setattr(self,line[0],False)
                else:
                    setattr(self,line[0],settyp(line[1]))
                line = f.readline().split()

            # Read A parameters from file:
            #   Order is: Exchange, Electrostatics, Induction, DHF
            error = '''Atomtypes need to be defined in the same order and with
            the same atomtypes for each energy type (exchange, electrostatics,
            induction, dhf. Please fix your input file.'''
            f.readline()
            f.readline()
            line = f.readline().split()
            count = 0
            while line[0] != 'ELECTROSTATICS':
                self.fixed_atomtypes[line[0]] = count
                count += 1
                self.Aparams[0].append([float(i) for i in line[1:]])
                line = f.readline().split()
            line = f.readline().split()
            count = 0
            while line[0] != 'INDUCTION':
                if self.fixed_atomtypes[line[0]] != count:
                    print error
                    sys.exit('Program exiting.')
                self.Aparams[1].append([float(i) for i in line[1:]])
                count += 1
                line = f.readline().split()
            line = f.readline().split()
            count = 0
            while line[0] != 'DHF':
                if self.fixed_atomtypes[line[0]] != count:
                    print error
                    sys.exit('Program exiting.')
                self.Aparams[2].append([float(i) for i in line[1:]])
                count += 1
                line = f.readline().split()
            line = f.readline().split()
            count = 0
            print self.fit_residual_errors
            if self.fit_residual_errors:
                while line[0] != 'RESIDUALS':
                    if self.fixed_atomtypes[line[0]] != count:
                        print error
                        sys.exit('Program exiting.')
                    self.Aparams[3].append([float(i) for i in line[1:]])
                    count += 1
                    line = f.readline().split()
                count = 0
                line = f.readline().split()
                while len(line) > 0:
                    if self.fixed_atomtypes[line[0]] != count:
                        print error
                        sys.exit('Program exiting.')
                    self.Aparams[5].append([float(i) for i in line[1:]])
                    count += 1
                    line = f.readline().split()
            else:
                while len(line) > 0:
                    if self.fixed_atomtypes[line[0]] != count:
                        print error
                        sys.exit('Program exiting.')
                    self.Aparams[3].append([float(i) for i in line[1:]])
                    count += 1
                    line = f.readline().split()

            # Read in anisotropic atomtypes
            f.readline()
            while True:
                atom = f.readline().split()
                if atom ==[]:
                    break
                self.anisotropic_atomtypes.append(atom[0])
                self.anisotropic_symmetries[atom[0]] = atom[1:]
                if not atom[1:]: #make sure symmetry elements are actually declared
                    print \
                    'You must specify which spherical harmonic expansion terms you wish to include for anisotropic atomtype "'\
                    +atom[0]+'".' 
                    sys.exit()

            f.readline()
            f.readline()
            f.readline()
            line = f.readline().split()

            # Read in coordinate axes for each anisotropic atom. Ordering is
            # to list axes alphabetically (i.e. self.anisotropic_axes1 is
            # ordered x axis first, followed by the z axis.)
            self.anisotropic_axes1 = [ [ [],[] ] for i in xrange(self.natoms1)]
            self.anisotropic_axes2 = [ [ [],[] ] for i in xrange(self.natoms2)]
            # Read coordinate axes in for monomer 1
            while line[0:2] != ['monomer','2']: # monomer 1 axis definitions
                iatom = int(line[0])
                iaxis = 0 if line[1] == 'z' else 1 # list x and z axes seperately
                if self.anisotropic_axes1[iatom][iaxis] != []:
                    print 'The '+line[1]+' axis for atom '+line[0]+\
                            ' in monomer 1 has already been specified.'
                    print 'Please only use one axis specification line per axis per atom.'
                    sys.exit()
                else:
                    self.anisotropic_axes1[iatom][iaxis] = [ int(i) for i in line[2:] ]
                line = f.readline().split()
            # Read coordinate axes in for monomer 2
            line = f.readline().split()
            while line != []:
                iatom = int(line[0])
                iaxis = 0 if line[1] == 'z' else 1 # list x and z axes seperately
                if self.anisotropic_axes2[iatom][iaxis] != []:
                    print 'The '+line[1]+' axis for atom '+line[0]+\
                            ' in monomer 2 has already been specified.'
                    print 'Please only use one axis specification line per axis per atom.'
                    sys.exit()
                else:
                    self.anisotropic_axes2[iatom][iaxis] = [ int(i) for i in line[2:] ]
                line = f.readline().split()

            # Read exponents (B parameters) from file:
            f.readline()
            f.readline()
            for i in xrange(self.natoms1):
                b = float(f.readline().split()[1])
                self.exponents1.append(b)
            f.readline()
            for i in xrange(self.natoms2):
                b = float(f.readline().split()[1])
                self.exponents2.append(b)
            self.save_exponents = {}
            all_atoms = self.atoms1 + self.atoms2
            exponents = self.exponents1 + self.exponents2
            unique_atoms = set(all_atoms)
            print unique_atoms
            if self.handle_parameter_conflicts == 'average_exponents':
                for atom in unique_atoms:
                    self.save_exponents[atom] = np.mean([exponents[i] 
                                        for i in range(len(all_atoms)) 
                                        if all_atoms[i] == atom ])
            if self.handle_parameter_conflicts == 'raise_error':
                for i,atom in enumerate(all_atoms):
                    if self.save_exponents.has_key(atom) and \
                            self.save_exponents[atom] != exponents[i]:
                        error_msg = 'Not all exponents for atomtype '+atom+\
                        ' are the same! Make sure each atomtype has only one set of A and B parameters.'
                        sys.exit(error_msg)
                    else:
                        self.save_exponents[atom] = exponents[i]

            # Read dispersion coefficients (Cn parameters) from file:
            f.readline()
            f.readline()
            f.readline()
            for i in xrange(self.natoms1):
                constraint = f.readline().split()
                constraint = [float(i) for i in constraint[1:]]
                self.Cparams1.append(constraint)
            f.readline()
            for i in xrange(self.natoms2):
                constraint = f.readline().split()
                constraint = [float(i) for i in constraint[1:]]
                self.Cparams2.append(constraint)

            Cparams_all = np.sqrt(self.Cparams1 + self.Cparams2)
            for i,atom in enumerate(all_atoms):
                if self.Cparams.has_key(atom) and \
                        np.all(self.Cparams[atom] - Cparams_all[i]):
                    error_msg = 'Not all exponents for atomtype '+atom+\
                    ' are the same! Make sure each atomtype has only one set of A and B parameters.'
                    sys.exit(error_msg)
                else:
                    self.Cparams[atom] = Cparams_all[i]

            # Read multipole file names
            f.readline()
            f.readline()
            f.readline()
            self.multipole_file1 = f.readline().strip('\n')
            f.readline()
            self.multipole_file2 = f.readline().strip('\n')


            # Read drude oscillator charges:
            f.readline()
            f.readline()
            f.readline()
            for i in xrange(self.natoms1):
                self.drude_charges1.append(float(f.readline()))
            self.drude_charges1 = np.array(self.drude_charges1)
            f.readline()
            for i in xrange(self.natoms2):
                self.drude_charges2.append(float(f.readline()))
            self.drude_charges2 = np.array(self.drude_charges2)

            # Read parameters for the weighting function, namely eff_mu and eff_kt
            # charges):
            f.readline()
            f.readline()
            self.eff_mu = float(f.readline().split()[1])
            self.eff_kt = float(f.readline().split()[1])

        return
####################################################################################################    


####################################################################################################    
    def initialize_parameters(self):
        '''Initialize arrays, variables, and constants that will later be required in the
        fitting routines.

        More specifically, perform the following tasks:
            Compute cross-terms for exponents and dispersion coefficients.
            Create arrays to keep track of which atomtypes are isotropic/anisotropic 
                and which are to be fit/constrained.
            For each anisotropic atom, calculate a local axis (in Cartesian
                coordinates) for each dimer configuration along with theta and phi
                values between each atom pair (stored so that these values don't
                need to be needlessly re-computed during the optimization
                procedure).

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        # Compute cross-terms for exponents and dispersion coefficients.
        self.combine_exponents()
        # TEST
        ## self.combine_Cparams()

        # Read in multipole energies from orient, if necessary.
        if self.read_multipole_energy_from_orient:
            self.read_multipoles_from_orient(self.orient_multipolar_energy_file)

        # Determine list of parameters to fit; initialize arrays for both
        # isotropic and anisotropic parameters
        if self.ignorecase:
            self.atomtypes = [ atom.upper() for atom in self.atomtypes ]
            self.anisotropic_atomtypes = [ atom.upper() for atom in self.anisotropic_atomtypes ]
            self.fixed_atomtypes = { key.upper() : value \
                                for key, value in self.fixed_atomtypes.items() }
        self.isotropic_atomtypes = list(set(self.atomtypes) - set(self.anisotropic_atomtypes))
        self.fit_atomtypes = list(set(self.atomtypes) - \
                set(self.fixed_atomtypes.keys()))
        self.fit_anisotropic_atomtypes = list(set(self.anisotropic_atomtypes) - \
                        set(self.fixed_atomtypes.keys()))
        self.fit_isotropic_atomtypes = list(set(self.fit_atomtypes) - set(self.fit_anisotropic_atomtypes))
        self.constrained_anisotropic_atoms = list(set(self.anisotropic_atomtypes) - set(self.fit_anisotropic_atomtypes))

        # Determine, for each atom, whether or not the A parameters are
        # isotropic or anisotropic
        self.atoms1_anisotropic = []
        self.atoms2_anisotropic = []
        for i in xrange(self.natoms1):
            atom1 = self.atoms1[i]
            if atom1 in self.fit_anisotropic_atomtypes + self.constrained_anisotropic_atoms:
                self.atoms1_anisotropic.append(True)
            else:
                self.atoms1_anisotropic.append(False)
        for i in xrange(self.natoms2):
            atom2 = self.atoms2[i]
            if atom2 in self.fit_anisotropic_atomtypes + self.constrained_anisotropic_atoms:
                self.atoms2_anisotropic.append(True)
            else:
                self.atoms2_anisotropic.append(False)

        # For anisotropic atoms, create tables detailing the angular
        # dependence of each of these atomtypes. As this dependence won't
        # change over the course of parameter fitting, computing this
        # dependence now will save time later on in the simulation.
        self.create_angular_dependence_tables()

        # Print out summary of initialized parameters to the user
        print '############################################################'
        print '          Force Field Fitting Program Start'
        print '############################################################'
        print 'The program has located the following '+str(len(self.atomtypes))+\
                ' atomtypes, '
        print ' split into isotropic and anisotropic atomtypes as shown below.'
        string = '\tIsotropic Atomtypes:   '+'{:4s}'*len(self.isotropic_atomtypes)
        print string.format(*self.isotropic_atomtypes)
        string = '\tAnisotropic Atomtypes: '+'{:4s}'*len(self.anisotropic_atomtypes)
        print string.format(*self.anisotropic_atomtypes)
        print
        print 'Of the above atomtypes, the following will be fit:'
        print ' (the rest have been entered as constraints)'
        string = '\tIsotropic Atomtypes to Fit:   '+'{:4s}'*len(self.fit_isotropic_atomtypes)
        print string.format(*self.fit_isotropic_atomtypes)
        string = '\tAnisotropic Atomtypes to Fit: '+'{:4s}'*len(self.fit_anisotropic_atomtypes)
        print string.format(*self.fit_anisotropic_atomtypes)
        print
        print 'Each anisotropic term will include the following '
        print 'spherical harmonic terms:'
        for atom in self.anisotropic_atomtypes:
            sphharm = self.anisotropic_symmetries[atom]
            string = '\t{:>2s}:  '+'{:>5s},'*(len(sphharm)-1)+'{:>5s}'
            print string.format(atom,*sphharm)
        print

        print '############################################################'

        return
####################################################################################################    


####################################################################################################    
    def combine_exponents(self):
        '''Create cross-terms for exponents according to input combination
        rule.

        Parameters
        ----------
        None, though implictly depends on choice of bij_combination_rule

        Returns
        -------
        self.exponents : 2darray (natoms1 x natoms2)
            Array of exponents bij
        
        '''
        bi, bj = sp.symbols(" bi bj")
        self.combine_num_exponent = lambdify((bi,bj),self.combine_exponent(bi,bj,self.bij_combination_rule),modules='numpy')

        self.exponents = [ [] for i in xrange(self.natoms1)]
        for i,bi in enumerate(self.exponents1):
            for bj in self.exponents2:
                bij = self.combine_num_exponent(bi,bj)
                self.exponents[i].append(bij)

        self.exponents = np.array(self.exponents)
        return self.exponents
####################################################################################################    


####################################################################################################    
    def recalculate_exponents(self):
        '''Recalculate Bij parameters using combination rules.

        This subroutine only gets called after exponents have been fit (in the
        exchange portion of the FF) and are to be constrained for all other
        component fits. 
        
        Parameters
        ----------
        None

        Returns
        -------
        None, though self.exponents is updated

        '''
        self.exponents1 = [ self.atom_params[atom][1] for atom in self.atoms1]
        self.exponents2 = [ self.atom_params[atom][1] for atom in self.atoms2]

        self.exponents = [ [] for i in xrange(self.natoms1)]
        for i,bi in enumerate(self.exponents1):
            for bj in self.exponents2:
                bij = self.combine_num_exponent(bi,bj)
                self.exponents[i].append(bij)

        self.fit_bii = False
        self.n_isotropic_params -= 1
        self.exponents = np.array(self.exponents)

        return
####################################################################################################    


####################################################################################################    
    def combine_exponent(self,bi,bj,combination_rule='geometric_mean'):
        '''Use combination rule for exponents (see below) to explicitly
        create Bij values.

        Parameters
        ----------
        bi : symbol or float
            Exponent for atom i on monomer 1.
        bj : symbol or float
            Exponent for atom j on monomer 2.
        combination_rule : str, optional.
            Combination rule. Default 'geometric_mean'. 'saptff',
            'waldman-hagler5', and 'arithmetic_mean' are also options.

        Returns
        -------
        bij : symbol or float
            Exponent to describe interaction between atom i and j on monomers
            1 and 2, respectively.

        '''
        if combination_rule == 'saptff':
            bij = (bi + bj)*bi*bj/(bi**2 + bj**2)
        elif combination_rule == 'waldman-hagler5':
            bij = (2/(bi**(-5) + bj**(-5)))**(1.0/5)
        elif combination_rule == 'geometric_mean':
            bij = sp.sqrt(bi*bj)
        elif combination_rule == 'arithmetic_mean':
            bij = (bi + bj)/2
        else:
            print combination_rule + ' not a known combination rule.'
            sys.exit()

        return bij
####################################################################################################    


####################################################################################################    
    def combine_prefactor(self,ai,aj,bi,bj,bij,combination_rule='geometric'):
        '''Uses combination rule for prefactors (see below) to explicitly
        create Aij cross-terms.

        Parameters
        ----------
        ai : symbol or float
            Pre-factor for atom i on monomer 1.
        aj : symbol or float
            Pre-factor for atom j on monomer 2.
        bi : symbol or float
            Exponent for atom i on monomer 1.
        bj : symbol or float
            Exponent for atom j on monomer 2.
        combination_rule : str, optional.
            Combination rule. Default 'geometric'. 'saptff',
            'waldman-hagler5', and 'arithmetic' are also options.

        Returns
        -------
        aij : symbol or float
            Pre-factor to describe interaction between atom i and j on monomers
            1 and 2, respectively.

        '''
        if 'geometric' in combination_rule:
            aij = ai*aj
        elif combination_rule == 'saptff':
            #aij = (ai*aj)**(0.5)
            aij = ai*aj
        elif combination_rule == 'waldman-hagler5':
            aij = ai*aj*(bij**6/(bi**3*bj**3))
        elif 'arithmetic' in combination_rule:
            aij = ai + aj
        elif combination_rule == 'stone':
            aij = (ai + aj)/bij
        else:
            print combination_rule + ' not a known combination rule.'
            sys.exit()


        return aij
####################################################################################################    


####################################################################################################    
    def combine_Cparam(self,ci,cj,combination_rule='geometric'):
        '''Use combination rule for dispersion coefficients (see below) to explicitly
        create C^n_{ij} values.

        Parameters
        ----------
        None; implicit dependence on Cparams[1,2]

        Returns
        -------
        None, though class variable Cparams is initialized.
        Cparams : 3d list
            List ( natoms1 x natoms2 x #Cncoeffs(usually 4) ) of all
            dispersion coefficients C^n_{ij}. This parameters are read in, not
            optimized, and always obey a geometric mean combination rule, so
            this routine only gets called once.

        '''

        if combination_rule == 'geometric':
            cij = ci*cj
        else:
            print combination_rule + ' not a known combination rule.'
            sys.exit()

        return cij
        ## # Initialize array:
        ## self.Cparams = [[ [] for j in xrange(self.natoms2) ] 
        ##                         for i in xrange(self.natoms1) ]
        ## # Combination rule for Cparams is as follows:
        ## #      Cn_ij = sqrt(Cn_i*Cn_j) 
        ## for i,ci in enumerate(self.Cparams1):
        ##     for j,cj in enumerate(self.Cparams2):
        ##         constraint  = [ np.sqrt(k[0]*k[1]) for k in zip(ci,cj)]
        ##         self.Cparams[i][j] = constraint
        ## return
####################################################################################################    


####################################################################################################    
    def get_radial_correction(self,rij,bi,bj,tol=1e-3):
        '''Use a polynomial based on the overlap of two Slater orbitals to
        calculate a radial correction to the Aij parameters.

        References
        ----------
        For the exact radial correction, formula obtained from 
        (1) Rosen, N.  Phys. Rev. Lett. 1931, 38, 255-276.  
        and a communication with Alston Misquitta. 

        Work by Tai (Tai, H. Phys. Rev. A 1986, 33, 3657-3666.) is also
        helpful in this regard, particularly the formula for K2 (p. 3658) for
        the overlap of two s orbitals.

        For the approximate radial correction, polynomial in the limiting case
        of bi=bj is used. The hope is that, given a correct combination rule,
        this form will also hold for cases where bi != bj. Current work
        suggests that a geometric combination rule is preferred here.

        Parameters
        ----------
        rij : 1darray or symbol
            Inter-site distance between i (on monomer 1) and j (on monomer 2)
        bi : float or symbol
            Exponent for atom i on monomer 1.
        bj : float or symbol
            Exponent for atom j on monomer 2.
        tol : float, optional
            Absolute difference in bi and bj at which the two will be
            considered equal. Only important if exact_radial_correction is set
            to True. Default 1e-3.

        Returns
        -------
        slater_overlap : symbol or 1darray
            Expression or array of numeric values (for each dimer
            configuration) for the radial correction.

        '''
        bi_equal_bj = (bi - bj < tol)
        bij = self.combine_exponent(bi,bj,self.bij_combination_rule)

        if self.exact_radial_correction and self.fit_bii:
            # If we're scaling bii, we need to evaluate the radial correction as a
            # piecewise function, in case bi and bj alternate between being
            # in and outside of tolerance
            test = (bi - bj > tol)
            return sp.Piecewise(\
                    (functional_forms.get_exact_slater_overlap(bi,bj,rij),test),\
                    (functional_forms.get_approximate_slater_overlap(bij,rij), True))
        elif self.exact_radial_correction:
            if bi_equal_bj:
                return functional_forms.get_approximate_slater_overlap(bij,rij)
            else:
                return functional_forms.get_exact_slater_overlap(bi,bj,rij)

        else:
            return functional_forms.get_approximate_slater_overlap(bij,rij)
####################################################################################################    


####################################################################################################    
    def create_angular_dependence_tables(self):
        '''Create a table of theta and phi angles for each anisotropic atom
        interacting with every other atom in the system.
        
        Specifically, given a list of anisotropic atoms, and a well-defined set of
        coordinate axes for each atom (using internal coordinates), computes
        the theta and phi angles for each anisotropic atom interacting with
        every other atom in the system.

        For example, consider an anisotropic element A1 in monomer 1, and an
        additional anisotropic element A2 in monomer 2. This subroutine will
        calculate theta1 and phi1, the azimuthal and polar angular coordinates
        for A2 under the coordinate system defined for A1. Note that these
        angles are different than theta2 and phi2, the angular coordinates for
        A1 with the coordinate system defined for A2.

        Note that, for an isotropic atom with index i, self.angles[i,:,:] will simply be
        a subarray of zeros.

        Parameters
        ----------
        None, though implicitly depends on coordinates input from
        anisotropic_axes[1,2].

        Returns
        -------
        self.angles1 : 4darray (natoms1 x natoms2 x 2 x ndatpts)
            Theta (self.angles1[i,j,0]) and phi (self.angles1[i,j,1]) values
            for atom j in monomer 2 under the coordinate system of atom i in
            monomer 1.
        self.angles2 : 4darray (natoms2 x natoms1 x 2 x ndatpts)
            Theta (self.angles2[j,i,0]) and phi (self.angles1[j,i,1]) values
            for atom i in monomer 1 under the coordinate system of atom j in
            monomer 2.
        
        '''
        print 'Setting up table of angular dependencies for anisotropic atoms.'

        # Initialize angles array
        self.angles1 = np.zeros((self.natoms1,self.natoms2,2,self.ndatpts))
        self.angles2 = np.zeros((self.natoms2,self.natoms1,2,self.ndatpts))

        # Compute angular dependence table for atoms in monomer 1
        for i in xrange(self.natoms1):
            if self.atoms1_anisotropic[i]:
                #if self.anisotropic_axes1[i][0] == [] or self.anisotropic_axes1[i][1] == []:
                if self.anisotropic_axes1[i][0] == []:
                    print '----------- WARNING -------------------'
                    print 'No Z axis has been specified for atom '+str(i)+' in monomer 1.'
                    print 'Please specify a z axis for this atom.'
                    print '---------------------------------------'
                    sys.exit()
                # Depending on whether the axes specification has 2 or 3
                # indices listed, we'll treat the axes as the vector between 2
                # points (in the case of a 2 index list) or as the bisecting
                # vector for the angle defined by the 3 index list. 
                if len(self.anisotropic_axes1[i][0]) == 2:
                    iatom1 = self.anisotropic_axes1[i][0][0]
                    iatom2 = self.anisotropic_axes1[i][0][1]
                    z_axis = self.xyz1[:,iatom2,:] - self.xyz1[:,iatom1,:]
                    z_axis /= np.sqrt((z_axis ** 2).sum(-1))[..., np.newaxis] #Normalize
                elif len(self.anisotropic_axes1[i][0]) > 2:
                    iatom1 = self.anisotropic_axes1[i][0][0]
                    z1 = self.xyz1[:,iatom1,:]
                    z2 = np.mean([self.xyz1[:,j,:] for j in self.anisotropic_axes1[i][0][1:]],axis=0)
                    z_axis = z2 - z1
                    z_axis /= np.sqrt((z_axis ** 2).sum(-1))[..., np.newaxis] #Normalize

                else:
                    print 'You must specify exactly two or three atomic indices for atom ' + str(i) + ' in monomer 1.' 
                    print 'The program does not know how to handle more or less atomic indices than what you have prescribed.'
                    sys.exit()

                if len(self.anisotropic_axes1[i][1]) == 2:
                    iatom1 = self.anisotropic_axes1[i][1][0]
                    iatom2 = self.anisotropic_axes1[i][1][1]
                    vec = self.xyz1[:,iatom2,:] - self.xyz1[:,iatom1,:]
                    x_axis = self.project_onto_plane(vec,z_axis)
                elif len(self.anisotropic_axes1[i][1]) == 3:
                    iatom1 = self.anisotropic_axes1[i][1][0]
                    iatom2 = self.anisotropic_axes1[i][1][1]
                    iatom3 = self.anisotropic_axes1[i][1][2]
                    vec = self.get_bisecting_vector(iatom1,iatom2,iatom3)
                    x_axis = self.project_onto_plane(vec,z_axis)
                elif len(self.anisotropic_axes1[i][1]) == 0:
                    vec = np.array([1,0,0])
                    if np.array_equal(z_axis, vec) or np.array_equal(z_axis, -vec):
                        vec = np.array([0,1,0])
                    x_axis = self.project_onto_plane(vec,z_axis)
                    print 'Since no x-axis was specified for atom ' \
                            + str(i) + ' in monomer 1,'
                    print 'assuming that atomtype no x/y angular dependence.'
                else:
                    print 'You must specify exactly zero, two, or three atomic indices for each atom.' 
                    print 'The program does not know how to handle more or less atomic indices than what you have prescribed.'
                    sys.exit()

                for j in xrange(self.natoms2):
                    theta, phi = self.get_angle(i,z_axis,x_axis,j,mon1=True)
                    self.angles1[i][j][0,:] = theta
                    self.angles1[i][j][1,:] = phi

            else:
                if self.anisotropic_axes1[i] != [[],[]]:
                    print 'Coordinate axes cannot be used for isotropic atoms.'
                    print 'Please either list atom '+str(i)+\
                            ' in monomer 1 as anisotropic or remove the line' +\
                            ' specifying a coordinate axis for this atom.'
                    sys.exit()

        # Compute angular dependence table for atoms in monomer 2
        for i in xrange(self.natoms2):
            if self.atoms2_anisotropic[i]:
                if self.anisotropic_axes2[i][0] == []:
                    print '----------- WARNING -------------------'
                    print 'No Z axis has been specified for atom '+str(i)+' in monomer 2.'
                    print 'Please specify a z axis for this atom.'
                    print '---------------------------------------'
                    sys.exit()
                # Depending on whether the axes specification has 2 or 3
                # indices listed, we'll treat the axes as the vector between 2
                # points (in the case of a 2 index list) or as (in the case of
                # a > 2 index list) the vector between the first point and the
                # midpoint of the remaining points. Thus, for example, with
                # water one could define the z axis using the oxygen atom
                # (index 0) and the midpoint between the two hydrogen atoms
                # (indices 1 and 2) by specifying the axis as 0 z 0 1 2
                if len(self.anisotropic_axes2[i][0]) == 2:
                    iatom1 = self.anisotropic_axes2[i][0][0]
                    iatom2 = self.anisotropic_axes2[i][0][1]
                    z_axis = self.xyz2[:,iatom2,:] - self.xyz2[:,iatom1,:]
                    z_axis /= np.sqrt((z_axis ** 2).sum(-1))[..., np.newaxis] #Normalize
                elif len(self.anisotropic_axes2[i][0]) > 2:
                    iatom1 = self.anisotropic_axes2[i][0][0]
                    z1 = self.xyz2[:,iatom1,:]
                    z2 = np.mean([self.xyz2[:,j,:] for j in self.anisotropic_axes2[i][0][1:]],axis=0)
                    z_axis = z2 - z1
                    z_axis /= np.sqrt((z_axis ** 2).sum(-1))[..., np.newaxis] #Normalize
                else:
                    print 'You must specify exactly two or three atomic indices for atom ' + str(i) + ' in monomer 2.' 
                    print 'The program does not know how to handle more or less atomic indices than what you have prescribed.'
                    sys.exit()

                if len(self.anisotropic_axes2[i][1]) == 2:
                    iatom1 = self.anisotropic_axes2[i][1][0]
                    iatom2 = self.anisotropic_axes2[i][1][1]
                    vec = self.xyz2[:,iatom2,:] - self.xyz2[:,iatom1,:]
                    x_axis = self.project_onto_plane(vec,z_axis)
                elif len(self.anisotropic_axes2[i][1]) == 3:
                    iatom1 = self.anisotropic_axes2[i][1][0]
                    iatom2 = self.anisotropic_axes2[i][1][1]
                    iatom3 = self.anisotropic_axes2[i][1][2]
                    vec = self.get_bisecting_vector(iatom1,iatom2,iatom3,mon1=False)
                    x_axis = self.project_onto_plane(vec,z_axis)
                elif len(self.anisotropic_axes2[i][1]) == 0:
                    vec = np.array([1,0,0])
                    if np.array_equal(z_axis, vec) or np.array_equal(z_axis, -vec):
                        vec = np.array([0,1,0])
                    x_axis = self.project_onto_plane(vec,z_axis)
                    print 'Since no x-axis was specified for atom ' \
                            + str(i) + ' in monomer 2,'
                    print 'assuming that atomtype no x/y angular dependence.'
                else:
                    print 'You must specify exactly zero, two or three atomic indices for atom ' + str(i) + ' in monomer 2.' 
                    print 'The program does not know how to handle more or less atomic indices than what you have prescribed.'
                    sys.exit()

                for j in xrange(self.natoms1):
                    theta, phi = self.get_angle(i,z_axis,x_axis,j,mon1=False)
                    self.angles2[i][j][0,:] = theta
                    self.angles2[i][j][1,:] = phi

            else:
                if self.anisotropic_axes2[i] != [[],[]]:
                    print 'Coordinate axes cannot be used for isotropic atoms.'
                    print 'Please either list atom '+str(i)+\
                            ' in monomer 2 as anisotropic or remove the line' +\
                            ' specifying a coordinate axis for this atom.'
                    sys.exit()

        return self.angles1, self.angles2

####################################################################################################    


####################################################################################################    
    def get_bisecting_vector(self, iatom1, iatom2, iatom3, mon1=True):
        '''Given 3 atom indices for atom1, atom2, and atom3, compute the
        bisecting angle between the vector atom1-atom2 and atom2-atom3 for
        each data point.

        Parameters
        ----------
        iatom[1,2,3] : int
            Atom index for atom[1,2,3]

        mon1 : bool, optional
            Determines whether atoms correspond to monomer 1 (default) or
            monomer 2.

        Returns
        -------
        bisector : 2darray (ndatpts x 3)
            Bisecting vector for each data point.
        an Ndatpts x 3 array containing the bisecting vector at each
        data point.
        '''

        # TODO: This subroutine is uneccessary (see how I code in z-axis, for
        # instance, and should be removed after I get rid of dependencies. 

        if mon1:
            xyz_atom1 = self.xyz1[:,iatom1,:]
            xyz_atom2 = self.xyz1[:,iatom2,:]
            xyz_atom3 = self.xyz1[:,iatom3,:]
        else:
            xyz_atom1 = self.xyz2[:,iatom1,:]
            xyz_atom2 = self.xyz2[:,iatom2,:]
            xyz_atom3 = self.xyz2[:,iatom3,:]

        vec1 = xyz_atom1 - xyz_atom2
        vec1 /= np.sqrt((vec1 ** 2).sum(-1))[..., np.newaxis]

        vec2 = xyz_atom3 - xyz_atom2
        vec2 /= np.sqrt((vec2 ** 2).sum(-1))[..., np.newaxis]

        # To see why this works, check out
        # https://proofwiki.org/wiki/Angle_Bisector_Vector
        bisector = vec1 + vec2
        bisector /= np.sqrt((bisector ** 2).sum(-1))[..., np.newaxis]

        return bisector
####################################################################################################    


####################################################################################################    
    def project_onto_plane(self,vec, z_axis):
        '''Project a vector vec onto a plane defined by its normal vector z_axis.

        References
        ----------
        http://www.euclideanspace.com/maths/geometry/elements/plane/lineOnPlane/

        Parameters
        ----------
        vec : array
            Vector to be projected onto the plane
        z_axis : array
            Normal vector defining the xy plane onto which vec will be
            projected.

        Returns
        -------
        plane_vec : array
            Projected vector.
        
        '''
        # See http://www.euclideanspace.com/maths/geometry/elements/plane/lineOnPlane/
        # for projecting a line onto a plane defined by its normal vector. The
        # magnitude and direction of this projection are given as below.
        direction = np.cross(z_axis,np.cross(vec,z_axis))

        # In the case where a vector is perfectly in line with the z-axis,
        # return some default value for the x-axis
        mask = np.all(direction==[0,0,0], axis=1)
        direction[mask] = [1,0,0]

        direction /= np.sqrt((direction ** 2).sum(-1))[..., np.newaxis]
        cos_angle = np.sum(vec*z_axis,axis=1) / np.sqrt((vec ** 2).sum(-1))
        angle = np.arccos(cos_angle)
        magnitude = np.sqrt((vec ** 2).sum(-1))*np.sin(angle)


        return magnitude[:,np.newaxis]*direction
####################################################################################################    


####################################################################################################    
    def get_angle(self,iatom1,z_axis,x_axis, iatom2, mon1=True):
        '''Obtain theta (azimuthal) and phi (polar) angular coordinates of
        atom2 in the coordinate system defined by z_axis and x_axis whose
        origin is at atom1.

        Parameters
        ----------
        iatom1 : int
            Atom index for atom1.
        z_axis : array
            Vector defining the z-axis.
        x_axis : array
            Vector defining the x-axis. Assumed to be orthogonal to z_axis.
        iatom2 : int
            Atom index for atom2.
        mon1 : bool, optional.
            Determines whether atom1 belongs to monomer 1 (default) or monomer
            2 (if mon1=False).

        Returns
        -------
        theta : 1darray (ndatpts)
            Theta values (azimuthal angle) for each dimer configuration.
        phi : 1darray (ndatpts)
            Phi values (polar angle) for each dimer configuration.
        
        '''
        # TODO : Ensure that z-axis and x-axis are orthogonal.

        # Define vector between atom1 and atom2
        if mon1:
            vec = - self.xyz1[:,iatom1,:] + self.xyz2[:,iatom2,:]
        else:
            vec = - self.xyz2[:,iatom1,:] + self.xyz1[:,iatom2,:]

        # Calculate Theta Angle
        theta_vec = self.project_onto_plane(vec,z_axis)
        # Angle between vectors (from -pi to pi) given by atan2 function
        # http://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
        dot = np.sum(vec*x_axis,axis=1) #gives cos_theta
        det = np.linalg.det(np.array([vec.T,x_axis.T,z_axis.T]).T) #gives sin_theta
        theta =  np.arctan2(det, dot)
        # For convenience, let theta have xrange [0,2pi] instead of [-pi,pi]
        theta = theta % (2*np.pi)
        
        # Calculate Phi Angle using definition of the dot product (a.b = |a||b|*cos(phi) )
        cos_phi = np.sum(vec*z_axis,axis=1)
        cos_phi /= np.sqrt((vec ** 2).sum(-1))
        cos_phi /= np.sqrt((z_axis ** 2).sum(-1))
        phi = np.arccos(cos_phi)

        return theta, phi
####################################################################################################    



####################################################################################################    
    def get_drude_oscillator_energy(self):
        '''Obtain the interaction energy due to long-range polarization.

        Specifically, for our force fields, we model the polarization energy
        (at long-range, where charge penetration can be neglected) by drude
        oscillators (that is, a charge on a spring model). More details can be
        found in the drude_oscillators module itself. Here, we call the
        drude_oscillators module (currently several modules exist, depending
        on speed considerations and whether or not we want to include
        higher-order multipole moments).

        Note that the drude oscillator energy is split into two-components: a
        2nd order component (edrude_ind) that is put in the induction portion
        of the FF energy, and a higher-order component (edrude_dhf) that is
        included in the DHF FF energy. Details on this splitting can be found
        in the drude_oscillator module.

        Parameters
        ----------
        None

        Returns
        -------
        self.edrude_ind : 1darray (ndatpts)
            2nd order drude oscillator energy.
        self.edrude_dhf : 1darray (ndatpts)
            Higher order drude oscillator energy.

        '''
        # If all drude charges are zero, skip oscillator convergence:
        if np.allclose(self.drude_charges1,np.zeros_like(self.drude_charges1)) \
            and np.allclose(self.drude_charges2,np.zeros_like(self.drude_charges2)):
            print 'No drude oscillators, so skipping oscillator convergence step.'
            self.edrude_ind = np.zeros_like(self.qm_energy[6])
            self.edrude_dhf = np.zeros_like(self.qm_energy[6])
            return self.edrude_ind, self.edrude_dhf

        # Otherwise, get drude oscillator energy from other modules:
        if not self.only_scale_exchange_bii:
            print 'WARNING: Exponents used in TT damping function arise from '+\
            'the exchange fit, and have not been optimized for the drude '+\
            'oscillators in particular.'
        if self.drude_method == 'multipole-gradient':
            print 'Calculating drude oscillator energy using a multipole-gradient method'
            from drude_oscillators import Drudes
            #from finite_differences_drude_oscillators import FDDrudes as Drudes
            d = Drudes(self.xyz1, self.xyz2, 
                        self.multipole_file1, self.multipole_file2,
                        self.drude_charges1, self.drude_charges2, 
                        self.exponents,
                        self.thole_param, self.springcon,
                        self.slater_correction,
                        self.electrostatic_damping_type)
            self.edrude_ind, self.edrude_dhf = d.get_induction_and_dhf_drude_energy()
        elif self.drude_method == 'finite-differences':
            print 'Calculating drude oscillator energy using finite-differences'
            from debug.finite_differences_drude_oscillators import FDDrudes as Drudes
            #from finite_differences_drude_oscillators import FDDrudes as Drudes
            d = Drudes(self.xyz1, self.xyz2, 
                        self.multipole_file1, self.multipole_file2,
                        self.drude_charges1, self.drude_charges2, 
                        self.exponents,
                        self.thole_param, self.springcon,
                        self.slater_correction,
                        self.electrostatic_damping_type)
            self.edrude_ind, self.edrude_dhf = d.get_induction_and_dhf_drude_energy()
        elif self.drude_method == 'pointcharge-gradient':
            print 'Calculating drude oscillator energy.'
            from debug.pointcharge_drude_oscillators import Drudes 
            # TODO: charges1 and charges2 are outdated and would need to be
            # read in from multipole_file[1,2]
            d = Drudes(self.xyz1, self.xyz2, 
                        self.charges1, self.charges2,
                        self.drude_charges1, self.drude_charges2, 
                        self.exponents,
                        self.thole_param, self.springcon,
                        self.slater_correction,
                        self.electrostatic_damping_type)
            self.edrude_ind, self.edrude_dhf = d.get_induction_and_dhf_drude_energy()
        elif self.drude_method == 'read':
            print 'Reading in drude oscillator energy from ',self.drude_file
            with open(self.drude_file,'r') as f:
                data = np.array([ [float(i) for i in line.split()] 
                                    for line in f.readlines()[1:] ])
                self.edrude_ind = data[:,0]
                self.edrude_dhf = data[:,1]
        else:
            raise NotImplementedError

        if self.drude_method != 'read':
            with open('edrudes.dat','w') as f:
                f.write('Edrude_ind \t\t Edrude_dhf\n')
                for i in xrange(len(self.edrude_ind)):
                    f.write('{:16.8f} {:16.8f}\n'.format(self.edrude_ind[i],self.edrude_dhf[i]))
            with open('drude_positions.dat','w') as f:
                f.write('Shell_xyz1 positions\n')
                for line in d.shell_xyz1-d.xyz1:
                    np.savetxt(f,line)
                    f.write('---\n')
                f.write('Shell_xyz2 positions\n')
                for line in d.shell_xyz2-d.xyz2:
                    np.savetxt(f,line)
                    f.write('---\n')

        return self.edrude_ind, self.edrude_dhf
####################################################################################################    


####################################################################################################    
    def fit_component_parameters(self):
        '''Optimize FF parameters for a given energy component to best fit, in
        a weighted least-squares sense, the corresponding QM energy component.

        Specifically, this high-level subroutine operates as follows:
        1. Subtract off energy due to hard constraints from the total QM
        energy.
        2. Assign initial guesses to all remaining parameters that will be
        fit, ordered in such a way to take advantage of scipy's optimization
        routines.
        3. Optimize parameters (using scipy.minimize) in a
        weighted-leastsquares sense, potentially using soft constraints (set
        using self.constrain_ab_positive = True). The weighting function is as
        described in functional_forms.weight.

        Parameters
        ----------
        None

        Returns
        -------
        None, though Aparams (and possibly Bij values) have been fit and
        output to file.

        '''

        print '-------------'
        print 'Optimizing parameters for ' + self.energy_component_names[self.component]
        print '-------------'

        # To speed up calculations, subtract off energy that is already known
        # on the basis of hard constraints.
        qm_fit_energy = self.subtract_hard_constraint_energy()

        # Add additional parameters for scaling exponents, if necessary
        if self.fit_bii:
            # Add one additional parameter per atomtype to account for scaling
            # exponents
            self.n_isotropic_params += 1

        # Determine total number of parameters to be fit
        ntot_isotropic_params = self.n_isotropic_params*len(self.fit_isotropic_atomtypes)
        ntot_anisotropic_params = sum([len(v) + self.n_isotropic_params \
                                    for k,v in self.anisotropic_symmetries.items() \
                                    if k in self.fit_anisotropic_atomtypes])
        ntot_params = ntot_isotropic_params + ntot_anisotropic_params

        # Set soft-constraints
        if self.constrain_ab_positive:
            if self.functional_form == 'born-mayer':
                abound = (0,1e3)
            else:
                abound = (-1e3,1e3)
            bbound = (1e-2,1e2)
            unbound = (None,None)
            # For isotropic atomtypes, constrain all parameters to be positive
            # For anisotropic atomtypes, only constrain first (and possibly
            # last) parameters (corresponding to A and B, respectively) to be
            # positive
            if self.fit_bii:
                bounds_iso =[ [abound for i in range(self.n_isotropic_params-1)] + 
                                [bbound] 
                                    for j in self.fit_isotropic_atomtypes ]
                bounds_aniso =[ [abound for i in range(self.n_isotropic_params-1)] + 
                                [unbound for i in v] +
                                [bbound] 
                               for k,v in self.anisotropic_symmetries.items() \
                                    if k in self.fit_anisotropic_atomtypes ]
            else:
                bounds_iso = [ [abound] for i in range(ntot_isotropic_params) ]
                bounds_aniso =[ [abound for i in range(self.n_isotropic_params)] + 
                                [unbound for i in v]
                               for k,v in self.anisotropic_symmetries.items() \
                                    if k in self.fit_anisotropic_atomtypes ]
            tmp = []
            for i in bounds_iso:
                tmp.extend(i)
            bounds_iso = tmp
            tmp = []
            for i in bounds_aniso:
                tmp.extend(i)
            bounds_aniso = tmp

            bnds = bounds_iso + bounds_aniso

        # Perform initial energy call to set up function and derivative
        # subroutines
        p0=np.array([1 for i in xrange(ntot_params)])
        self.qm_fit_energy = np.array(qm_fit_energy)

        self.final_energy_call = False
        self.calc_ff_energy(p0, init=True)

        # Use scipy.optimize to perform a least-squares fitting:
        # Initial paramaters are given by p0, and the weighted least squares
        # fitting procedure is given in a subroutine below. Weights here are
        # given by a Fermi-Dirac distribution.
        if len(p0): # Only perform fitting if there are unconstrained parameters
            print 'Optimizing parameters:'
            # pgtol and ftol control convergence criteria. Good convergence
            # seems to be achievable with the values shown here, although in
            # certain cases (particularly if a fit does not look good) these
            # values can be made smaller in order to tighten convergence
            # criteria.
            maxiter=5000
            pgtol=1e-15 
            ftol=1e-17

            # *Finally*, we're ready to perform the least-squares fitting
            # procedure:
            if self.constrain_ab_positive:
                res = minimize(self.calc_leastsq_ff_fit,p0,method='L-BFGS-B',\
                        jac=True,\
                        options={'disp':True,'gtol':pgtol,'ftol':ftol,'maxiter':maxiter},\
                        bounds=bnds)
            else:
                res = minimize(self.calc_leastsq_ff_fit,p0,method='L-BFGS-B',\
                        jac=True,\
                        options={'disp':True,'gtol':pgtol,'ftol':ftol,'maxiter':maxiter})
            popt = res.x
            success = res.success
            message = res.message

            if not res.success:
                print 'Warning! Optimizer did not terminate successfully, and quit with the following error message:'
                print
                print res.message
                print

        else:
            print 'All parameters are constrained, so skipping parameter optimization step.'
            popt = []
            success = True
            message = ''

        # Print out results into output file
        self.final_energy_call = True
        # Here ff_fit_energy is the force field energy, minus constrained
        # energy. ff_energy is the total force field energy for a given energy
        # component.
        ff_fit_energy = self.calc_ff_energy(popt)[0]
        ff_energy = np.array(self.qm_energy[self.component])\
                            -np.array(qm_fit_energy) + ff_fit_energy

        self.rms_error = self.calc_rmse(ff_energy)
        self.weighted_absolute_error = self.calc_mae(ff_energy, cutoff=self.weighted_rmse_cutoff)
        self.weighted_rms_error = self.calc_rmse(ff_energy, cutoff=self.weighted_rmse_cutoff)
        self.lsq_error = self.calc_leastsq_ff_fit(popt)[0]

        print '------'
        print 'RMS Error for the fit to ',\
                self.energy_component_names[self.component] + ':'
        print '{:.5e}'.format(self.rms_error)

        self.write_output_file(success,message)
        self.write_energy_file(ff_energy)

        if self.fit_bii and self.only_scale_exchange_bii and self.component == 0:
            self.recalculate_exponents()

        return ff_energy
####################################################################################################    


####################################################################################################    
    def subtract_hard_constraint_energy(self,tol=1e-3):
        '''Calculate energy known on the basis of hard constraints; eliminate
        this energy from the energy to be fit.

        Subtracting the hard constraint energy in this manner is purely
        employed for reasons of computational efficiency.

        Parameters
        ----------
        None; note that the routine behaves somewhat different depending on
        what energy component we are fitting. 

        Returns
        -------
        qm_fit_energy : 1darray (ndatpts)
            QM energy from which FF parameters will actually be optimized. If
            needed, the hard constraint energy can be obtained by subtracting
            qm_fit_energy from self.qm_energy[component] for a given
            component.

        '''

        print 'Subtracting off energy from hard constraints.'
        # Initialize fit array
        qm_fit_energy = np.copy(self.qm_energy[self.component])

        if self.component == 4: #dispersion
            return qm_fit_energy

        # Certain required functions are written symbolically (which will be
        # important during the optimization), but need to be calculated
        # numerically here. The lambdify function converts these symbolic
        # functions to ones that can be evaulated numerically for this
        # subroutine.
        aij, rij, bij, bi, bj = sp.symbols("aij rij bij bi bj")
        get_num_eij = \
            lambdify((aij,rij,bij),functional_forms.get_eij(self.component,aij,rij,bij,self.functional_form,self.slater_correction),modules='numpy')
        if self.slater_correction and self.exact_radial_correction:
            get_exact_radial_correction = \
                    lambdify((bi,bj,rij),functional_forms.get_exact_slater_overlap(bi,bj,rij),modules='numpy')
        if self.slater_correction:
            get_approx_radial_correction = \
                    lambdify((bij,rij),functional_forms.get_approximate_slater_overlap(bij,rij),modules='numpy')

        # Iterate over atom pairs, subtracting off energies where all relevant A and B
        # parameters are already known, and skipping energy evaluation for
        # atom pairs with any free parameters.
        charge_energy = np.zeros_like(qm_fit_energy)
        for i in xrange(self.natoms1):
            for j in xrange(self.natoms2):
                atom1 = self.atoms1[i]
                atom2 = self.atoms2[j]
                xi = self.xyz1[:,i,:]
                xj = self.xyz2[:,j,:]
                bi = self.exponents1[i]
                bj = self.exponents2[j]
                bij = self.exponents[i][j]
                rij = (xi - xj)**2
                rij = np.sqrt(np.sum(rij,axis=1))

                try:
                    if self.ignorecase:
                        atom1 = atom1.upper()
                        atom2 = atom2.upper()
                    iatom1 = self.fixed_atomtypes[atom1]
                    iatom2 = self.fixed_atomtypes[atom2]

                except KeyError:
                    # Ignore atom pairs without hard constraints
                    continue
                else:
                    Ai = self.Aparams[self.component][iatom1]
                    Aj = self.Aparams[self.component][iatom2]
                    if self.atoms1_anisotropic[i]:
                        sph_harm = self.anisotropic_symmetries[atom1]
                        a = Ai[0]
                        Aangular = Ai[1:]
                        theta = self.angles1[i,j,0,:]
                        phi = self.angles1[i,j,1,:]
                        ai = functional_forms.get_anisotropic_ai(sph_harm,a,Aangular,rij,theta,phi)
                    else: #if isotropic
                        ai = Ai[0]
                    if self.atoms2_anisotropic[j]:
                        sph_harm = self.anisotropic_symmetries[atom2]
                        a = Aj[0]
                        Aangular = Aj[1:]
                        theta = self.angles2[j,i,0,:]
                        phi = self.angles2[j,i,1,:]
                        aj = functional_forms.get_anisotropic_ai(sph_harm,a,Aangular,rij,theta,phi)
                    else: #if isotropic
                        aj = Aj[0]

                    aij = self.combine_prefactor(ai,aj,bi,bj,bij,self.aij_combination_rule)

                    if self.slater_correction:
                        if not self.exact_radial_correction or bi - bj < tol:
                            a_rad = get_approx_radial_correction(bij,rij)
                        else:
                            a_rad = get_exact_radial_correction(bi,bj,rij)
                    else:
                        a_rad = 1

                    qm_fit_energy -= a_rad*get_num_eij(aij,rij,bij)


        # For induction and DHF, subtract off drude oscillator energy
        if self.component == 2:
            print 'Subtracting off 2nd order drude oscillator energy'
            qm_fit_energy -= self.edrude_ind
            
        if self.component == 3:
            print 'Subtracting off higher order drude oscillator energies'
            qm_fit_energy -= self.edrude_dhf

        if self.component == 1:
            if self.read_multipole_energy_from_orient:
                qm_fit_energy -= self.multipole_energy
                try:
                    assert self.electrostatic_damping_type == 'None'
                except AssertionError:
                    print 'Damping type needs to be None for consistency with the Orient program.'
                    raise
                #self.damping_type = 'None' #right now Orient doesn't damp electrostatics
            else:
                m = Multipoles(self.xyz1,self.xyz2,self.multipole_file1,self.multipole_file2,
                            self.exponents,self.slater_correction,self.electrostatic_damping_type)
                qm_fit_energy -= m.get_multipole_electrostatic_energy()
                #self.damping_type = m.damping_type 

        return qm_fit_energy
####################################################################################################    
               

####################################################################################################    
    def calc_rmse(self,ff_energy,cutoff=None):
        '''Compute the root-mean-square (RMS) error between the QM and FF
        energies.

        Paramters
        ---------
        ff_energy : 1darray
            Force field energy for a given component
        cutoff : None or float
            If None, returns the RMS error for the entire data set. If given
            as a float, data points with total interaction energies (QM)
            larger than cutoff are excluded from the fit.

        Returns
        -------
        rms_error : float
            RMS error (weighted according to cutoff) for the fit

        '''
        if cutoff == None:
            weight = np.ones_like(ff_energy)
        else:
            i_eint = 6
            weight = (self.qm_energy[i_eint] < cutoff) # Boolean mask to exclude high-energy points

        rms_error = np.sqrt(np.average(weight*(self.qm_energy[self.component] - ff_energy)**2))

        return rms_error
####################################################################################################    


####################################################################################################    
    def calc_mae(self,ff_energy,cutoff=None):
        '''Compute the mean absolute error (MAE) between the QM and FF
        energies.

        Paramters
        ---------
        ff_energy : 1darray
            Force field energy for a given component
        cutoff : None or float
            If None, returns the RMS error for the entire data set. If given
            as a float, data points with total interaction energies (QM)
            larger than cutoff are excluded from the fit.

        Returns
        -------
        mae : float
            MAE (weighted according to cutoff) for the fit

        '''

        if cutoff == None:
            weight = np.ones_like(ff_energy)
        else:
            i_eint = 6
            weight = (self.qm_energy[i_eint] < cutoff) # Boolean mask to exclude high-energy points

        mae = np.average(weight*(self.qm_energy[self.component] - ff_energy))
        return mae
####################################################################################################    


####################################################################################################    
    def calc_harmonic_constraint_error(self,params,k=1e-5):
        '''Calculate the error penalty associated with any employed harmonic
        constraints. 
        
        Limitations
        -----------
        1. Currently only implemented for B parameters.
        2. Current k value chosen somewhat arbitrarily; may be worth tweaking
        in the future.


        Paramters
        ---------
        params : dictionary of lists
            Dictionary containing parameters for each atomtype. Called here to
            obtain B parameters, which can be compared to the input B
            parameters.
        k : float
            Weighitng parameter; higher k values employ more stringent
            constraints on the values B can take.

        Returns
        -------
        harmonic_error : float
            Energy penalty associated with employed harmonic constraints.
        dharmonic_Error : 1darray (nparams)
            Derivatives of the harmonic error with respect to each B
            parameter.

        '''
        # Update self.atom_params array
        self.output_params(params)
        # TODO: I probably should call map_params here (or replace map_params
        # entirely with output_params

        # Calculate error penalty for b params, assuming harmonic constraints.
        # Trick here is to make sure derivatives for each constraint get
        # mapped onto the correct parameter
        harmonic_error = 0.0
        dharmonic_error = []
        count = 0
        shift = self.n_isotropic_params
        for atom in self.fit_isotropic_atomtypes:
            dharmonic_error.extend(0 for i in xrange(count,count+shift))
            b = self.atom_params[atom][-1]
            b0 = self.save_exponents[atom]
            harmonic_error += k*(b - b0)**2
            dharmonic_error[-1] = 2*k*(b - b0)*b0
            count += shift
        for atom in self.fit_anisotropic_atomtypes:
            shift = self.n_isotropic_params + len(self.anisotropic_symmetries[atom])
            dharmonic_error.extend(0 for i in xrange(count,count+shift))
            b = self.atom_params[atom][-1]
            b0 = self.save_exponents[atom]
            harmonic_error += k*(b - b0)**2
            dharmonic_error[-1] = 2*k*(b - b0)*b0
            count += shift

        dharmonic_error = np.array(dharmonic_error)

        return harmonic_error, dharmonic_error
####################################################################################################    


####################################################################################################    
    def calc_leastsq_ff_fit(self,params):
        '''Compute weighted least squares error for a given set of force field
        parameters.

        Note that this leastsq energy may also include the energy penalty due
        to harmonic constraints.

        Parameters
        ----------
        params : 1d tuple 
            Tuple containing all fit parameters 

        Returns
        -------
        lsq_error : float
            Least-squares error (+ harmonic constraint penalty)
        dlsq_Error : 1darray (nparams)
            Derivative of the lsq error with respect to each fit parameter.

        '''
        xdata = xrange(len(self.qm_energy[self.component]))
        ff_fit_energy, dff_fit_energy = self.calc_ff_energy(params)
        ff_energy = np.array(self.qm_energy[self.component])\
                            -np.array(self.qm_fit_energy) + ff_fit_energy
        qm_energy = self.qm_energy[self.component]
        weight = functional_forms.weight(qm_energy, self.eff_mu, self.eff_kt)
        lsq_error =  weight*(ff_energy - qm_energy)**2
        try:
            dlsq_error = 2*weight*(ff_energy - qm_energy)*dff_fit_energy
        except ValueError: # if params = []
            dlsq_error = 0

        lsq_error = np.sum(lsq_error)
        dlsq_error = np.sum(dlsq_error,axis=-1)

        if self.fit_bii and self.harmonic_constraints:
            harmonic_error, dharmonic_error = self.calc_harmonic_constraint_error(params)
            lsq_error += harmonic_error
            dlsq_error += dharmonic_error

        return lsq_error, dlsq_error
####################################################################################################    


####################################################################################################    
    def calc_ff_energy(self,params,init=False):
        '''Compute the force field energy for a given component (exchange,
        induction, etc.) given a set of parameters. Also return the gradient
        of the FF energy with respect to each fit parameter.

        In more detail, this subroutine works as follows:
        The first time this function gets called (init=True), the SymPy
        package is used to symbollically evaluate the force field energy and
        automatically compute the gradient. Once these symbolic quantities
        have been calculated, the lamdify function is called to generate
        (fast) numerical subroutines for calculation of the energy and
        gradient as a function of parameter values. All subsequent calls to
        calc_ff_energy use these numerical subroutines when calculating the
        force field energy.

        Parameters
        ----------
        params : 1d tuple 
            Tuple containing all fit parameters 
        init : bool, optional
            Detailed above, first call (init=True) to function requires
            symbolic evaluation of function derivatives. Default False.

        Returns
        -------
        ff_energy : 1darray (ndatpts)
            Force field energy for each dimer configuration.
        dff_energy : list of 1darrays (nparams x ndatpts)
            Derivative of the force field energy with respect to each
            parameter.

        '''

        if init:
            num_params = params
            params = sp.symbols('p0:%d'%len(params))
            param_symbols = params
            params = self.map_params(params)

            # Construct a mapping between atomtypes to fit and the
            # corresponding index of params that contains parameters for that
            # atomtype
            param_map = {}
            paramindex = 0
            for atomtype in self.fit_isotropic_atomtypes + self.fit_anisotropic_atomtypes:
                if self.ignorecase:
                    param_map[atomtype.upper()] = paramindex
                    paramindex += 1
                else:
                    param_map[atomtype] = paramindex
                    paramindex += 1


            # Next we create a mapping between each atom in self.atoms1/2 and the set of parameters
            # to which it corresponds. This mapping occurs for all of the atoms
            # in both atoms1 and atoms2:
            A_atoms1 = []
            A_atoms2 = []
            self.skip_atom1 = []
            self.skip_atom2 = []
            for i in xrange(self.natoms1):
                atom1 = self.atoms1[i]
                if self.ignorecase:
                    atom1 = atom1.upper()
                # Look for atomtype atom1 in list of atoms to fit and set A; if not found,
                # flag atom1 as a constrained atomtype:
                try:
                    A_atoms1.append(params[param_map[atom1]])
                    self.skip_atom1.append(False)
                except KeyError:
                    ai = self.Aparams[self.component][self.fixed_atomtypes[atom1]]
                    A_atoms1.append(ai)
                    self.skip_atom1.append(True)

            for i in xrange(self.natoms2):
                atom2 = self.atoms2[i]
                if self.ignorecase:
                    atom2 = atom2.upper()
                try:
                    A_atoms2.append(params[param_map[atom2]])
                    self.skip_atom2.append(False)
                except KeyError:
                    ## ai = [self.Aparams[self.fixed_atomtypes[atom2]][self.component]]
                    ## ai = [self.Aparams[self.component][self.fixed_atomtypes[atom2]]]
                    ai = self.Aparams[self.component][self.fixed_atomtypes[atom2]]
                    A_atoms2.append(ai)
                    self.skip_atom2.append(True)

            # Now that we know which set of parameters to use for each atomtype,
            # we can finally compute the component energy of the system for each
            # data point:
            # Declare some symbols
            r = sp.symbols('r_0:%d_0:%d'%(self.natoms1,self.natoms2))
            theta1 = sp.symbols('theta1_0:%d_0:%d'%(self.natoms1,self.natoms2))
            phi1 = sp.symbols('phi1_0:%d_0:%d'%(self.natoms1,self.natoms2))
            theta2 = sp.symbols('theta2_0:%d_0:%d'%(self.natoms2,self.natoms1))
            phi2 = sp.symbols('phi2_0:%d_0:%d'%(self.natoms2,self.natoms1))

            print 'Symbolically evaluating FF energy.'
            ff_energy = 0.0
            for i in xrange(self.natoms1):
                Ai = A_atoms1[i]
                atom1 = self.atoms1[i]
                for j in xrange(self.natoms2):
                    Aj = A_atoms2[j]
                    atom2 = self.atoms2[j]

                    # Check if Ai and Aj originate from constrained
                    # parameters. If so, don't evaluate their energy, as this
                    # energy was already subtracted off earlier in the
                    # program. Otherwise, add their energy to the total
                    # ff energy, and compute 
                    if self.skip_atom1[i] and self.skip_atom2[j]:
                        continue

                    ij = i*self.natoms2 + j
                    ji = j*self.natoms1 + i
                    if self.component != 4:
                        eij = self.calc_sym_eij(i,j,r[ij], Ai, Aj, \
                                            theta1[ij], theta2[ji], phi1[ij], phi2[ji])
                    else:
                        eij = self.calc_sym_disp_ij(i,j,r[ij], Ai, Aj, \
                                            theta1[ij], theta2[ji], phi1[ij], phi2[ji])
                    ff_energy += eij

            # Use sympy to compute the deriviative of ff_energy
            print 'Symbolically evaluating the derivative of the FF energy.'
            dff_energy = [sp.diff(ff_energy,p) for p in param_symbols]

            # Lambdify the ff_energy and dff_energy functions
            args = param_symbols, r, theta1, theta2, phi1, phi2
            print 'Calculating numeric energies.'
            self.evaluate_ff_energy = \
                lambdify(flatten(args), ff_energy, modules='numpy')

            print 'Calculating numeric 1st derivatives.'
            self.evaluate_dff_energy = \
                [ lambdify(flatten(args), i, modules='numpy') for i in dff_energy ]
            print 'Finished calculating numeric energies and 1st derivatives'

            # Turn multi-dimensional params list back into a 1D list
            params = num_params


        # By the time this part of the code is called, numeric subroutines
        # self.evaluate_ff_energy and self.evaluate_dff_energy should have
        # already been generated. Since these functions take 1D arrays as
        # input, we need to flatten some of our multi-dimensional arrays, such
        # as r, before they can be used as inputs.
        r = self.r12
        theta1 = self.angles1[:,:,0,:]
        phi1 = self.angles1[:,:,1,:]
        theta2 = self.angles2[:,:,0,:]
        phi2 = self.angles2[:,:,1,:]

        r = [ [j for j in i ] for i in r]
        r_flat = list(itertools.chain.from_iterable(r))
        theta1 = [ [j for j in i ] for i in theta1]
        theta1_flat = list(itertools.chain.from_iterable(theta1))
        theta2 = [ [j for j in i ] for i in theta2]
        theta2_flat = list(itertools.chain.from_iterable(theta2))
        phi1 = [ [j for j in i ] for i in phi1]
        phi1_flat = list(itertools.chain.from_iterable(phi1))
        phi2 = [ [j for j in i ] for i in phi2]
        phi2_flat = list(itertools.chain.from_iterable(phi2))
        vals = list(params) + r_flat + theta1_flat + theta2_flat + phi1_flat + phi2_flat

        # Actual call to evaluate force field energy and gradient occurs here.
        try:
            ff_energy = self.evaluate_ff_energy(*vals)
            dff_energy = [ dp(*vals) for dp in self.evaluate_dff_energy ]
        except (FloatingPointError,ZeroDivisionError):
            print params
            raise

        # TODO ADD this block of code to lambdify'd version to check for
        # erroneous negative energies (which should only arise for anisotropy)
        ## eij_min = np.amin(eij)
        ## if self.final_energy_call and eij_min < 0:
        ##     print 'WARNING: Negative pairwise exchange energies encountered.'
        ##     print '     Lowest energy exchange energy:', eij_min

        if self.verbose and not self.final_energy_call:
            print 'Current parameter values:'
            for i,atom in enumerate(self.fit_isotropic_atomtypes+self.fit_anisotropic_atomtypes):
                print atom, params[i]
        if self.final_energy_call:
            if len(params):
                print 'Final parameter values:'
                for i,atom in enumerate(self.fit_isotropic_atomtypes+self.fit_anisotropic_atomtypes):
                    print atom, self.map_params(params)[i]
            self.output_params(params)
            self.final_energy_call = False
        
        return ff_energy, dff_energy
####################################################################################################    


####################################################################################################    
    def map_params(self,params):
        '''Map a 1d tuple of parameters onto a list of lists, such that the ith
        item in the return list corresponds to all A and B free parameters for
        the ith atom to be fit.

        Parameters
        ----------
        params : 1dtuple (symbolic or numeric)
            Tuple of all free parameters.

        Returns
        -------
        mapped_params : list of lists
            2d ordered list such that mapped_params[i] corresponds to the ith
            atomtype with free parameters (isotropic listed first, followed by
            anisotropic atomtypes)

        '''
        count = 0
        mapped_params = []
        shift = self.n_isotropic_params
        for atom in self.fit_isotropic_atomtypes:
            mapped_params.append([params[i] for i in xrange(count,count+shift)])
            count += shift
        for atom in self.fit_anisotropic_atomtypes:
            shift = self.n_isotropic_params + len(self.anisotropic_symmetries[atom])
            mapped_params.append([params[i] for i in xrange(count,count+shift)])
            count += shift

        return mapped_params
####################################################################################################    


####################################################################################################    
    def calc_sym_eij(self, i, j, rij, Ai, Aj, theta1, theta2, phi1, phi2):
        '''Symbollically compute the pairwise interaction energy between atom
        i in monomer 1 and atom j in monomer 2.

        Parameters
        ----------
        i : int
            Index for atom i in monomer 1.
        j : int
            Index for atom j in monomer 2.
        rij : symbol
            Interatomic distance between atoms i and j.
        Ai : list of symbols
            List of all A parameters for atom i.
        Aj : list of symbols
            List of all A parameters for atom j.
        theta1 : symbol
            Azimuthal angle (in the local coordinate system of atom i) of atom
            i with respect to atom j.
        theta2 : symbol
            Azimuthal angle (in the local coordinate system of atom j) of atom
            j with respect to atom i.
        phi1 : symbol
            Polar angle (in the local coordinate system of atom i) of atom
            i with respect to atom j.
        phi2 : symbol
            Polar angle (in the local coordinate system of atom j) of atom
            j with respect to atom i.
        
        Returns
        -------
        eij : symbolic expression
            Pairwise interaction energy for a given component between atoms i
            and j.

        '''
        # Calculate exponent
        if self.fit_bii:
            if not self.skip_atom1[i]:
                bi = Ai[-1]*self.exponents1[i] # exponent scaling factor is last parameter
            else:
                bi = self.exponents1[i]
            if not self.skip_atom2[j]:
                bj = Aj[-1]*self.exponents2[j] # exponent scaling factor is last parameter
            else:
                bj = self.exponents2[j]
            bij = self.combine_exponent(bi,bj,self.bij_combination_rule)
        else:
            bi = self.exponents1[i]
            bj = self.exponents2[j]
            bij = self.exponents[i][j]

        # Calculate the A coefficient for each atom. This
        # coefficient is computed differently if the atom is
        # isotropic or anisotropic. 
        if self.atoms1_anisotropic[i]:
            sph_harm = self.anisotropic_symmetries[self.atoms1[i]]
            a = Ai[0]
            if (not self.skip_atom1[i]) and self.fit_bii:
                Aangular = Ai[1:-1]
            else:
                Aangular = Ai[1:]
            ai = functional_forms.get_anisotropic_ai(sph_harm, a,Aangular,rij,theta1,phi1)
        else: #if isotropic
            ai = Ai[0]
        if self.atoms2_anisotropic[j]:
            sph_harm = self.anisotropic_symmetries[self.atoms2[j]]
            a = Aj[0]
            if (not self.skip_atom2[j]) and self.fit_bii:
                Aangular = Aj[1:-1]
            else:
                Aangular = Aj[1:]
            aj = functional_forms.get_anisotropic_ai(sph_harm,a,Aangular,rij,theta2,phi2)
        else: #if isotropic
            aj = Aj[0]

        aij = self.combine_prefactor(ai,aj,bi,bj,bij,self.aij_combination_rule)

        # Get radial pre-factor correction
        if self.slater_correction:
            a_rad = self.get_radial_correction(rij,bi,bj)
        else:
            a_rad = 1

        # Calculate the ff energy for the atom pair.
        eij = functional_forms.get_eij(self.component,aij,rij,bij,
                            self.functional_form,self.slater_correction)

        eij *= a_rad

        return eij
####################################################################################################    


####################################################################################################    
    def calc_dispersion_energy(self):
        '''Calculate the FF energy due to dispersion.

        Dispersion is calcuated as a series of 1/r^n (n=6,8,10,12) terms.

        Parameters
        ----------
        None

        Returns
        -------
        dispersion_energy : 1darray (ndatpts)
            Dispersion energy for each dimer configuration.

        '''
        if self.component != 4:
            print 'This subroutine should not be called except to calculate the dispersion energy (self.component = 4).'
            sys.exit()

        if self.exact_radial_correction:
            print 'Dispersion energies are not implemented for the exact Slater overlap; Program exiting.'
            sys.exit()

        print 'Evaluating Dispersion energies from input Cn coefficients.'

        dispersion_energy = np.zeros_like(self.qm_energy[self.component])
        for i in xrange(self.natoms1):
            atom1 = self.atoms1[i]
            for j in xrange(self.natoms2):
                atom2 = self.atoms2[j]

                ci = self.Cparams[atom1]
                cj = self.Cparams[atom2]
                cij = self.combine_Cparam(ci,cj,self.cij_combination_rule)
                rij = self.r12[i][j]
                # Note, for now the dispersion energy calculation doesn't
                # allow for optimization of bij parameters.
                bij = self.exponents[i][j]

                for n in range(6,14,2):
                    cijn = cij[n/2 -3]
                    dispersion_energy += \
                        functional_forms.get_dispersion_energy(n,cijn,rij,bij,
                                    self.slater_correction)

        # Output results to files
        self.write_energy_file(dispersion_energy)
        self.rms_error = self.calc_rmse(dispersion_energy)
        self.weighted_rms_error = self.calc_rmse(dispersion_energy, cutoff=self.weighted_rmse_cutoff)
        self.weighted_absolute_error = self.calc_mae(dispersion_energy, cutoff=self.weighted_rmse_cutoff)
        self.lsq_error = 0.0
        self.write_output_file(dispersion_energy)


        return dispersion_energy
####################################################################################################    


####################################################################################################    
    def calc_sym_disp_ij(self, i, j, rij, Ai, Aj, theta1, theta2, phi1, phi2):
        '''Symbollically compute the pairwise interaction energy between atom
        i in monomer 1 and atom j in monomer 2.

        Parameters
        ----------
        i : int
            Index for atom i in monomer 1.
        j : int
            Index for atom j in monomer 2.
        rij : symbol
            Interatomic distance between atoms i and j.
        Ai : list of symbols
            List of all A parameters for atom i.
        Aj : list of symbols
            List of all A parameters for atom j.
        theta1 : symbol
            Azimuthal angle (in the local coordinate system of atom i) of atom
            i with respect to atom j.
        theta2 : symbol
            Azimuthal angle (in the local coordinate system of atom j) of atom
            j with respect to atom i.
        phi1 : symbol
            Polar angle (in the local coordinate system of atom i) of atom
            i with respect to atom j.
        phi2 : symbol
            Polar angle (in the local coordinate system of atom j) of atom
            j with respect to atom i.
        
        Returns
        -------
        eij : symbolic expression
            Pairwise dispersion energy between atoms i and j.

        '''
        # Calculate exponent
        if self.fit_bii:
            if not self.skip_atom1[i]:
                bi = Ai[-1]*self.exponents1[i] # exponent scaling factor is last parameter
            else:
                bi = self.exponents1[i]
            if not self.skip_atom2[j]:
                bj = Aj[-1]*self.exponents2[j] # exponent scaling factor is last parameter
            else:
                bj = self.exponents2[j]
            bij = self.combine_exponent(bi,bj,self.bij_combination_rule)
        else:
            bi = self.exponents1[i]
            bj = self.exponents2[j]
            bij = self.exponents[i][j]

        # Calculate the A coefficient for each atom. This
        # coefficient is computed differently if the atom is
        # isotropic or anisotropic. 
        eij = 0 
        for n in range(6,14,2):
            if self.atoms1_anisotropic[i]:
                sph_harm = self.anisotropic_symmetries[self.atoms1[i]]
                a = self.Cparams[self.atoms1[i]][n/2-3]
                if (not self.skip_atom1[i]) and self.fit_bii:
                    Aangular = Ai[0:-1]
                else:
                    Aangular = Ai[0:]
                ai = functional_forms.get_anisotropic_ai(sph_harm, a,Aangular,rij,theta1,phi1)
            else: #if isotropic
                ai = self.Cparams[self.atoms1[i]][n/2-3]
            if self.atoms2_anisotropic[j]:
                sph_harm = self.anisotropic_symmetries[self.atoms2[j]]
                a = self.Cparams[self.atoms2[j]][n/2-3]
                if (not self.skip_atom2[j]) and self.fit_bii:
                    Aangular = Aj[0:-1]
                else:
                    Aangular = Aj[0:]
                aj = functional_forms.get_anisotropic_ai(sph_harm,a,Aangular,rij,theta2,phi2)
            else: #if isotropic
                aj = self.Cparams[self.atoms2[j]][n/2-3]

            aij = self.combine_prefactor(ai,aj,bi,bj,bij,self.aij_combination_rule)

            # Calculate the ff energy for the atom pair.
            eij += functional_forms.get_dispersion_energy(n,aij,rij,bij,
                                self.slater_correction)

        return eij
####################################################################################################    


####################################################################################################    
    def output_params(self,params):
        '''Map a 1dtuple (corresponding to both free and constrainted
        parameters for each atomtypes) onto a dictionary of lists.

        Parameters
        ----------
        params : 1d tuple 
            Tuple containing all fit parameters 

        Returns
        -------
        self.atom_params : dict
            Dictionary whose keys are each atomtype and whose values are a
            list of A and B parameters for that atomtype. Due to anisotropy,
            there may exist multiple A parameters, but only one B parameter
            should be present; B parameters are listed last.

        '''
        params = self.map_params(params)
        self.atom_params = {}
        exponents= { atom: exp for (atom,exp) in zip(self.atoms1, self.exponents1) }
        exponents2= { atom: exp for (atom,exp) in zip(self.atoms2, self.exponents2) }
        # Combine exponents1 and exponents 2. Note this may cause
        # overwriting values if the same atomtype has multiple exponents
        # (which is a problem for other reasons...)
        exponents.update(exponents2)

        # Collect parameters from the fitted atomtypes
        for i,atom in enumerate(self.fit_isotropic_atomtypes+self.fit_anisotropic_atomtypes):

            b = exponents[atom]

            if self.fit_bii:
                Aparams = params[i][:-1]
                b *= params[i][-1]
            else:
                Aparams = params[i]

            self.atom_params[atom] = [Aparams, b]

        # Collect parameters from constrained atomtypes
        for atom in self.fixed_atomtypes:
            iatom = self.fixed_atomtypes[atom]
            Aparams = self.Aparams[self.component][iatom]
            b = exponents[atom]
            self.atom_params[atom] = [Aparams, b]

        return self.atom_params
####################################################################################################    


####################################################################################################    
    def write_output_file(self,success=True,message=''):
        '''Write a summary of the fitting procedure to an output file. 
        
        The output_params function should be run prior to running this
        subroutine in order to collect the parameters.

        Parameters
        ----------
        success : bool, optional.
            If False, writes warning message (see below) to file to indicate
            that an optimization did not terminate successfully.
        message : str, optional
            Message to print in the case that the optimization did not
            terminate successfully.

        Returns
        -------
        None

        '''
        short_break = '------------\n'
        long_break = '--------------------------------------------\n'

        with open(self.output_file,'a') as f:

            if self.component == 0:
                f.write('########################## FF Fitting Summary ###########################\n')
                f.write(long_break)
                f.write('Program Version: '+ __version__ + '\n')
                f.write('Short-range Functional Form: '+str(self.functional_form)+'\n')
                f.write('Combination Rules: aij = '+str(self.aij_combination_rule)+'\n')
                f.write('                   bij = '+str(self.bij_combination_rule)+'\n')
                f.write('Electrostatic Damping Type: '+str(self.electrostatic_damping_type)+'\n')
                f.write('Fitting weight: eff_mu = '+str(self.eff_mu)+' Ha\n')
                f.write('                eff_kt = '+str(self.eff_kt)+' Ha\n')
                f.write('Weighted RMSE cutoff: '+str(self.weighted_rmse_cutoff)+' Ha\n')
                if self.anisotropic_atomtypes:
                    template = '{:5s}'*len(self.anisotropic_atomtypes) + '\n'
                    f.write('Anisotropic Atomtypes: ' + \
                                str(template.format(*self.anisotropic_atomtypes)))
                else:
                    f.write('Anisotropic Atomtypes: None\n')

                if self.slater_correction:
                    f.write('All parameters have been radially corrected to account for Slater overlap\n')
                    if self.exact_radial_correction:
                        f.write('This radial correction is the exact overlap as computed in Rosen et al.\n')
                    else:
                        f.write('This radial correction is approximate, and is only rigorous for the case where bi = bj.\n')
                f.write(short_break)
                if self.fit_bii:
                    f.write('Exponents (Optimized):\n')
                else:
                    f.write('Exponents:\n')
                for (atom,param) in self.atom_params.items():
                    template = '{:5s}{:8.6f}\n'
                    bi = param[1]
                    f.write(template.format(atom,bi))
                f.write(short_break)
                f.write('Monomer 1 Multipole File:\n')
                f.write(self.multipole_file1 + '\n')
                f.write('Monomer 2 Multipole File:\n')
                f.write(self.multipole_file1 + '\n')
                f.write(long_break)
            
                # Exchange Parameters
                f.write('Exchange Parameters:\n')
                f.write('    Functional Form = \n')
                if self.slater_correction:
                    f.write('\tE(exch)_ij = A*K2(rij)*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')
                    f.write('    where the a coefficient for each spherical harmonic term Y_ml\n')
                    f.write('    is listed in the parameters below and \n')
                    f.write('\tK2(rij) = 1/3*(bij*rij)**2 + bij*rij + 1 \n')
                else:
                    f.write('\tE(exch)_ij = A*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')
                    f.write('    where the a coefficient for each spherical harmonic term Y_ml\n')
                    f.write('    is listed in the parameters below.\n')

            # Electrostatic Parameters
            elif self.component == 1:
                if self.read_multipole_energy_from_orient:
                    f.write('Note: Multipole moments have been read in from a previous ORIENT calculation.\n')
                f.write('Electrostatic Parameters:\n')
                f.write('    Functional Form = \n')
                if self.slater_correction:
                    f.write('\tE(elst)_ij = f_damp*qi*qj/rij - A*K2(rij)*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')
                else:
                    f.write('\tE(elst)_ij = f_damp*qi*qj/rij - A*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')

            # Induction Parameters
            elif self.component == 2:
                f.write('Drude oscillator energy has been calculated using the following method: ' + self.drude_method + '\n')
                f.write('Induction Parameters:\n')
                f.write('    Functional Form = \n')
                if self.slater_correction:
                    f.write('\tE(ind)_ij = shell_charge - A*K2(rij)*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')
                else:
                    f.write('\tE(ind)_ij = shell_charge - A*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')

            # DHF Parameters
            elif self.component == 3:
                f.write('DHF Parameters:\n')
                f.write('    Functional Form = \n')
                if self.slater_correction:
                    f.write('\tE(dhf)_ij = - A*K2(rij)*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')
                else:
                    f.write('\tE(dhf)_ij = - A*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')

            # Dispersion Parameters
            elif self.component == 4:
                #print dispersion parameters
                f.write('Dispersion Parameters:\n')
                f.write('    Functional Form = \n')
                f.write('\tE(disp)_ij = sum_(n=6,8,10,12){fdamp_n*(Cij_n/r_ij^n)}\n')

            # DHF Parameters
            elif self.component == 5:
                f.write('Residual Error Parameters:\n')
                f.write('    Functional Form = \n')
                if self.slater_correction:
                    f.write('\tE(residual)_ij = - A*K2(rij)*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')
                else:
                    f.write('\tE(residual)_ij = - A*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')

            elif self.component == 6:
                f.write('Total Energy:\n')
            else:
                print 'Writing output file not yet implemented for component ',self.component
                sys.exit()

            # Write fitting parameters to file
            if self.component not in [4,6]: # Fits not applicable for dispersion, total_energy
                f.write('Fitted Atomtypes \n')
                for i,atom in enumerate(self.fit_isotropic_atomtypes):
                    template='{:5s}   '+'{:^16s}\n'
                    f.write(template.format('','A'))
                    template='{:5s}'+'{:16.6f}'*len(self.atom_params[atom][0])+'\n'
                    f.write(template.format(atom,*self.atom_params[atom][0]))
                for i,atom in enumerate(self.fit_anisotropic_atomtypes):
                    template='{:5s}   '+'{:^16s}'*(len(self.anisotropic_symmetries[atom])+1)+'\n'
                    args = ['a_' + y for y in self.anisotropic_symmetries[atom] ]
                    f.write(template.format('','A',*args))
                    template='{:5s}'+'{:16.6f}'*len(self.atom_params[atom][0])+'\n'
                    f.write(template.format(atom,*self.atom_params[atom][0]))
                if not self.fit_atomtypes:
                    f.write('  None\n')

                f.write('Constrained Atomtypes \n')
                for atom in self.fixed_atomtypes:
                    if atom in self.anisotropic_atomtypes:
                        template='{:5s}   '+'{:^16s}'*(len(self.anisotropic_symmetries[atom])+1)+'\n'
                        args = ['a_' + y for y in self.anisotropic_symmetries[atom] ]
                        f.write(template.format('','A',*args))
                    else:
                        template='{:5s}   '+'{:^16s}\n'
                        f.write(template.format('','A'))
                    i = self.fixed_atomtypes[atom]
                    template='{:5s}'+'{:16.6f}'*len(self.Aparams[self.component][i])+'\n'
                    f.write(template.format(atom,*self.Aparams[self.component][i]))
                if not self.fixed_atomtypes:
                    f.write('  None\n')

                if not success:
                    f.write('Warning! Optimizer did not terminate successfully, but rather quit with the following error message:\n')
                    f.write(message + '\n')
            elif self.component == 4:
                template='{:5s}   '+'{:^16s}'*4 +'\n'
                f.write(template.format('Mon1','C6','C8','C10','C12'))
                for atom,cn_coeffs in zip(self.atoms1,self.Cparams1):
                    template='{:5s}'+'{:16.6f}'*len(cn_coeffs)+'\n'
                    f.write(template.format(atom,*cn_coeffs))
                template='{:5s}   '+'{:^16s}'*4 +'\n'
                f.write(template.format('Mon2','C6','C8','C10','C12'))
                for atom,cn_coeffs in zip(self.atoms2,self.Cparams2):
                    template='{:5s}'+'{:16.6f}'*len(cn_coeffs)+'\n'
                    f.write(template.format(atom,*cn_coeffs))
            else:
                # Nothing to print right now for total energy
                pass

            # This section applicable to all fits
            template = '{:s} RMS Error: '+'{:.5e}'+'\n'
            f.write(short_break)
            f.write(template.format(self.energy_component_names[self.component], self.rms_error))
            template = '{:s} Weighted RMS Error: '+'{:.5e}'+'\n'
            f.write(template.format(self.energy_component_names[self.component], self.weighted_rms_error))
            template = '{:s} Weighted Absolute Error: '+'{:.5e}'+'\n'
            f.write(template.format(self.energy_component_names[self.component],
                self.weighted_absolute_error))
            template = '{:s} Weighted Least-Squares Error: '+'{:.5e}'+'\n'
            f.write(template.format(self.energy_component_names[self.component], self.lsq_error))
            f.write(long_break)

            if self.component == 6:
                f.write('#########################################################################\n')
                f.write('\n\n')

            return
####################################################################################################    


####################################################################################################    
    def write_energy_file(self,ff_energy):
        '''Write QM and FF energies to file for a given energy component.

        Parameters
        ----------
        ff_energy : 1darray
            Force field energy

        Returns
        -------
        None

        '''
        with open(self.energy_component_file[self.component],'w') as f:
            f.write('QM Energy\tFF Energy\n')
            for pt in zip(self.qm_energy[self.component],ff_energy):
                template='{:16.6g}'*2+'\n'
                f.write(template.format(pt[0],pt[1]))

        return
####################################################################################################    


####################################################################################################    
#################### Debugging subroutines ##########################################################    
####################################################################################################    
    def perform_tests(self):
        '''Testing subroutine. Changes frequently depending on what I need to
        debug.
        '''

        print 'Dispersion tests:'

        self.component = 4
        #self.calc_dispersion_energy()
        self.fit_component_parameters()

        sys.exit()

        ## # polarizability check
        ## from calculate_molecular_polarizability import Drudes
        ## d1 = Drudes(self.xyz1, self.xyz2, 
        ##             self.multipole_file1, self.multipole_file2,
        ##             self.drude_charges1, self.drude_charges2, 
        ##             self.exponents,
        ##             self.thole_param, self.springcon,
        ##             self.slater_correction)
        ## d1.get_drude_energy()
        ## print d1.find_drude_positions()[0] - self.xyz1
        ## print '---'
        ## dipole = d1.get_molecular_dipole()
        ## print 'Molecular dipole:'
        ## print dipole
        ## sys.exit()

        # multipole check
        ## from multipoles import Multipoles
        ## m = Multipoles(self.xyz1,self.xyz2,self.multipole_file1,self.multipole_file2,self.exponents,self.slater_correction)
        ## print 'Electrostatic energy:', m.get_multipole_electrostatic_energy()[0]
        ## from multipoles import Multipoles
        ## print 'sympy'
        ## m = Multipoles(self.xyz1,self.xyz2,self.multipole_file1,self.multipole_file2,self.exponents,self.slater_correction)
        ## print 'Electrostatic energy:', m.get_multipole_electrostatic_energy()[0]
        ## sys.exit()

        # Drude test
        ## d = Drudes(self.xyz1, self.xyz2, 
        ##             self.charges1, self.charges2,
        ##             self.drude_charges1, self.drude_charges2, 
        ##             self.exponents,
        ##             self.thole_param, self.springcon,
        ##             self.slater_correction)

        ## from drude_oscillators import Drudes as OldDrudes
        ## from multipoles import Multipoles
        ## print 'Old Drude Code'
        ## d = OldDrudes(self.xyz1, self.xyz2, 
        ##             self.charges1, self.charges2,
        ##             self.drude_charges1, self.drude_charges2, 
        ##             self.exponents,
        ##             self.thole_param, self.springcon,
        ##             self.slater_correction)
        ## d.find_drude_positions()
        ## print 'New Drude Code'
        ## d.find_drude_positions()
        ## from multipole_drude_oscillators import Drudes 
        ## from multipoles import Multipoles
        ## d1 = Drudes(self.xyz1, self.xyz2, 
        ##             self.multipole_file1, self.multipole_file2,
        ##             self.drude_charges1, self.drude_charges2, 
        ##             self.exponents,
        ##             self.thole_param, self.springcon,
        ##             self.slater_correction)
        ## d1.get_drude_energy()
        ## d1.find_drude_positions()
        ## sys.exit()
        ## #self.edrude_ind, self.edrude_dhf = d.get_induction_and_dhf_drude_energy()
        ## # print d.m1.multipoles1
        ## # print d.m2.multipoles2
        ## from drude_oscillators import Drudes as OldDrudes
        ## d = OldDrudes(self.xyz1, self.xyz2, 
        ##             self.charges1, self.charges2,
        ##             self.drude_charges1, self.drude_charges2, 
        ##             self.exponents,
        ##             self.thole_param, self.springcon,
        ##             self.slater_correction)
        ## d.find_drude_positions()
        ## #self.edrude_ind, self.edrude_dhf = d.get_induction_and_dhf_drude_energy()
        ## print d1.drude_energy
        ## print d.get_drude_energy()[0:1]

        ## print np.allclose(d1.drude_energy, d.get_drude_energy(),atol=1e-5)
        ## print '----New Code-----'
        ## print d1.shell_xyz1[0]
        ## print d1.shell_xyz2[0]
        ## print '----Old Code-----'
        ## print d.shell_xyz1[0]
        ## print d.shell_xyz2[0]
        ## print '----- XYZ Positions -----'
        ## print d1.xyz1[0] - d.xyz1[0]
        ## print d1.xyz2[0] - d.xyz2[0]
        ## print '----- Shell Positions -----'
        ## print d1.shell_xyz1[0] - d.shell_xyz1[0]
        ## print d1.shell_xyz2[0] - d.shell_xyz2[0]

        ## print np.allclose(d1.shell_xyz1, d.shell_xyz1,atol=1e-5)
        ## print np.allclose(d1.shell_xyz2, d.shell_xyz2,atol=1e-5)
        ## print np.allclose(d1.xyz1, d.xyz1,atol=1e-5)
        ## print np.allclose(d1.xyz2, d.xyz2,atol=1e-5)
        ## sys.exit()

        return
####################################################################################################    


####################################################################################################    
    def read_multipoles_from_orient(self,ifile='orient_multipolar_energy.ornt',start_flag='total'):
        '''Read multipole energy from a prior orient calculation.

        This subroutine is primarily for debugging purposes.

        '''
        with open(ifile,'r') as f:
            data = [line.split() for line in f.readlines()]

        start = [ i for i in xrange(len(data)) if 'total' in data[i] ][0] + 1
        end = [ i for i in xrange(len(data)) if data[i] and data[i][0] == 'Finished'][0] - 1
        self.multipole_energy = np.array([ float(i[-1]) for i in data[start:end]])

        return self.multipole_energy
####################################################################################################    




###########################################################################
###########################################################################
if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    slaterhelp="Use a radial correction (based on the Slater Overlap) in the functional form for exchange."
    scalebihelp="Parametrize bi coefficients by scaling initial guesses for bi coefficients."
    energyhelp="Name of the file containing the energy decomposition for each configuration to fit."
    paramhelp="Name of the parameter file."
    outputhelp="Name of file for storing results of the fitting procedure."
    aijhelp="Combination rule for aij. Valid options are 'saptff', 'waldman-hagler5', and 'geometric_mean'."
    bijhelp="Combination rule for bij. Valid options are 'saptff', 'waldman-hagler5', 'arithmetic_mean', and 'geometric_mean'."

    parser.add_argument("energy_file", type=str, help=energyhelp)
    parser.add_argument("param_file", type=str, help=paramhelp)
    parser.add_argument("output_file", type=str, help=outputhelp)
    parser.add_argument("-s","--slatercorrection", help=slaterhelp,\
            action="store_true", default=False)
    parser.add_argument("-b","--scalebi", help=scalebihelp,\
            action="store_true", default=False)
    parser.add_argument("--aij", help=aijhelp,\
            type=str, default='geometric_mean')
    parser.add_argument("--bij", help=bijhelp,\
            type=str, default='geometric_mean')

    args = parser.parse_args()

    FitFFParameters(args.energy_file,args.param_file,args.output_file,args.slatercorrection,args.scalebi,
            args.aij, args.bij)
###########################################################################
###########################################################################
