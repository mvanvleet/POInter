#!/share/apps/python
__version__ = '1.0.2'

# Standard Packages
import numpy as np
import sys
import os
from scipy.optimize import minimize
from copy import copy
import sympy as sym
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
        String to set Aij combination rule. Default 'geometric'; can also
        be set to 'saptff', or 'waldman-hagler5'.
    bij_combination_rule : str, optional
        String to set Bij combination rule. Default 'geometric_mean'; can also
        be set to 'saptff', 'arithmetic_mean', or 'waldman-hagler5'.
    cij_combination_rule : str, optional
        String to set Cij combination rule. Default 'geometric'.
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

    References
    ----------
    (1) Stone, A. J.; Misquitta, A. J. Atom-Atom potentials from ab initio
    calculations; 2007; Vol. 26.
    (2) McDaniel, J. G.; Schmidt, J. R. J. Phys. Chem. A 2013, 117, 2053-2066.

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
        #   cij: 'geometric'
        self.aij_combination_rule = aij_combination_rule
        self.bij_combination_rule = bij_combination_rule
        self.cij_combination_rule = cij_combination_rule

        # Functional form can be chosen to either be the Born-Mayer or Stone
        # potentials; see Stone's book for more details.
        # Options are 'stone', 'born-mayer', or 'lennard-jones'
        self.functional_form = functional_form
        ###########################################################################
        ###########################################################################


        ###########################################################################
        ################## Program-Defined Class Variables ########################
        # Program-Defined Class Variable defaults, below, can be redefined in
        # the .param file as necessary, but can be left unchanged in most
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
                                      'Residuals',
                                      'Total Energy']

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

        # The overlap model formalism (see ref. 1) would suggest that the
        # exchange-repulsion energy (and perhaps other quantities) should be
        # directly proportional to the electron density overlap, with
        # proportionality constant k. The following two (mutually-exclusive)
        # options control to what extent this proportionality constant should
        # be enforced: fit_universal_k = True will only fit 1 parameter to the
        # force field (D parameters from ISA should be read in in this case),
        # and fit_atomtype_k will fit 1 k parameter for each atomtype
        # in the force field. (In this latter case, provided one exponent is
        # used per atom, self.fit_atomtype_k yields the same optimization as
        # in ref. 2).
        self.fit_universal_k = False
        self.fit_atomtype_k = False

        # ----------------------------------------------------------------------
        # Functional Form Variables; controls options related to the
        # mathematical form of the force field itself
        # ----------------------------------------------------------------------

        # Number of parameters per isotropic atomtype
        self.default_n_isotropic_params = 1

        # If a radial correction is being employed, choose whether or not this
        # correction should correspond to the exact Slater overlap correction
        # or a more approximate form, which is only formally exact for bi=bj.
        self.exact_radial_correction = False

        # Choose damping method for electrostatic interactions. Currently
        # accepted options are 'None' and 'Tang-Toennies'. In addition, it is
        # possible to damp only the point-charge interactions, which ignores
        # the effect of damping the higher-order multipole moments. If the
        # electrostatic damping type is set to 'Tang-Toennies',
        # separate_induction_damping can be set to True, in which case damping
        # exponents for polarization will be read in separately from the
        # exponents used for repulsion.
        self.electrostatic_damping_type = 'None'
        self.induction_damping_type = 'Tang-Toennies'
        self.damp_charges_only = True
        self.separate_induction_damping = True

        # When fitting parameters, choose whether or not to fit
        # dispersion. If fit_isotropic_dispersion is set to True, isotropic
        # dispersion parameters are scaled by a constant. In either case,
        # setting fit_dispersion to True will fit anisotropic parameters.
        self.fit_dispersion = False
        self.fit_isotropic_dispersion = False

        # If set to true, fits a final A parameter to errors in the total
        # energy, in an effort to reduce systematic errors in the total energy
        self.fit_residuals = False

        # Choose whether to allow for anisotropic drude spring constants; if
        # set to True, kx, ky, and kz spring constants must all be read in
        # (rather than listing one universal spring constant)
        self.anisotropic_drudes = True


        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # The following variables should be changed rarely, if ever, and are
        # primarily included for debugging purposes:
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        # Choose whether or not to use common subexpression elimination to
        # speed up parameter optimization
        self.use_cse = True

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

        #self.perform_tests()

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
        #if self.fit_dispersion and self.anisotropic_atomtypes:
        if not self.fit_isotropic_dispersion:
            # Subtract one free parameter per atomtype; a0 is constrained to
            # be the input (isotropic) cn coefficient
            self.default_n_isotropic_params -= 1
            ff_energy += self.fit_component_parameters()
            self.default_n_isotropic_params += 1
        else:
            ff_energy += self.fit_component_parameters()
        ## else:
        ##     ff_energy += self.calc_dispersion_energy()

        # If fitting the residual errors, compute this energy and output to
        # file
        if self.fit_residuals:
            if not self.functional_form == 'lennard-jones':
                self.fit_bii = True
            self.component = 5
            self.qm_energy[5] = self.qm_energy[6] - ff_energy
            ff_energy += self.fit_component_parameters()

        # Sum up all energy components and output to the total energy file 
        self.component = 6
        qm_energy = self.qm_energy[self.component]
        weight = functional_forms.weight(qm_energy, self.eff_mu, self.eff_kt)
        self.lsq_error =  np.sum(weight*(ff_energy - qm_energy)**2)
        self.write_energy_file(ff_energy)
        self.rms_error = self.calc_rmse(ff_energy)
        self.weighted_rms_error = self.calc_rmse(ff_energy, cutoff=self.weighted_rmse_cutoff)
        self.weighted_absolute_error = self.calc_mae(ff_energy, cutoff=self.weighted_rmse_cutoff)
        self.write_output_file()

        print
        print '============================================'
        print 'Finished fitting all force field parameters.' 
        print '============================================'
        print

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
        #self.Aparams = [ [] for i in xrange(self.ncomponents) ] # 4 components; exch, elst, ind, dhf
        self.Aparams = [ ] # 4 components; exch, elst, ind, dhf
        self.exponents = {}
        self.Cparams = {}
        self.Dparams = {}
        self.anisotropic_atomtypes = []
        self.anisotropic_symmetries = {}
        self.drude_charges1 = []
        self.drude_charges2 = []
        self.springcon1 = []
        self.springcon2 = []
        self.induction_exponents1 = []
        self.induction_exponents2 = []

        # Initialize list of all hard constraints
        self.fixed_atomtypes = {}

        # Read in force field parameters
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

            # Ensure settings read in aren't mutually exclusive
            atomtype_err = 'fit_universal_k and fit_atomtype_k cannot both be set to True!'
            assert not (self.fit_universal_k and self.fit_atomtype_k), atomtype_err

            if self.functional_form == 'lennard-jones':
                # Make sure we don't try and fit exponents, and that the
                # 'residual energy' (which in this case is the total energy
                # minus hard constraints) is being fit
                #self.fit_bii = False
                self.fit_bii = True
                self.fit_residuals = True
                self.slater_correction = False
                self.harmonic_constraints = False

            # Read A parameters from file:
            #   Order is: Exchange, Electrostatics, Induction, DHF,
            #   (Dispersion), (Residuals)
            error = '''Atomtypes need to be defined in the same order and with
            the same atomtypes for each energy type (exchange, electrostatics,
            induction, dhf. Please fix your input file.'''
            error1 = 'Exchange parameters should be read in first.'
            error2 = 'fit_dispersion has been set to True, but dispersion parameters not found here.'
            error3 = 'fit_residuals has been set to True, but residual parameters not found here.'
            f.readline()
            line = f.readline().split()
            assert line[0] == 'EXCHANGE', error1
            line = f.readline().split()
            # Read exchange parameters to determine number of fixed atom types
            count = 0
            max_count = 0
            while line[0] != 'ELECTROSTATICS':
                atom = line[0].split('(')[0]
                if atom not in self.fixed_atomtypes:
                    count = max_count
                    self.fixed_atomtypes[atom] = count
                    max_count += 1
                else:
                    count = self.fixed_atomtypes[atom]
                #self.fixed_atomtypes[line[0]] = count
                self.Aparams.append([[] for _ in xrange(self.ncomponents)])
                self.Aparams[count][0].append([float(i) for i in line[1:]])
                line = f.readline().split()
                #count += 1
            nfixed = max_count
            # Electrostatic parameters
            assert line[0] == 'ELECTROSTATICS', error
            line = f.readline().split()
            count = 0
            while line[0] != 'INDUCTION':
            #while count < nfixed:
                atom = line[0].split('(')[0]
                assert self.fixed_atomtypes.has_key(atom), error
                count = self.fixed_atomtypes[atom]
                self.Aparams[count][1].append([float(i) for i in line[1:]])
                line = f.readline().split()
            # Induction parameters
            assert line[0] == 'INDUCTION', error
            line = f.readline().split()
            count = 0
            while line[0] != 'DHF':
                atom = line[0].split('(')[0]
                assert self.fixed_atomtypes.has_key(atom), error
                count = self.fixed_atomtypes[atom]
                self.Aparams[count][2].append([float(i) for i in line[1:]])
                line = f.readline().split()
            # DHF parameters
            assert line[0] == 'DHF', error
            line = f.readline().split()
            if self.fit_dispersion:
                flag = 'DISPERSION'
            elif self.fit_residuals:
                flag = 'RESIDUALS'
            else:
                flag = ''
            while line and line[0] != flag:
                atom = line[0].split('(')[0]
                assert self.fixed_atomtypes.has_key(atom), error
                count = self.fixed_atomtypes[atom]
                #assert self.fixed_atomtypes[atom] == count, error
                self.Aparams[count][3].append([float(i) for i in line[1:]])
                line = f.readline().split()
            if self.fit_dispersion:
                # Dispersion parameters
                assert line[0] == 'DISPERSION', error2
                line = f.readline().split()
                if self.fit_residuals:
                    flag = 'RESIDUALS'
                else:
                    flag = ''
                while line and line[0] != flag:
                    atom = line[0].split('(')[0]
                    assert self.fixed_atomtypes.has_key(atom), error
                    count = self.fixed_atomtypes[atom]
                    self.Aparams[count][4].append([float(i) for i in line[1:]])
                    line = f.readline().split()
            else:
                for atom in self.fixed_atomtypes:
                    count = self.fixed_atomtypes[atom]
                    # Unscaled dispersion parameters being used; set scale
                    # factor to 1.0 for each atomtype
                    self.Aparams[count][4].append([1.0])

            if self.fit_residuals:
                assert line[0] == 'RESIDUALS', error3
                line = f.readline().split()
                while line and line[0] != '':
                    atom = line[0].split('(')[0]
                    assert self.fixed_atomtypes.has_key(atom), error
                    count = self.fixed_atomtypes[atom]
                    self.Aparams[count][5].append([float(i) for i in line[1:]])
                    line = f.readline().split()

            # Read in anisotropic atomtypes
            line = f.readline()
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
            line = f.readline().split()
            while len(line) > 0:
                atom = line[0]
                bi = [float(i) for i in line[1:]]
                if self.exponents.has_key(atom) and \
                        np.all(self.exponents[atom] - bi ):
                    error_msg = 'Multiple exponents for atomtype '+atom+\
                    ' have been given! Make sure each atomtype has only one set of B parameters.'
                    sys.exit(error_msg)
                else:
                    self.exponents[atom] = bi
                line = f.readline().split()

            # Read dispersion coefficients (Cn parameters) from file:
            f.readline()
            line = f.readline().split()

            # Read in Cii parameters and convert to Ci parameters. For now,
            # only a geometric mean combination rule seems appropriate for
            # dispersion parameters, so we take a sqrt here.
            assert self.cij_combination_rule == 'geometric'
            while len(line) > 0:
                # Cn parameters are read in as Cii parameters; to extract Ci
                # parameters, we need to take the square root of Cii
                atom = line[0]
                ci = np.sqrt([float(i) for i in line[1:]])
            #for i,atom in enumerate(all_atoms):
                if self.Cparams.has_key(atom) and \
                        np.all(self.Cparams[atom] - ci ):
                    error_msg = 'Multiple Cn parameters for atomtype '+atom+\
                    ' have been given! Make sure each atomtype has only one set of C parameters.'
                    sys.exit(error_msg)
                else:
                    self.Cparams[atom] = ci
                line = f.readline().split()

            if self.fit_universal_k or self.fit_atomtype_k:
                f.readline()
                # Read in D parameters
                line = f.readline().split()
                while len(line) > 0:
                    atom = line[0]
                    di = [float(i) for i in line[1:]]
                    if self.Dparams.has_key(atom) and \
                            np.all(self.Dparams[atom] - di ):
                        error_msg = 'Multiple sets of D parameters for atomtype '+atom+\
                        ' have been given! Make sure each atomtype has only one set of D parameters.'
                        sys.exit(error_msg)
                    self.Dparams[atom] = di
                    line = f.readline().split()
                #f.readline()

            # Ensure exponents and dispersion parameters have been given for
            # each atomtype
            for atom in self.atomtypes:
                if not self.Cparams.has_key(atom):
                    print '!!!!!!!!!!!'
                    print 'Dispersion parameters (Cn) must be given for all atomtypes. ' \
                    +  'Please provide Cn coefficients for the following atomtype: ' , atom
                    print '!!!!!!!!!!!'
                    sys.exit('Exiting.')
                if not self.exponents.has_key(atom):
                    print '!!!!!!!!!!!'
                    print 'Exponents (B parameters) must be given for all atomtypes. ' \
                    +  'Please provide B coefficients for the following atomtype: ' , atom
                    print '!!!!!!!!!!!'
                    sys.exit('Exiting.')

            # Read multipole file names
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
                line = f.readline().split()[1:]
                self.drude_charges1.append(float(line[0]))
                if self.anisotropic_drudes:
                    self.springcon1.append([float(j) for j in line[1:4]])
                else:
                    self.springcon1.append([3*float(line[1])])
                if self.electrostatic_damping_type == 'Tang-Toennies' \
                        and self.separate_induction_damping:
                    self.induction_exponents1.append(float(line[4]))
            self.drude_charges1 = np.array(self.drude_charges1)
            self.springcon1 = np.array(self.springcon1)
            self.induction_exponents1 = np.array(self.induction_exponents1)
            f.readline()
            for i in xrange(self.natoms2):
                line = f.readline().split()[1:]
                self.drude_charges2.append(float(line[0]))
                if self.anisotropic_drudes:
                    self.springcon2.append([float(j) for j in line[1:4]])
                else:
                    self.springcon2.append([3*float(line[1])])
                if self.electrostatic_damping_type == 'Tang-Toennies' \
                        and self.separate_induction_damping:
                    self.induction_exponents2.append(float(line[4]))
            self.drude_charges2 = np.array(self.drude_charges2)
            self.springcon2 = np.array(self.springcon2)
            self.induction_exponents2 = np.array(self.induction_exponents2)

            # Read parameters for the weighting function, namely eff_mu and eff_kt
            # charges):
            f.readline()
            f.readline()
            self.eff_mu = float(f.readline().split()[1])
            self.eff_kt = float(f.readline().split()[1])

        # Create params dictionary of dictionaries
        self.params = {}
        for atom in self.fixed_atomtypes:
            i = self.fixed_atomtypes[atom]
            self.params[atom] = []
            for ib, b in enumerate(self.exponents[atom]):
                atom_dic = {}
                atom_dic['A'] = [ j[ib][0] if j else None for j in self.Aparams[i] ]
                atom_dic['aniso'] = [ j[ib][1:] if j else [] for j in self.Aparams[i] ]
                atom_dic['B'] = b
                atom_dic['C'] = self.Cparams[atom]
                self.params[atom].append(atom_dic)

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

        # Exponents may vary over the course of the simulation; save their
        # original input values in a separate dictionary 
        self.save_exponents = self.exponents

        # Make sure dispersion fitting options are consistant
        error = 'You cannot set fit_dispersion False and fit_isotropic_dispersion True.'
        if self.fit_isotropic_dispersion:
            assert self.fit_dispersion, error

        # Multipole and drude modules treat exponents as an array; save
        # exponents in this format for use in these modules
        self.all_exponents = [ [] for i in xrange(self.natoms1)]
        for i,atom1 in enumerate(self.atoms1):
            for atom2 in self.atoms2:
                bi = self. exponents[atom1]
                bj = self. exponents[atom2]
                bij = [ [] for _ in bi ]
                for ik, bik in enumerate(bi):
                    for bjl in bj:
                        bijkl = self.combine_exponent(bik,bjl)
                        bij[ik].append(bijkl)
                self.all_exponents[i].append(bij)
        self.all_exponents = np.array(self.all_exponents)

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


## ####################################################################################################    
##     def combine_exponents(self):
##         '''Create cross-terms for exponents according to input combination
##         rule.
## 
##         Parameters
##         ----------
##         None, though implictly depends on choice of bij_combination_rule
## 
##         Returns
##         -------
##         self.exponents : 2darray (natoms1 x natoms2)
##             Array of exponents bij
##         
##         '''
##         bi, bj = sym.symbols(" bi bj")
##         self.combine_num_exponent = lambdify((bi,bj),self.combine_exponent(bi,bj,self.bij_combination_rule),modules='numpy')
## 
##         self.exponents = [ [] for i in xrange(self.natoms1)]
##         for i,bi in enumerate(self.exponents1):
##             for bj in self.exponents2:
##                 bij = self.combine_num_exponent(bi,bj)
##                 self.exponents[i].append(bij)
## 
##         self.exponents = np.array(self.exponents)
##         return self.exponents
## ####################################################################################################    


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
        # Update exponents dictionary
        self.exponents = { atom : [self.params[atom][i]['B'] for i in
                            xrange(len(self.exponents[atom]))] for atom in self.atomtypes }

        # Save array of exponents for later use
        self.all_exponents = [ [] for i in xrange(self.natoms1)]
        for i,atom1 in enumerate(self.atoms1):
            for atom2 in self.atoms2:
                pi = self.params[atom1]
                pj = self.params[atom2]
                bij = [ [] for _ in pi ]
                for ik, pik in enumerate(pi):
                    for jl, pjl in enumerate(pj):
                        bik = pi[ik]['B']
                        bjl = pj[jl]['B']
                        bijkl = self.combine_exponent(bik,bjl)
                        bij[ik].append(bijkl)
                self.all_exponents[i].append(bij)
        self.all_exponents = np.array(self.all_exponents)

        self.fit_bii = False
        self.n_isotropic_params -= 1

        return
####################################################################################################    


####################################################################################################    
    def combine_exponent(self,bi,bj,combination_rule='geometric_mean',mode='np'):
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
        mode: str, optional.
            Evaluate using numpy routines ('np', default) or sympy routines
            ('sp').

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
            if mode == 'sp':
                bij = sym.sqrt(bi*bj)
            else:
                bij = np.sqrt(bi*bj)
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
            aij = ai*aj
        elif combination_rule == 'waldman-hagler5':
            aij = ai*aj*(bij**6/(bi**3*bj**3))
        elif 'arithmetic_mean' in combination_rule:
            aij = (ai + aj)/2
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
        bij = self.combine_exponent(bi,bj,self.bij_combination_rule,mode='sp')

        if self.exact_radial_correction and self.fit_bii:
            # If we're scaling bii, we need to evaluate the radial correction as a
            # piecewise function, in case bi and bj alternate between being
            # in and outside of tolerance
            test = (bi - bj > tol)
            return sym.Piecewise(\
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

        # Initialize local-global axes transformation arrays
        self.axes1 = np.zeros((self.ndatpts, self.natoms1,3,3))
        self.axes1[:,:] = np.eye(3)
        self.axes2 = np.zeros((self.ndatpts, self.natoms2,3,3))
        self.axes2[:,:] = np.eye(3)

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
                elif len(self.anisotropic_axes1[i][1]) == 3:
                    iatom1 = self.anisotropic_axes1[i][1][0]
                    iatom2 = self.anisotropic_axes1[i][1][1]
                    iatom3 = self.anisotropic_axes1[i][1][2]
                    vec = self.get_bisecting_vector(iatom1,iatom2,iatom3)
                elif len(self.anisotropic_axes1[i][1]) == 0:
                    vec = np.array([1,0,0])
                    if np.array_equal(z_axis, vec) or np.array_equal(z_axis, -vec):
                        vec = np.array([0,1,0])
                    print 'Since no x-axis was specified for atom ' \
                            + str(i) + ' in monomer 1,'
                    print 'assuming that atomtype no x/y angular dependence.'
                else:
                    print 'You must specify exactly zero, two, or three atomic indices for each atom.' 
                    print 'The program does not know how to handle more or less atomic indices than what you have prescribed.'
                    sys.exit()
                x_axis = self.project_onto_plane(vec,z_axis)

                x_axis /= np.sqrt((x_axis ** 2).sum(-1))[..., np.newaxis] #Normalize
                y_axis = np.cross(z_axis,x_axis)
                self.axes1[:,i,0] = x_axis
                self.axes1[:,i,1] = y_axis
                self.axes1[:,i,2] = z_axis

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
                elif len(self.anisotropic_axes2[i][0]) > 2:
                    iatom1 = self.anisotropic_axes2[i][0][0]
                    z1 = self.xyz2[:,iatom1,:]
                    z2 = np.mean([self.xyz2[:,j,:] for j in self.anisotropic_axes2[i][0][1:]],axis=0)
                    z_axis = z2 - z1
                else:
                    print 'You must specify exactly two or three atomic indices for atom ' + str(i) + ' in monomer 2.' 
                    print 'The program does not know how to handle more or less atomic indices than what you have prescribed.'
                    sys.exit()
                z_axis /= np.sqrt((z_axis ** 2).sum(-1))[..., np.newaxis] #Normalize

                if len(self.anisotropic_axes2[i][1]) == 2:
                    iatom1 = self.anisotropic_axes2[i][1][0]
                    iatom2 = self.anisotropic_axes2[i][1][1]
                    vec = self.xyz2[:,iatom2,:] - self.xyz2[:,iatom1,:]
                elif len(self.anisotropic_axes2[i][1]) == 3:
                    iatom1 = self.anisotropic_axes2[i][1][0]
                    iatom2 = self.anisotropic_axes2[i][1][1]
                    iatom3 = self.anisotropic_axes2[i][1][2]
                    vec = self.get_bisecting_vector(iatom1,iatom2,iatom3,mon1=False)
                elif len(self.anisotropic_axes2[i][1]) == 0:
                    vec = np.array([1,0,0])
                    if np.array_equal(z_axis, vec) or np.array_equal(z_axis, -vec):
                        vec = np.array([0,1,0])
                    print 'Since no x-axis was specified for atom ' \
                            + str(i) + ' in monomer 2,'
                    print 'assuming that atomtype no x/y angular dependence.'
                else:
                    print 'You must specify exactly zero, two or three atomic indices for atom ' + str(i) + ' in monomer 2.' 
                    print 'The program does not know how to handle more or less atomic indices than what you have prescribed.'
                    sys.exit()
                x_axis = self.project_onto_plane(vec,z_axis)

                for j in xrange(self.natoms1):
                    theta, phi = self.get_angle(i,z_axis,x_axis,j,mon1=False)
                    self.angles2[i][j][0,:] = theta
                    self.angles2[i][j][1,:] = phi
                x_axis /= np.sqrt((x_axis ** 2).sum(-1))[..., np.newaxis] #Normalize

                y_axis = np.cross(z_axis,x_axis)
                self.axes2[:,i,0] = x_axis
                self.axes2[:,i,1] = y_axis
                self.axes2[:,i,2] = z_axis

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
        if self.induction_damping_type == 'Tang-Toennies' \
                and self.separate_induction_damping:
            exponents = [ [ self.combine_exponent(bi,bj) 
                            for bj in self.induction_exponents2 ] 
                            for bi in self.induction_exponents1 ]
            exponents = np.array(exponents)[:,:,np.newaxis,np.newaxis]
            ## print self.all_exponents.shape
            ## print exponents.shape
            ## sys.exit()
        else:
            exponents = self.all_exponents
        if self.drude_method == 'multipole-gradient':
            print 'Calculating drude oscillator energy using a multipole-gradient method'
            #from drude_oscillators import Drudes
            from reverse_drude_oscillators import Drudes
            d = Drudes(self.xyz1, self.xyz2, 
                        self.multipole_file1, self.multipole_file2,
                        self.axes1,self.axes2,
                        self.drude_charges1, self.drude_charges2, 
                        self.springcon1, self.springcon2,
                        #self.all_exponents,
                        exponents,
                        self.thole_param, 
                        self.slater_correction,
                        #self.electrostatic_damping_type,
                        self.induction_damping_type,
                        self.damp_charges_only)
            self.edrude_ind, self.edrude_dhf = d.get_induction_and_dhf_drude_energy()
        elif self.drude_method == 'finite-differences':
            print 'Calculating drude oscillator energy using finite-differences'
            from debug.finite_differences_drude_oscillators import FDDrudes as Drudes
            d = Drudes(self.xyz1, self.xyz2, 
                        self.multipole_file1, self.multipole_file2,
                        self.drude_charges1, self.drude_charges2, 
                        self.all_exponents,
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
                        self.all_exponents,
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
            with open(self.drude_file,'w') as f:
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
        self.n_isotropic_params = self.default_n_isotropic_params

        # Add additional parameters for scaling exponents, if necessary
        if self.fit_bii:
            # Add one additional parameter per atomtype to account for scaling
            # exponents
            self.n_isotropic_params += 1

        # Determine total number of parameters to be fit
        if self.fit_universal_k:
            self.n_isotropic_params -= 1 # Not fitting individual A parameters
            n_general_params = 1
        else:
            n_general_params = 0

        # Set soft-constraints on a and b parameters
        #if self.constrain_ab_positive:
        if self.functional_form == 'born-mayer':
            abound = (0,1e3)
            bbound = (1e-2,1e2)
            if self.component == 4:
                abound = (0.7,1.3)
                bbound = (0.7,1.3)
                #bbound = (1e-2,1e2)

        elif self.functional_form == 'stone':
            abound = (-1e1,1e1)
            bbound = (1e-2,1e2)
        elif self.functional_form == 'lennard-jones':
            abound = (1e0,1e2)
            bbound = (1e-2,1e3)
        else:
            raise NotImplementedError
        unbound = (-1e1,1e1)
        # For isotropic atomtypes, constrain all parameters to be positive
        # For anisotropic atomtypes, only constrain first (and possibly
        # last) parameters (corresponding to A and B, respectively) to be
        # positive
        n_aiso = self.n_isotropic_params if not self.fit_bii else self.n_isotropic_params - 1
        if self.component == 4:
            n_aaniso = n_aiso if self.fit_isotropic_dispersion else n_aiso + 1
        else:
            n_aaniso = n_aiso 

        if not self.fit_atomtype_k:
            # Here n is the number of parameter sets to fit per atomtype,
            # and m is the number of exponents to fit per parameter set
            nsets_iso = [ len(self.exponents[a]) for a in self.fit_isotropic_atomtypes ]
            nsets_aniso = [ len(self.exponents[a]) for a in self.anisotropic_atomtypes ]
            msets_iso = [ n_aiso for a in self.fit_isotropic_atomtypes ]
            msets_aniso = [ n_aaniso for a in self.anisotropic_atomtypes ]
        else:
            nsets_iso = [ n_aiso for a in self.fit_isotropic_atomtypes ]
            nsets_aniso = [ n_aaniso for a in self.anisotropic_atomtypes ]
            msets_iso = [ len(self.exponents[a]) for a in self.fit_isotropic_atomtypes ]
            msets_aniso = [ len(self.exponents[a]) for a in self.anisotropic_atomtypes ]
        if self.fit_bii:
            bounds_iso =[ n*([abound for i in range(self.n_isotropic_params-1)] + 
                            m*[bbound])
                            for n,m,j in zip(nsets_iso,msets_iso,self.fit_isotropic_atomtypes) ]

            bounds_aniso =[ n*([abound for i in range(self.n_isotropic_params-1)] + 
                            [unbound for i in v] +
                            m*[bbound] )
                            for n,m,k,v in zip(nsets_aniso,
                                            msets_aniso,
                                            self.anisotropic_symmetries.keys(), 
                                            self.anisotropic_symmetries.values()) \
                            if k in self.fit_anisotropic_atomtypes ]
            # Store locations of b parameters for later use in harmonic
            # constraint error
            pos_bparams = [ n*([0 for i in range(self.n_isotropic_params-1)] + 
                            m*[1])
                            for n,m,j in zip(nsets_iso,msets_iso,self.fit_isotropic_atomtypes) ]
            pos_aniso_bparams =[ n*([0 for i in range(self.n_isotropic_params-1)] + 
                            [0 for i in v] +
                            m*[1] )
                            for n,m,k,v in zip(nsets_aniso,
                                            msets_aniso,
                                            self.anisotropic_symmetries.keys(), 
                                            self.anisotropic_symmetries.values()) \
                            if k in self.fit_anisotropic_atomtypes ]
            if pos_bparams + pos_aniso_bparams == []:
                self.i_bparams = []
            else:
                self.i_bparams = [i for i,b in 
                                    enumerate(flatten(pos_bparams + pos_aniso_bparams)) 
                                    if b == 1]


        elif self.fit_universal_k:
            bounds_iso = [ n*([abound for i in range(self.n_isotropic_params)]) 
                            for n,i in zip(nsets_iso,self.fit_isotropic_atomtypes) ]
            bounds_aniso =[ n*([abound for i in range(self.n_isotropic_params)]) + 
                            [unbound for i in v] 
                            for n,k,v in zip(nsets_aniso,
                                            self.anisotropic_symmetries.keys(), 
                                            self.anisotropic_symmetries.values()) \
                            if k in self.fit_anisotropic_atomtypes ]
        else:
            bounds_iso = [ n*([abound for i in range(self.n_isotropic_params)]) 
                            for n,i in zip(nsets_iso,self.fit_isotropic_atomtypes) ]
            bounds_aniso =[ n*([abound for i in range(self.n_isotropic_params)] + 
                            [unbound for i in v] )
                            for n,k,v in zip(nsets_aniso,
                                            self.anisotropic_symmetries.keys(), 
                                            self.anisotropic_symmetries.values()) \
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

        ntot_params = len(bnds)

        if self.fit_universal_k and self.component != 4:
            # Add parameter for K proportionality constant
            ntot_params += 1

        # Perform initial energy call to set up function and derivative
        # subroutines
        p0=np.array([1.0 for i in xrange(ntot_params)])
        if self.functional_form == 'lennard-jones' and self.component == 5:
            # I've found that the convergence properties for LJ are very
            # sensitive to the intial parameter guesses; hence the choice of
            # p0 for this force field is much more exact than for the BM
            # functional form. The choice of sigma relies on the VdW radii for
            # an element, and the well depth is (somewhat arbitrarily) set to 0.5 mH.
            from chemistry import elementdata
            from chemistry.constants import ang2bohr
            elements = [atom[0]+filter(str.islower,atom[1:]) for atom in self.fit_atomtypes]
            sigma0 = np.array([2*elementdata.VdWRadius(e) for e in elements])
            sigma0 *= ang2bohr
            eps = 0.500
            p0[::2] = sigma0
            p0[1::2] *= eps
        self.final_energy_call = False
        self.generate_num_eij(p0)

        # If using a Lennard Jones force field, the only term we should be
        # fitting is the 'residual' (non-drude oscillator or
        # electrostatic) term; for all other energy components we should avoid
        # this optimization step
        if self.functional_form == 'lennard-jones' and self.component != 5:
            qm_fit_energy = self.subtract_hard_constraint_energy()
            return self.qm_energy[self.component] - qm_fit_energy

        print '-------------'
        print 'Optimizing parameters for ' + self.energy_component_names[self.component]
        print '({} parameters in total)'.format(ntot_params)
        print '-------------'

        # To speed up calculations, subtract off energy that is already known
        # on the basis of hard constraints.
        self.ntot_params = ntot_params
        qm_fit_energy = self.subtract_hard_constraint_energy()
        self.qm_fit_energy = np.array(qm_fit_energy)


        # Use scipy.optimize to perform a least-squares fitting:
        # Initial paramaters are given by p0, and the weighted least squares
        # fitting procedure is given in a subroutine below. Weights here are
        # given by a Fermi-Dirac distribution.
        if len(p0): # Only perform fitting if there are unconstrained parameters
        #if self.component == 4 and len(p0): 

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

        ## if self.component == 4 and not self.fit_dispersion: 
        ##     # No hard constraints to subtract for dispersion
        ##     return qm_fit_energy

        # For electrostatics, subtract off multipole energies
        if self.component == 1:
            if self.read_multipole_energy_from_orient:
                qm_fit_energy -= self.multipole_energy
                error = 'Damping type needs to be None for consistency with the Orient program.'
                assert self.electrostatic_damping_type == 'None', error
            else:
                m = Multipoles(self.xyz1,self.xyz2,
                               self.multipole_file1,self.multipole_file2,
                               self.all_exponents,self.slater_correction,
                               self.electrostatic_damping_type,self.damp_charges_only)
                multipole_energy = m.get_multipole_electrostatic_energy()
                qm_fit_energy -= multipole_energy
                with open('multipoles.dat','w') as f:
                    f.write('# Eelst \t\t Emultipole \n')
                    for q, m in zip(self.qm_energy[self.component],multipole_energy):
                        f.write('{:16.8f} {:16.8f}\n'.format(q,m))

        # For induction and DHF, subtract off drude oscillator energy
        elif self.component == 2:
            print 'Subtracting off 2nd order drude oscillator energy'
            qm_fit_energy -= self.edrude_ind
            
        elif self.component == 3:
            print 'Subtracting off higher order drude oscillator energies'
            qm_fit_energy -= self.edrude_dhf

        else:
            pass

        if self.functional_form == 'lennard-jones' and self.component != 5:
            return qm_fit_energy

        # Subtract off constrained short-range energies
        for i, atom1 in enumerate(self.atoms1):
            for j, atom2 in enumerate(self.atoms2):
                if (atom1 in self.fit_atomtypes or atom2 in self.fit_atomtypes):
                    # If atom pair has free parameters, skip this step
                    continue
                rij = self.r12[i][j]
                theta1ij = self.angles1[i,j,0]
                phi1ij = self.angles1[i,j,1]
                theta2ji = self.angles2[j,i,0]
                phi2ji = self.angles2[j,i,1]
                pair = (atom1,atom2)
                # Only stored interactions for each interaction once, so
                # need to check what order cross terms were stored in
                if self.get_num_eij.has_key(pair):
                    args = [0 for _ in xrange(self.ntot_params)] + [rij] + [theta1ij] + [theta2ji] +  [phi1ij] + [phi2ji]
                else:
                    args = [0 for _ in xrange(self.ntot_params)] + [rij] + [theta2ji] + [theta1ij] +  [phi2ji] + [phi1ij]
                    pair = (atom2, atom1)

                if not self.use_cse:
                    energy = [ f(*args) for f in self.get_num_eij[pair]]
                else:
                    num_eij, subexp = self.get_num_eij[pair]
                    energy = self.evaluate_num_f(args,subexp,num_eij)

                qm_fit_energy -= energy[0]


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
        #self.output_params(params)
        mapped_params = self.map_params(params)
        # TODO: I probably should call map_params here (or replace map_params
        # entirely with output_params

        # Calculate error penalty for b params, assuming harmonic constraints.
        # Trick here is to make sure derivatives for each constraint get
        # mapped onto the correct parameter
        harmonic_error = 0.0
        dharmonic_error = []
        count = 0
        shift = self.n_isotropic_params
        dharmonic_error = [ 0 for _ in params]
        for i,atom in enumerate(self.fit_isotropic_atomtypes + self.fit_anisotropic_atomtypes):
        #for atom in self.fit_isotropic_atomtypes + self.fit_anisotropic_atomtypes:
            for ib in xrange(len(self.exponents[atom])):
                b = mapped_params[i][ib]['B']
                b0 = self.save_exponents[atom][ib]
                harmonic_error += k*(b - b0)**2
                p = self.i_bparams[count]
                dharmonic_error[p] = 2*k*(b - b0)*b0
                count +=1

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

        ## if self.component == 4:
        ##     for i,ff in enumerate(ff_energy):
        ##         print i, ff
        ##     print 'ff'
        ##     print np.min(ff_energy), np.max(ff_energy)
        ##     print 'qm'
        ##     print np.min(qm_energy), np.max(qm_energy)
        ##     print 'weight'
        ##     print np.min(weight), np.max(weight)
        ##     print 'lsq'
        ##     print np.min(lsq_error), np.max(lsq_error)

        ##     sys.exit()
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
    def generate_num_eij(self,params):
        '''Generate numerical functions to calculate the pairwise interaction
        energy for a given component (exchange, induction,
        etc.) given a set of parameters. Also return the gradient of the pair
        energy with respect to each fit parameter.

        In detail, the SymPy package is used to symbollically evaluate the
        force field energy and automatically compute the gradient. Once these
        symbolic quantities have been calculated, the lamdify function is
        called to generate (fast) numerical subroutines for calculation of the
        energy and gradient as a function of parameter values. Numerical
        functions are stored in a dictionary for later calls to
        calc_ff_energy.

        Parameters
        ----------
        params : 1d tuple 
            Tuple containing all fit parameters 

        Returns
        -------
        get_num_eij : dictionary of lambda functions
            Dictionary of numerical functions for the interaction energy (and
            derivatives with respect to each free parameter) between atomtypes i
            and j.

        '''

        if self.functional_form == 'lennard-jones' and self.component != 5:
            return

        params = sym.symbols('p0:%d'%len(params))
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

        # Now that we know which set of parameters to use for each atomtype,
        # we can finally compute the component energy of the system for each
        # data point:
        # Declare some symbols
        rij, theta1ij, theta2ji, phi1ij, phi2ji = \
                sym.symbols('rij theta1ij theta2ji phi1ij phi2ji')

        print 'Generating subroutines for pairwise interactions.'
        self.get_num_eij = {}
        for i,atom1 in enumerate(self.atomtypes):
            if self.component == 4:
                # For dispersion, get Ai parameters from exchange to calculate
                # damping
                Ai_atom1 = self.params[atom1]
            elif atom1 in self.fixed_atomtypes:
                Ai_atom1 = self.params[atom1]
            else:
                Ai_atom1 = params[param_map[atom1]]
            for atom2 in self.atomtypes[i:]:
                # Check that atom pair energy will actually be calculated;
                # avoid computation for atom pairs only appearing in one
                # monomer
                if atom1 in self.atoms1:
                    if atom2 not in self.atoms2:
                        continue
                elif atom1 in self.atoms2:
                    if atom2 not in self.atoms1:
                        continue

                if self.component == 4:
                    Aj_atom2 = self.params[atom2]
                elif atom2 in self.fixed_atomtypes:
                    Aj_atom2 = self.params[atom2]
                else:
                    Aj_atom2 = params[param_map[atom2]]

                eij = 0.0
                d_eij = [0.0 for _ in param_symbols]
                pair = (atom1, atom2)
                for iAi, Ai in enumerate(Ai_atom1):
                    for iAj, Aj in enumerate(Aj_atom2):
                        eij += self.calc_sym_eij(atom1, atom2 ,rij, Ai, Aj, \
                                            theta1ij, theta2ji, phi1ij, phi2ji)

                # For dispersion, get Ci coefficients
                if self.component == 4:
                    if atom1 in self.fixed_atomtypes:
                        # Should only be one set of Cparams per atomtype
                        Ci = self.params[atom1][0]
                    else:
                        Ci = params[param_map[atom1]]
                    if atom2 in self.fixed_atomtypes:
                        Cj = self.params[atom2][0]
                    else:
                        Cj = params[param_map[atom2]]

                    eij = self.calc_sym_disp_ij(atom1 , atom2 ,rij, eij, Ci, Cj,
                                        theta1ij, theta2ji, phi1ij, phi2ji)

                d_eij = [sym.diff(eij,p) for p in param_symbols]

                args = (param_symbols, rij, theta1ij, theta2ji, phi1ij, phi2ji)
                if not self.use_cse:
                    num_eij = lambdify( flatten(args),  eij, modules='numpy')
                    num_d_eij = [ lambdify(flatten(args), i,
                                          modules='numpy') for i in
                                          d_eij ]
                    self.get_num_eij[pair] = [num_eij] + num_d_eij

                else:
                    num_eij, subexp = self.generate_num_f(flatten(args), eij, d_eij)

                    self.get_num_eij[pair] = [num_eij, subexp]

        return self.get_num_eij
####################################################################################################    


####################################################################################################    
    def calc_ff_energy(self,params):
        '''Compute the force field energy for a given component (exchange,
        induction, etc.) given a set of parameters. Also return the gradient
        of the FF energy with respect to each fit parameter.

        Parameters
        ----------
        params : 1d tuple 
            Tuple containing all fit parameters 

        Returns
        -------
        ff_energy : 1darray (ndatpts)
            Force field energy for each dimer configuration.
        dff_energy : list of 1darrays (nparams x ndatpts)
            Derivative of the force field energy with respect to each
            parameter.

        '''
        # Map A parameters (both free and constrained) onto each atomtype
        save_params = params
        params = self.map_params(params)
        param_map = {}
        paramindex = 0
        for atomtype in self.fit_isotropic_atomtypes + self.fit_anisotropic_atomtypes:
            if self.ignorecase:
                param_map[atomtype.upper()] = paramindex
                paramindex += 1
            else:
                param_map[atomtype] = paramindex
                paramindex += 1

        # Calculate force field energy and derivatives for each atom pair
        ff_energy = np.zeros_like(self.qm_energy[self.component])
        dff_energy = np.array([ np.zeros_like(ff_energy) for _ in save_params ])
        for i, atom1 in enumerate(self.atoms1):
            for j, atom2 in enumerate(self.atoms2):
                if not (atom1 in self.fit_atomtypes or atom2 in self.fit_atomtypes):
                    # Constrained energies already subtracted
                    continue
                rij = self.r12[i][j]
                theta1ij = self.angles1[i,j,0]
                phi1ij = self.angles1[i,j,1]
                theta2ji = self.angles2[j,i,0]
                phi2ji = self.angles2[j,i,1]
                pair = (atom1,atom2)
                # Only stored interactions for each interaction once, so
                # need to check what order cross terms were stored in
                if self.get_num_eij.has_key(pair):
                    #args = Ai + Aj + [rij] + [theta1ij] + [theta2ji] +  [phi1ij] + [phi2ji]
                    args = list(save_params) + [rij] + [theta1ij] + [theta2ji] +  [phi1ij] + [phi2ji]
                else:
                    #args = Aj + Ai + [rij] + [theta2ji] + [theta1ij] +  [phi2ji] + [phi1ij]
                    args = list(save_params)+ [rij] + [theta2ji] + [theta1ij] +  [phi2ji] + [phi1ij]
                    pair = (atom2,atom1)

                if not self.use_cse:
                    energy = [ f(*args) for f in self.get_num_eij[pair]]
                else:
                    num_eij, subexp = self.get_num_eij[pair]
                    energy = self.evaluate_num_f(args,subexp,num_eij)

                
                # Fix return values of int(0) to be appropriately shaped np
                # arrays
                zeros = np.zeros_like(ff_energy)
                for i1,line in enumerate(energy):
                    if type(line) == int:
                        energy[i1] = zeros

                ff_energy += energy[0]

                dff_energy += np.array(energy[1:]) 

                # Check for erroneous negative energies in exchange (which
                # should only arise with bad parameters for anisotropy)
                if self.final_energy_call and self.component == 0:
                    eij_min = np.amin(energy[0])
                    if eij_min < 0:
                        print 'WARNING: Negative pairwise exchange energies encountered.'
                        print '     Lowest energy exchange energy:', eij_min

        # Print and return parameters
        params = save_params
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
        shift0 = self.n_isotropic_params
        for iatom,atom in enumerate(self.fit_isotropic_atomtypes+self.fit_anisotropic_atomtypes):
            if self.component == 4:
                nparam_sets = 1
            else:
                nparam_sets = len(self.exponents[atom])
            shift = shift0
            if self.fit_bii and self.fit_atomtype_k:
                shift += nparam_sets - 1 
                #shift += len(self.exponents[atom]) - 1 
            if atom in self.fit_anisotropic_atomtypes:
                shift += len(self.anisotropic_symmetries[atom])
            if not (self.fit_atomtype_k or self.fit_universal_k): 
                #shift *= len(self.exponents[atom])
                shift *= nparam_sets
            mapped_params.append([])

            if self.component == 4:
                assert not self.fit_bii,\
                'Exponent fitting not implemented for dispersion'
                param_dic = {}
                param_dic['C'] = self.Cparams[atom]
                if self.fit_isotropic_dispersion:
                    param_dic['A'] = params[count]
                    param_dic['aniso'] = params[count+1:count+shift]
                    count += shift
                else:
                    param_dic['A'] = 1.0
                    param_dic['aniso'] = params[count:count+shift]
                    count += shift

                mapped_params[iatom] = param_dic
                continue


            # Partition parameters into short-range parameters for each
            # atomtype. A/K parameters are listed first, followed by Aniso
            # parameters and lastly B scale factors.
            for ib,b in enumerate(self.exponents[atom]):
                param_dic = {}

                if self.functional_form == 'lennard-jones':
                    iexp = count + shift - len(self.exponents[atom]) + ib
                    param_dic['B'] = params[iexp]
                elif self.fit_bii:
                    iexp = count + shift - len(self.exponents[atom]) + ib
                    param_dic['B'] = params[iexp]*b
                else:
                    param_dic['B'] = b

                if self.fit_atomtype_k:
                    param_dic['K'] = params[count]
                    param_dic['D'] = self.Dparams[atom][ib]
                    param_dic['A'] = functional_forms.get_ai(param_dic['K'],param_dic['B'],param_dic['D'])
                elif self.fit_universal_k:
                    param_dic['K'] = params[-1]
                    param_dic['D'] = self.Dparams[atom][ib]
                    param_dic['A'] = functional_forms.get_ai(param_dic['K'],param_dic['B'],param_dic['D'])
                else:
                    param_dic['A'] = params[count]

                if atom in self.fit_anisotropic_atomtypes:
                    ianiso_start = count + 1 
                    if not self.fit_atomtype_k or self.fit_universal_k:
                        ianiso_start += ib*len(self.anisotropic_symmetries[atom])
                    ianiso_end = ianiso_start + len(self.anisotropic_symmetries[atom])
                    param_dic['aniso'] = params[ianiso_start:ianiso_end]
                else:
                    param_dic['aniso'] = []

                param_dic['C'] = self.Cparams[atom]

                mapped_params[iatom].append(param_dic)


            count += shift

        return mapped_params
####################################################################################################    


####################################################################################################    
    def calc_sym_eij(self, atom1, atom2, rij, params1, params2, theta1, theta2, phi1, phi2):
        '''Symbollically compute the pairwise interaction energy between atom
        i in monomer 1 and atom j in monomer 2.

        Parameters
        ----------
        atom1 : string
            Atomtype for atom i in monomer 1.
        atom2 : int
            Atomtype for atom j in monomer 2.
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
        bi = params1['B']
        bj = params2['B']
        bij = self.combine_exponent(bi,bj,self.bij_combination_rule,mode='sp')

        if self.component == 4:
            # For dispersion calculations, need to evaluate exchange energy in
            # this subroutine
            component = 0
        else:
            component = self.component

        # Calculate the A coefficient for each atom. This
        # coefficient is computed differently if the atom is
        # isotropic or anisotropic. 
        if atom1 in self.anisotropic_atomtypes:
            sph_harm = self.anisotropic_symmetries[atom1]
            if self.component == 4 or atom1 in self.fixed_atomtypes:
                a = params1['A'][component]
                Aangular = params1['aniso'][component]
            else:
                a = params1['A']
                Aangular = params1['aniso']
            ai = functional_forms.get_anisotropic_ai(sph_harm, a,Aangular,rij,theta1,phi1)
        else: #if isotropic
            if self.component == 4 or atom1 in self.fixed_atomtypes:
                ai = params1['A'][component] 
            else:
                ai = params1['A']
        if atom2 in self.anisotropic_atomtypes:
            sph_harm = self.anisotropic_symmetries[atom2]
            if self.component == 4 or atom2 in self.fixed_atomtypes:
                a = params2['A'][component]
                Aangular = params2['aniso'][component]
            else:
                a = params2['A']
                Aangular = params2['aniso']
            aj = functional_forms.get_anisotropic_ai(sph_harm, a,Aangular,rij,theta2,phi2)
        else: #if isotropic
            if self.component == 4 or atom2 in self.fixed_atomtypes:
                aj = params2['A'][component] 
            else:
                aj = params2['A']

        aij = self.combine_prefactor(ai,aj,bi,bj,bij,self.aij_combination_rule)
        if self.functional_form == 'stone':
            # Stone functional form incorporates the pre-factor term in the
            # exponential
            aij = sym.exp(aij*bij)

        # Get radial pre-factor correction
        if self.slater_correction:
            a_rad = self.get_radial_correction(rij,bi,bj)
        else:
            a_rad = 1

        # Calculate the ff energy for the atom pair.
        eij = functional_forms.get_eij(component,rij,bij,aij,
                            self.functional_form,self.slater_correction)
        # Incorporate radial correction factor and pre-factor
        if self.functional_form != 'lennard-jones':
            eij *= a_rad*aij

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
                bi = self.exponents[atom1]
                bj = self.exponents[atom2]
                bij = self.combine_exponent(bi,bj,self.bij_combination_rule)

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

        self.write_output_file()

        return dispersion_energy
####################################################################################################    


####################################################################################################    
    def calc_sym_disp_ij(self, atom1, atom2, rij, eij_exch, params1, params2, theta1, theta2, phi1, phi2):
        '''Symbollically compute the pairwise interaction energy between atom
        i in monomer 1 and atom j in monomer 2.

        Parameters
        ----------
        atom1 : string
            Atomtype for atom i in monomer 1.
        atom2 : int
            Atomtype for atom j in monomer 2.
        rij : symbol
            Interatomic distance between atoms i and j.
        eij : symbolic expression
            Pairwise exchange energy (for use in calculting TT damping factor)
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

        # Calculate the A coefficient for each atom. This
        # coefficient is computed differently if the atom is
        # isotropic or anisotropic. 
        ci = params1['C']
        cj = params2['C']

        if self.fit_dispersion:
            if atom1 in self.anisotropic_atomtypes:
                sph_harm = self.anisotropic_symmetries[atom1]
                if atom1 in self.fixed_atomtypes:
                    a = params1['A'][self.component]
                    Aangular = params1['aniso'][self.component]
                else:
                    a = params1['A']
                    Aangular = params1['aniso']
                ai = functional_forms.get_anisotropic_ai(sph_harm, a,Aangular,rij,theta1,phi1)
            else: #if isotropic
                ai = params1['A'][self.component] if atom1 in self.fixed_atomtypes else params1['A']
            if atom2 in self.anisotropic_atomtypes:
                sph_harm = self.anisotropic_symmetries[atom2]
                if atom2 in self.fixed_atomtypes:
                    a = params2['A'][self.component]
                    Aangular = params2['aniso'][self.component]
                else:
                    a = params2['A']
                    Aangular = params2['aniso']
                aj = functional_forms.get_anisotropic_ai(sph_harm, a,Aangular,rij,theta2,phi2)
            else: #if isotropic
                aj = params2['A'][self.component] if atom2 in self.fixed_atomtypes else params2['A']
        else:
            ai = 1.0
            aj = 1.0

        ci = [ai*i for i in ci]
        cj = [aj*j for j in cj]

        # Calculate the A coefficient for each atom. This
        # coefficient is computed differently if the atom is
        # isotropic or anisotropic. 
        eij = 0 

        # Calculate TT damping term
        y = sym.diff(-sym.ln(eij_exch),rij)
        x = y*rij
        for i,n in enumerate(range(6,14,2)):
            cij = self.combine_Cparam(ci[i],cj[i],self.cij_combination_rule)

            # Calculate the ff energy for the atom pair.
            bij = None
            eij += functional_forms.get_dispersion_energy(n,cij,rij,bij,x,
                                self.slater_correction)

        return eij
####################################################################################################    


####################################################################################################    
    def generate_num_f(self,fargs,f,fprime):
        '''Using common sub expressions (CSE), take symbolic functions and
        derivatives and return a fast numerical energy evaluation routine.

        Parameters
        ----------
        fargs : list of sympy symbols
            Arguments for f and fprime
        f : sympy expression
            Symbolic function
        fprime : list of sympy expressions
            Derivatives of f (need not specify which partial derivatives are
            included; all derivatives with respect to fargs could be given as
            input, or just a subset).

        Returns
        -------
        fexp : list of lambda functions
            The first element of fexp is a numerical subroutine for f; the
            remaining elements are (in input order) expressions for each derivative contained
            in fprime.
        subexp : list of function substitutions
            This list of substitution expressions is required for future
            evaluations of fexp

        '''
        # CSE returns a 2-tuple. The first element is a list of 2-tuples containing
        # (xi, value) pairs (sub). The second element is a list of the reduced input
        # expression(s) with values replaced by xi (red)
        sub, red = sym.cse([f] + fprime)

        subsym = [] # List of symbols evaluated so far
        subexp = [] # List of functions to evaluate xi

        for xi, val in sub:
            args = fargs + subsym
            subexp.append(sym.lambdify(flatten(args), val, modules="numpy"))
            subsym.append(xi)

        args = fargs + subsym
        fexp = [sym.lambdify(flatten(args), ired, modules="numpy") for ired in red]

        return fexp, subexp
####################################################################################################    


####################################################################################################    
    def evaluate_num_f(self,fargs, subexp, fexp):
        '''Evaluate the numerical lambda function(s) fexp given a list of
        arguments (fargs) to fexp and a list of common sub expressions (subexp) used
        in generating fexp.

        Parameters
        ----------
        fargs : list of numpy floats and/or arrays
            Arguments for fexp; should match input arguments used to generate
            fexp (see self.generate_num_f)
        subexp : list of symbols
            Subexpressions generated by self.generate_num_f
        fexp : list of lambda functions
            Numerical expressions for f and its derivatives, generated by
            self.generate_num_f

        Returns
        -------
        fout : list of numpy floats and/or arrays
            Numerical output of fexp(fargs)

        '''

        fout = [ [] for i in fexp ]
        for expr in subexp:
            fargs.append(expr(*fargs))

        for i, expr in enumerate(fexp):
            fout[i] = expr(*fargs)

        return fout
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

        # Collect params differently if dealing with Lennard-Jones functional
        # form
        if self.functional_form == 'lennard-jones' and self.component == 5:
            for i,atom in enumerate(self.fit_isotropic_atomtypes+self.fit_anisotropic_atomtypes):
                self.params[atom] = []
                atom_dic = {}
                atom_dic['A'] = [ params[i][0]['A'] ]
                atom_dic['aniso'] = [ params[i][0]['aniso'] ]
                atom_dic['B'] = params[i][0]['B'] 
                self.params[atom].append(atom_dic)

            return self.params

        # Collect parameters from the fitted atomtypes
        for i,atom in enumerate(self.fit_isotropic_atomtypes+self.fit_anisotropic_atomtypes):
            if self.component == 0:
                self.params[atom] = []
                for ib, b in enumerate(self.exponents[atom]):
                    atom_dic = {}
                    atom_dic['A'] = [ params[i][ib]['A'] ]
                    atom_dic['aniso'] = [ params[i][ib]['aniso'] ]
                    atom_dic['B'] = params[i][ib]['B'] 
                    atom_dic['C'] = self.Cparams[atom]
                    self.params[atom].append(atom_dic)
            elif self.component == 4:
                for ib, b in enumerate(self.exponents[atom]):
                    self.params[atom][ib]['A'].append(params[i]['A'])
                    self.params[atom][ib]['aniso'].append(params[i]['aniso'])
            else:
                for ib, b in enumerate(self.exponents[atom]):
                    self.params[atom][ib]['A'].append(params[i][ib]['A'])
                    self.params[atom][ib]['aniso'].append(params[i][ib]['aniso'])
                if self.fit_bii:
                    self.params[atom][ib]['B'] = params[i][ib]['B']

        return self.params
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
                for atom in self.atomtypes:
                    for ib in xrange(len(self.exponents[atom])):
                        name = atom + '(' + str(ib) + ')'
                        template = '{:10s} {:8.6f}\n'
                        b = self.params[atom][ib]['B']
                        f.write(template.format(name,b))
                f.write(short_break)
                f.write('Monomer 1 Multipole File:\n')
                f.write(self.multipole_file1 + '\n')
                f.write('Monomer 2 Multipole File:\n')
                f.write(self.multipole_file2 + '\n')
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
                f.write('Dispersion Parameters:\n')
                f.write('    Functional Form = \n')
                f.write('\tE(disp)_ij = sum_(n=6,8,10,12){A*fdamp_n*(Cij_n/r_ij^n)}\n')

            # Residual Parameters
            elif self.component == 5:
                f.write('Residual Error Parameters:\n')
                f.write('    Functional Form = \n')
                if self.functional_form == 'lennard-jones':
                    f.write('\tE(LJ)_ij = A/r^12 - B/r^6\n')
                elif self.slater_correction:
                    f.write('\tE(residual)_ij = - A*K2(rij)*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')
                else:
                    f.write('\tE(residual)_ij = - A*(1 + a_yml*Y_ml)*exp(-bij*rij)\n')

            elif self.component == 6:
                f.write('Total Energy:\n')
            else:
                print 'Writing output file not yet implemented for component ',self.component
                sys.exit()


            # Write dispersion Cn coefficients
            if self.component == 4:
                # Print input Cn coefficients
                template='{:5s}   '+'{:^16s}'*4 +'\n'
                f.write(template.format('','C6','C8','C10','C12'))
                for k,v in self.Cparams.items():
                    template='{:5s}'+'{:16.6f}'*len(v)+'\n'
                    f.write(template.format(k,*v))

            # Write fitting parameters to file
            if self.functional_form == 'lennard-jones' and self.component == 5:
                f.write('Fitted Atomtypes \n')
                for i,atom in enumerate(self.fit_isotropic_atomtypes):
                    template='{:5s}   '+'{:^16s}'*2+'\n'
                    f.write(template.format('','A','B'))
                    name = atom + '(' + str(0) + ')'
                    a = self.params[atom][0]['A'][0]
                    b = self.params[atom][0]['B']
                    template='{:5s}'+'{:16.6f}'*2 + '\n'
                    f.write(template.format(name,a,b))
                f.write('Constrained Atomtypes \n')
                for atom in self.fixed_atomtypes:
                    template='{:5s}   '+'{:^16s}'*2+'\n'
                    f.write(template.format('','A','B'))
                    name = atom + '(' + str(0) + ')'
                    a = self.params[atom][0]['A'][self.component]
                    b = self.params[atom][0]['B']
                    template='{:5s}'+'{:16.6f}'*2 + '\n'
                    f.write(template.format(name,a,b))
                if not self.fixed_atomtypes:
                    f.write('  None\n')
            else:
                fit_list = range(0,4)
                if self.fit_dispersion:
                    fit_list += [4]
                if self.fit_residuals:
                    fit_list += [5]
                if self.component in fit_list: # Fits not applicable for dispersion, total_energy
                    f.write('Fitted Atomtypes \n')
                    for i,atom in enumerate(self.fit_isotropic_atomtypes):
                        template='{:5s}   '+'{:^16s}\n'
                        f.write(template.format('','A'))
                        for ib in xrange(len(self.exponents[atom])):
                            name = atom + '(' + str(ib) + ')'
                            a = self.params[atom][ib]['A'][self.component]
                            template='{:5s}'+'{:16.6f}\n'
                            f.write(template.format(name,a))
                    for i,atom in enumerate(self.fit_anisotropic_atomtypes):
                        template='{:5s}   '+'{:^16s}'*(len(self.anisotropic_symmetries[atom])+1)+'\n'
                        args = ['a_' + y for y in self.anisotropic_symmetries[atom] ]
                        f.write(template.format('','A',*args))
                        for ib in xrange(len(self.exponents[atom])):
                            name = atom + '(' + str(ib) + ')'
                            a = self.params[atom][ib]['A'][self.component]
                            aniso = self.params[atom][ib]['aniso'][self.component]
                            template='{:5s}'+'{:16.6f}'*(len(args) + 1) + '\n'
                            f.write(template.format(name,a,*aniso))
                    if not self.fit_atomtypes:
                        f.write('  None\n')

                    f.write('Constrained Atomtypes \n')
                    for atom in self.fixed_atomtypes:
                        if atom in self.anisotropic_atomtypes:
                            template='{:5s}   '+'{:^16s}'*(len(self.anisotropic_symmetries[atom])+1)+'\n'
                            args = ['a_' + y for y in self.anisotropic_symmetries[atom] ]
                            f.write(template.format('','A',*args))
                            for ib in xrange(len(self.exponents[atom])):
                                name = atom + '(' + str(ib) + ')'
                                a = self.params[atom][ib]['A'][self.component]
                                aniso = self.params[atom][ib]['aniso'][self.component]
                                template='{:5s}'+'{:16.6f}'*(len(args) + 1) + '\n'
                                f.write(template.format(name,a,*aniso))
                        else:
                            template='{:5s}   '+'{:^16s}\n'
                            f.write(template.format('','A'))
                            for ib in xrange(len(self.exponents[atom])):
                                name = atom + '(' + str(ib) + ')'
                                a = self.params[atom][ib]['A'][self.component]
                                template='{:5s}'+'{:16.6f}\n'
                                f.write(template.format(name,a))
                    if not self.fixed_atomtypes:
                        f.write('  None\n')

            if not success:
                    f.write('Warning! Optimizer did not terminate successfully, but rather quit with the following error message:\n')
                    f.write(message + '\n')
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

        #return
        ## m = Multipoles(self.xyz1,self.xyz2,
        ##                self.multipole_file1,self.multipole_file2,
        ##                self.all_exponents,self.slater_correction,
        ##                self.electrostatic_damping_type,self.damp_charges_only)
        ## m.get_multipole_electrostatic_energy()


        #self.get_drude_oscillator_energy()
        if self.induction_damping_type == 'Tang-Toennies' \
                and self.separate_induction_damping:
            exponents = [ [ self.combine_exponent(bi,bj) 
                            for bj in self.induction_exponents2 ] 
                            for bi in self.induction_exponents1 ]
            exponents = np.array(exponents)[:,:,np.newaxis,np.newaxis]
            ## print self.all_exponents.shape
            ## print exponents.shape
            ## sys.exit()
        else:
            exponents = self.all_exponents

        print 'Calculating drude oscillator energy using a multipole-gradient method'
        from reverse_drude_oscillators import Drudes
        d = Drudes(self.xyz1, self.xyz2, 
                    self.multipole_file1, self.multipole_file2,
                    self.axes1,self.axes2,
                    self.drude_charges1, self.drude_charges2, 
                    self.springcon1, self.springcon2,
                    #self.all_exponents,
                    exponents,
                    self.thole_param, 
                    self.slater_correction,
                    #self.electrostatic_damping_type,
                    self.induction_damping_type,
                    self.damp_charges_only)

        ## from drude_oscillators import Drudes
        ## d = Drudes(self.xyz1, self.xyz2, 
        ##             self.multipole_file1, self.multipole_file2,
        ##             self.axes1,self.axes2,
        ##             self.drude_charges1, self.drude_charges2, 
        ##             self.springcon1, self.springcon2,
        ##             #self.all_exponents,
        ##             exponents,
        ##             self.thole_param, 
        ##             self.slater_correction,
        ##             #self.electrostatic_damping_type,
        ##             self.induction_damping_type,
        ##             self.damp_charges_only)

        # Set each monomer's drude charges to zero and get drude energy in
        # order to get 2nd order induction energy
        d.qshell2 = np.zeros_like(d.qshell2)
        efield = d.get_efield(0,mon=2)
        template = '{:16.8f}'*3 + '\n'
        print 'writing efield file'
        with open('test_efield.dat','w') as f:
            f.write('Efield\n')
            for line in efield:
                f.write(template.format(*line))

        sys.exit()

        self.find_drude_positions()
        edrude_ind1 = self.get_drude_energy()

        self.qshell1 = np.zeros_like(self.qshell1)
        self.qshell2 = qshell2_save
        self.find_drude_positions()
        edrude_ind2 = self.get_drude_energy()

        edrude_ind = edrude_ind1 + edrude_ind2
        edrude_high_order = edrude_total - edrude_ind

        self.edrude_ind, self.edrude_dhf = d.get_induction_and_dhf_drude_energy()

        sys.exit()

        print 'Dispersion tests:'

        # Compute the dispersion energy and output to file; parameters may not
        # be fit here
        self.component = 4
        ff_energy = 0.0
        if self.fit_isotropic_dispersion:
            ff_energy += self.fit_component_parameters()
        elif self.fit_dispersion and self.anisotropic_atomtypes:
            # Subtract one free parameter per atomtype; a0 is constrained to
            # be the input (isotropic) cn coefficient
            self.n_isotropic_params -= 1
            ff_energy += self.fit_component_parameters()
            self.n_isotropic_params += 1
        else:
            ff_energy += self.calc_dispersion_energy()

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
