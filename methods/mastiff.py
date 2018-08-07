__version__ = '1.1.0'

##########################################################################
########################## MASTIFF Defaults ##############################
##########################################################################
# Radial functional forms f(r); see Stone's book for more details.
# Options are 'slater', 'stone', 'born-mayer', 'born-mayer-sisa', or 'lennard-jones'
functional_form           =    'slater'

# Combination rule settings: Select combination rules for each A prefactors, B
# exponents, and C dispersion coefficients. Options are as follows:
#   aij: 'saptff', 'waldman-hagler5', 'geometric' 
#   bij: 'saptff', 'waldman-hagler5', 'geometric_mean', 'arithmetic_mean'
#   cij: 'geometric'
aij_combination_rule      =    'geometric'
bij_combination_rule      =    'geometric_mean'
cij_combination_rule      =    'geometric'

# Weighting function settings
scale_weighting_temperature         =  5.0
weighted_rmse_cutoff                =  0.0

# Atomtype defaults
lone_pair_flags                     =  ['Du' , 'lp']
constrained_atomtypes               =  []
ignorecase                          =  False
##########################################################################
##########################################################################


##########################################################################
##################### Component-Specific Settings ########################
##########################################################################
# Electrostatic Settings: choose which multipole files the program should use,
# and specify whether or not each of the monomer configurations from the .sapt
# file have the same internal coordinates as the monomers in each .mom file
multipoles_suffix          =   '_ISA_L2.mom'
rigid_monomers             =    True
electrostatic_damping_type          =  'None'
include_slater_charge_penetration   =  False

# Exchange Settings: fit_bii selects whether or not to treat the ISA
# short-range exponents are soft- (fit_bii=True) or hard-constraints
# (fit_bii=False)
fit_bii                    =    False
exact_radial_correction             =  False
exponent_source                     =  'ISA'

# Induction Settings: Choose the type and parameters for the polarization
# damping functions. Options for thole_damping_type are 'thole_tinker' and
# 'thole_linear', and good defaults for thole_param are the 0.33 and 2.0 with
# respect to the two different damping types
# respectively
thole_damping_type         =   'thole_tinker'
thole_param                =    0.36
springcon                           =  0.1
separate_induction_exponents        =  False
induction_damping_type              =  'Thole'
drude_method                        =  'multipole-gradient'

# Dispersion Settings: Choose which parameters to fit to the dispersion energies. Fit options
# include 'none' (to fit no parameters), 'anisotropic' (to just fit
# anisotropic dispersion parameters, but to leave isotropic dispersion
# coefficients unscaled), and 'all' (to fit both anisotropic and isotropic
# dispersion coefficients)
fit_dispersion             =    'anisotropic'


# Residual error Settings: If set to true, fits a final A parameter to errors in the total
# energy in an effort to reduce systematic errors in the total energy
fit_residuals              =    False
##########################################################################
##########################################################################


##########################################################################
######################### I/O Settings ###################################
##########################################################################
# Input Settings
inputdir = 'input/'


##########################################################################
##########################################################################
