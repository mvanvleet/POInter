"""
"""

__version__ = '1.0.0'
__author__ = 'Mary Van Vleet'

# Standard Packages
#from __future__ import division
import importlib
import numpy as np
import sys
import json
import os

# Local Packages
from methods import default, mastiff

############################## Global Variables and Constants ######################################
# Born-Mayer-sISA scaling exponent, as defined in Van Vleet et al. JCTC 2016
_bmsisa_exp_scale = 0.84

# Constraints suffix
_constraints_suffix = '.constraints'

####################################################################################################    
####################################################################################################    

class Parameters():
    '''Read parameters from user input.

    References
    ----------

    Attributes
    ----------

    Methods
    -------

    Known Issues
    ------------

    Units
    -----
    Atomic units are assumed throughout this module.

    '''
    
    def __init__(self, mon1, mon2, inputdir, **kwargs
                   ):

        '''Initialize input variables.'''

        ###########################################################################
        ###################### Variable Initialization ############################
        self.mon1 = mon1
        self.mon2 = mon2
        self.inputdir = inputdir

        ###########################################################################
        ###########################################################################


        ###########################################################################
        ##################### Instance Variable Defaults ##########################
        # File suffixes
        self.drude_suffix = '.drude'
        self.exp_suffix = '.exp'
        self.indexp_suffix = '.indexp'
        self.disp_suffix = '.disp'
        self.axes_suffix = '.axes'
        self.atomtypes_suffix = '.atomtypes'
        self.constraints_suffix = _constraints_suffix

        self.ignorecase = False

        ###########################################################################
        ###########################################################################


        ###########################################################################
        ###################### Initialization Routines ############################
        # Overwrite defaults with kwargs
        self.__dict__.update(kwargs)

        ###########################################################################
        ###########################################################################

        return
####################################################################################################    


####################################################################################################    
    def readParameters(self):
        '''

        Parameters
        ----------


        Returns
        -------

        '''

        self.readAtomtypes()
        self.readConstraints(self.constraint_files)

        # Exponents: Read in any constraints, from input, scale depending on functional form
        exponent_files = [self.inputdir + mon + self.exp_suffix for mon
                            in (self.mon1, self.mon2)]
        self.exponents = { atom:[params[i]['B'] for i in range(len(params))] for atom,params in
                self.params.items()}

        if self.exp_source.lower() == 'read':
            self.readExponents(exponent_files,self.exponents)
        elif self.exp_source.lower() == 'ip':
            print 'exponents from IP!'
            self.calculateIPExponents()


        if self.separate_induction_exponents:
            indexponent_files = [self.inputdir + mon + self.indexp_suffix for mon
                                in (self.mon1, self.mon2)]
            self.induction_exponents = {}
            self.readExponents(indexponent_files,self.induction_exponents)
        else:
            self.induction_exponents = self.exponents

        self.readAxes()
        self.readCParams()
        self.readDrudeCharges()

        return 
####################################################################################################    


####################################################################################################    
    def readAtomtypes(self):
        '''

        Parameters
        ----------


        Format
        -------
        .atomtypes file, whose format is identical to that of a standard .xyz
        file except that inclusion of atomic coordinates is not required.
        Example:

        <number of atoms>
        comment line
        <atomtype> [X] [Y] [Z]
        ...


        Returns
        -------

        '''
        error_message = '''The atomtypes given in your .atomtypes file
        doesn\'t match that of the .sapt file!
        
        Atomtypes (from .sapt file): {}
        Atomtypes (from .atomtypes file): {}
        '''

        # Read in atomtypes from the .atomtypes file(s)
        atomtype_files = [self.inputdir + mon + self.atomtypes_suffix for mon
                            in (self.mon1, self.mon2)]
        natoms = []
        atoms = [[],[]]
        for i,ifile in enumerate(atomtype_files):
            with open(ifile,'r') as f:
                data = [line.split() for line in f.readlines()]
                natoms.append(int(data[0][0]))
                for line in data[2:]:
                    atoms[i].append(line[0])

        if self.ignorecase:
            atoms = [ [a.upper() for a in atom] for atom in atoms ]

        self.natomtypes1, self.natomtypes2 = natoms
        self.atomtypes1, self.atomtypes2 = atoms
        self.natoms1, self.natoms2 = len(self.atoms1), len(self.atoms2)

        # Ensure that the number of atoms matches that from the .sapt file
        assert len(set(self.atoms1)) == len(self.atomtypes1),\
                    error_message.format(self.atoms1,self.atomtypes1)
        assert len(set(self.atoms2)) == len(self.atomtypes2),\
                    error_message.format(self.atoms2,self.atomtypes2)

        # Construct list of atomtypes
        self.atomtypes = list(set(self.atomtypes1 + self.atomtypes2))

        # Construct atomtypes -> atoms hash; this is only needed if we later
        # calculate exponents based on IP values
        self.atomtypes_to_atoms = { k:v for k,v in zip(self.atomtypes1,self.atoms1) }
        self.atomtypes_to_atoms.update({ k:v for k,v in zip(self.atomtypes2,self.atoms2) })

        return 
####################################################################################################    


####################################################################################################    
    def readExponents(self,exponent_files,exponents):
        '''

        Parameters
        ----------

        Format
        ------
        Atomtype and associated exponent for each atomtype present in the
        force field, with atomtype declarations separated by a newline.
        Comment lines are ignored. Units in a.u.
        Example:

        <atomtype1> <exponent1>
        <atomtype2> <exponent2>
        ...

        Returns
        -------

        '''
        format_error = '''Improperly formatted exponents file "{}". Aside from
        comments, each line of the .exp file should be in two-column format,
        with the first column representing the atomtype and the second column
        representing the exponent. Please fix the input error and re-run
        POInter.
        '''
        overide_error = '''{0} attempts to set atomtype {1}'s exponent to
        {2}, however elsewhere you have set this exponent to {3}. 
        Please fix the conflicting exponent definition and re-run POInter.
        Some suggestions:

        1. Did you set different exponent values between your .exp and
        .constraints files?
        2. Did you use the Born-Mayer-sISA functional form? If so, values from
        the .exp file have been scaled before being read into POInter, and may
        differ from the input .exp values.
        
        '''
        missing_error = '''You have not defined an exponent parameter for atomtype "{0}". 
        Please define this parameter in one of your .exp files -- {1}
        -- and re-run POInter.
        '''

        for i,ifile in enumerate(exponent_files):
            with open(ifile,'r') as f:
                raw_lines = [line.split('#')[0] for line in f.readlines()]
                data = [line.split() for line in raw_lines]
                for line in data:
                    if not line or line[0].startswith('#'):
                        continue #ignore blank lines and comments
                    assert len(line) == 2, format_error.format(ifile)
                    # Read in new atomtype exponent, check for conflicting
                    # values
                    atom, exponent = line
                    exponent = [float(exponent)*self.exp_scale]
                    if self.ignorecase:
                        atom = atom.upper()
                    assert not (exponents.has_key(atom) and
                            exponents[atom] != exponent),\
                            overide_error.format(ifile,atom,exponent,exponents[atom])
                    exponents[atom] = exponent

        # Make sure all atomtypes have an exponent parameter defined
        for atom in self.atomtypes:
            assert exponents.has_key(atom),\
                missing_error.format(atom,'\n\t\t\t'.join([''] + list(set(exponent_files))))

        return
####################################################################################################    


####################################################################################################    
    def calculateIPExponents(self,checkOveride=False):
        '''

        Parameters
        ----------
        checkOveride : boolean, default False
            Ensures that any exponents in the constraints files match
            exponents calculated from ionization potential (IP)-based
            exponents. False by default.


        Returns
        -------
        None explicitly, though self.exponents is updated

        '''
        overide_error = '''{0} attempts to set atomtype {1}'s exponent to
        {2}, however elsewhere you have set this exponent to {3}. 
        Please fix the conflicting exponent definition and re-run POInter.
        Some suggestions:

        1. Did you use the Born-Mayer-IP functional form? If so, values from
        the .exp file have been scaled before being read into POInter, and may
        differ from the input .exp values.
        
        '''

        from elementdata import Exponent

        for atom in self.atomtypes:
            element = self.atomtypes_to_atoms[atom]
            exponent = Exponent(element)
            if not self.exponents.has_key(atom):
                self.exponents[atom] = [exponent]
            elif checkOveride:
                # Make sure exponents given in constraints match IP values
                assert not self.exponents[atom] != exponent,\
                        overide_error.format(ifile,atom,exponent,self.exponents[atom])
                self.exponents[atom] = exponent

        return
####################################################################################################    


####################################################################################################    
    def readConstraints(self,constraints_files):
        '''

        Parameters
        ----------

        Format
        ------
        Atomtype and associated exponent for each atomtype present in the
        force field, with atomtype declarations separated by a newline.
        Comment lines are ignored. Units in a.u.
        Example:

        <atomtype1> <exponent1>
        <atomtype2> <exponent2>
        ...

        Returns
        -------

        '''
        format_error = '''Improperly formatted exponents file "{}". Aside from
        comments, each line of the .exp file should be in two-column format,
        with the first column representing the atomtype and the second column
        representing the exponent. Please fix the input error and re-run
        POInter.
        '''
        overide_error = '''{0} attempts to set atomtype {1}'s exponent to
        {2}, however elsewhere you have set this exponent to {3}. Please fix
        the conflicting exponent definition and re-run POInter.
        '''
        missing_error = '''You have set atomtype "{0}" as constrained, but
        no parameters were found for this atomtype in any of the defined
        constraints files: {1}
        Constraints *were* found for the following atomtypes:
            "{2}"
        Please either eliminate atomtype "{0}" from the list of constrained
        atomtypes, or change this atomtype to match one of the atomtypes
        above, and re-run POInter.
        '''
        
        constraints = {}
        for ifile in constraints_files:
            with open(ifile,'r') as f:
                constraints.update(json.load(f))

        for atom,c in constraints.items():
            print atom, c['A']
            print atom, c['comments']

        self.Aparams = [ ] # 4 components; exch, elst, ind, dhf

        # Create params dictionary of dictionaries
        self.params = {}
        for atom in self.constrained_atomtypes:
            assert constraints.has_key(atom), missing_error.format(
                            atom,'\n\t\t-- '.join([''] + constraints_files),
                            '", "'.join(constraints.keys()))
            constraints[atom]['aniso'] = [ np.array(a) for a in constraints[atom]['aniso']]
            constraints[atom]['C'] = np.array(constraints[atom]['C'])
            self.params[atom] = []
            self.params[atom].append(constraints[atom])

        return
####################################################################################################    


####################################################################################################    
    def readAxes(self):
        '''

        Parameters
        ----------

        Format
        ------
        The .axes file contains two sections: the first section declares
        anisotropic atomtypes, and the second defines the local coordinate
        system for each anisotropic atomtype.
        Format for the first section should be the atomtype followed by the
        spherical harmonic moments (y10, y20, etc.) used to describe the
        anisotropy of that atomtype. ex:
        O_H2O y10 y20

        Format for the second section should be the atom index (based on its
        position in the .atomtypes file, with indexing from zero) followed by
        the axis that is being defined (z or x), the origin of the axis vector
        (given as an atom index), and the endpoint of the axis vector (as one
        or more atom indices; the terminus of the axis vector will be treated
        as the midpoint of all atom indices). For instance, defining the
        z-axis for the oxygen atom [index 0] in water as the vector bisecting
        the two hydrogen atoms [indices 1 and 2] would be accomplished by the
        declaration
        0 z 0 1 2

        Comment lines and section headers (lines beginning with 'Anisotropic'
        or 'Axes') are ignored. The two sections should be separated by a
        blank line. 
        Template:

        <atomtype1> [sph_harm1] [sph_harm2] ...
        <atomtype2> [sph_harm1] ...
        ...
        <blank line>
        <atomtype1_index> <z/x axis> <atomtype1_origin> <atomtype1_endpoint>
        <atomtype2_index> <z/x axis> <atomtype2_origin> <atomtype2_endpoint1> [atomtype2_endpoint2] ...
        <blank line>

        Returns
        -------

        '''
        format_error = '''Improperly formatted axes file "{}". Aside
        from comments, each line of the axes subsection file should have at
        least four columns representing, respectively, the atom index, axis
        (z or x), axis origin (as an atomic index), and axis endpoint (also as
        Please fix the input error and re-run POInter.
        '''

        axes_files = [self.inputdir + mon + self.axes_suffix for mon
                            in (self.mon1, self.mon2)]

        self.anisotropic_atomtypes = []
        self.anisotropic_symmetries = {}
        self.anisotropic_axes1 = [ [ [],[] ] for i in xrange(self.natoms1)]
        self.anisotropic_axes2 = [ [ [],[] ] for i in xrange(self.natoms2)]
        for imon,ifile in enumerate(axes_files):
            with open(ifile,'r') as f:
                sections = f.read().split('\n\n')
            anisotropic_lines = [line.split('#')[0] 
                                for line in sections[0].split('\n') ]
            anisotropic_data = [line.split() for line in anisotropic_lines]
            for line in anisotropic_data:
                if not line or \
                line[0].lower().startswith('aniso'):
                #line[0].startswith('#') or
                    continue #ignore blank lines and comments
                self.anisotropic_atomtypes.append(line[0])
                self.anisotropic_symmetries[line[0]] = line[1:]
            
            axes_lines = [line.split('#')[0] 
                                for line in sections[1].split('\n') ]
            axes_data = [line.split() for line in axes_lines]
            for line in axes_data:
                # Ignore blank lines, comments, section headers, and isotropic
                # atoms (ones without spherical harmonic declarations)
                if not line or line[0].lower().startswith('axes') or\
                line[0].lower().startswith('atom'):
                #line[0].startswith('#') or\
                    continue 
                assert len(line) > 3, format_error.format(ifile)
                iatom = int(line[0])
                iaxis = 0 if line[1] == 'z' else 1 # list x and z axes seperately
                axes = self.anisotropic_axes1 if imon == 0 else self.anisotropic_axes2
                coords = [ int(i) for i in line[2:] ]
                assert not (axes[iatom][iaxis] != [] and axes[iatom][iaxis] != coords),\
                    '''Conflicting specifications for the {} axis for atom {}
                    in the file {}.  Please only use one axis specification
                    line per axis per atom.
                    '''.format(iaxis,iatom,ifile)
                axes[iatom][iaxis] = coords

        self.anisotropic_atomtypes = list(set(self.anisotropic_atomtypes))

        return
####################################################################################################    


####################################################################################################    
    def readCParams(self):
        '''

        Parameters
        ----------

        Format
        ------
        Atomtype and associated dispersion coefficients for each atomtype present in the
        force field, with atomtype declarations separated by a newline.
        Comment lines are ignored. Units in a.u.
        Example:

        <atomtype1> <c6> <c8> <c10> <c12>
        <atomtype2> <c6> <c8> <c10> <c12>
        ...

        Returns
        -------

        '''
        format_error = '''Improperly formatted dispersion file "{}". Aside
        from comments, each line of the .disp file should be in 5-column
        format, with the first column representing the atomtype and the
        remaining columns representing the C6-C12 coefficients. Please fix the
        input error and re-run POInter.
        '''
        overide_error = '''{0} attempts to set atomtype {1}'s dispersion
        coefficients to {2}, however elsewhere you have set these coefficients
        to {3}. Please fix the conflicting parameter declaration(s) and re-run
        POInter.
        '''
        missing_error = '''You have not defined dispersion parameters for atomtype "{0}". 
        Please define this parameter in one of your .disp files -- {1}
        -- and re-run POInter.
        '''

        dispersion_files = [self.inputdir + mon + self.disp_suffix for mon
                            in (self.mon1, self.mon2)]

        # Read in Cii parameters and convert to Ci parameters. For now,
        # only a geometric mean combination rule seems appropriate for
        # dispersion parameters, so we take a sqrt here.
        assert self.cij_combination_rule == 'geometric'

        self.Cparams = {}
        for i,ifile in enumerate(dispersion_files):
            with open(ifile,'r') as f:
                raw_lines = [line.split('#')[0] for line in f.readlines()]
                data = [line.split() for line in raw_lines]
                for line in data:
                    if not line or line[0].startswith('#'):
                        continue #ignore blank lines and comments
                    assert len(line) == 5, format_error.format(ifile)
                    # Read in dispersion coefficients for each atomtype, check
                    # for conflicting values
                    atom, ci = line[0], line[1:]
                    # Cn parameters are read in as Cii parameters, but internally
                    # treated as Ci parameters. To extract Ci
                    # parameters, we need to take the square root of Cii
                    ci = np.sqrt([float(i) for i in line[1:]])
                    if self.ignorecase:
                        atom = atom.upper()
                    assert not (self.Cparams.has_key(atom) and not \
                            np.allclose(self.Cparams[atom] , ci )),\
                            overide_error.format(ifile,atom,ci,self.Cparams[atom])
                    self.Cparams[atom] = ci

        # Make sure all atomtypes have a dispersion parameter defined
        for atom in self.atomtypes:
            assert self.Cparams.has_key(atom),\
                missing_error.format(atom,'\n\t\t\t'.join([''] + list(set(dispersion_files))))

        return
####################################################################################################    


####################################################################################################    
    def readDrudeCharges(self):
        '''

        Parameters
        ----------

        Format
        ------
        Atomtype and associated Drude charge for each atomtype present in the
        force field, with atomtype declarations separated by a newline.
        Optionally, atomtype-specific and/or anisotropic spring coefficients may be
        set; in this case, the file format is either 3- or 5-column, with the
        atomtype-specific spring coefficient (or x, y, and
        z local-coordinate spring coefficients, if anisotropic) declared after the
        Drude charge. Comment lines are ignored. Units in a.u.
        Example:

        <atomtype1> <drude_charge1> 
        <atomtype2> <drude_charge2> [springcon2]
        <atomtype3> <drude_charge3> [springcon3_x] [springcon3_y] [springcon3_z] 
        ...

        Returns
        -------

        '''
        format_error = '''Improperly formatted Drude file "{}". Aside from
        comments, each line of the .drude file should follow one of the
        following formats:
        
        Isotropic atomtypes with universal spring constant:
        <atomtype1> <drude_charge1> 

        Isotropic atomtypes with atomtype-specific spring constant:
        <atomtype2> <drude_charge2> <springcon2>

        Anisotropic atomtypes with atomtype-specific spring constants:
        <atomtype3> <drude_charge3> <springcon3_x> <springcon3_y> <springcon3_z> 

        Please fix the formatting error(s) and re-run POInter.
        '''
        overide_error = '''{0} attempts to set atomtype {1}'s Drude parameters to
        {2}, however elsewhere you have set this parameter to {3}. Please fix
        the conflicting parameter definitions and re-run POInter.
        '''
        missing_error = '''You have not defined a Drude charge for atomtype "{0}". 
        Please define this parameter in one of your .drude files -- {1}
        -- and re-run POInter.
        '''

        drude_files = [self.inputdir + mon + self.drude_suffix for mon
                            in (self.mon1, self.mon2)]

        self.drudes = {}
        for i,ifile in enumerate(drude_files):
            with open(ifile,'r') as f:
                raw_lines = [line.split('#')[0] for line in f.readlines()]
                data = [line.split() for line in raw_lines]
                for line in data:
                    if not line or line[0].startswith('#'):
                        continue #ignore blank lines and comments
                    if len(line) == 2:
                        # Reading atomtype and Drude charge; springcon is
                        # inferred from self.springcon param
                        atom = line[0]
                        charge = float(line[1])
                        springcon = self.springcon
                        drudes = [charge, springcon, springcon, springcon]
                    elif len(line) == 3:
                        atom = line[0]
                        charge = float(line[1])
                        springcon = float(line[2])
                        drudes = [charge, springcon, springcon, springcon]
                    elif len(line) == 5:
                        atom = line[0]
                        charge = float(line[1])
                        springcons = [float(coeff) for coeff in line[2:]]
                        drudes = [charge] + springcons
                    else:
                        raise AssertionError, format_error.format(ifile)
                    # Check for conflicting parameter definitions
                    if self.ignorecase:
                        atom = atom.upper()
                    assert not (self.drudes.has_key(atom) and not
                            np.allclose(self.drudes[atom] , drudes)),\
                            overide_error.format(ifile,atom,drudes,self.drudes[atom])
                    self.drudes[atom] = drudes

        # Make sure all atomtypes have a drude parameter defined
        for atom in self.atomtypes:
            assert self.drudes.has_key(atom),\
                missing_error.format(atom,'\n\t\t\t'.join([''] + list(set(drude_files))))

        # Put drude charges and spring coefficients as a list of lists for
        # each atom in each molecule; this is the format later required by the
        # drude_oscillators module
        self.springcon1 = []
        self.springcon2 = []
        self.drude_charges1 = []
        self.drude_charges2 = []

        for atom in self.atoms1:
            params = self.drudes[atom]
            self.drude_charges1.append(params[0])
            self.springcon1.append(params[1:])
        for atom in self.atoms2:
            params = self.drudes[atom]
            self.drude_charges2.append(params[0])
            self.springcon2.append(params[1:])

        self.springcon1 = np.array(self.springcon1)
        self.springcon2 = np.array(self.springcon2)

        self.drude_charges1 = np.array(self.drude_charges1)
        self.drude_charges2 = np.array(self.drude_charges2)

        return
####################################################################################################    


####################################################################################################    
####################################################################################################    





####################################################################################################    
####################################################################################################    
class Energies():
    '''Read parameters from user input.

    References
    ----------

    Attributes
    ----------

    Methods
    -------

    Known Issues
    ------------

    Units
    -----
    Atomic units are assumed throughout this module.

    '''
    
    def __init__(self, **kwargs
                   ):

        '''Initialize input variables and interaction function tensors.'''

        ###########################################################################
        ###################### Variable Initialization ############################

        ###########################################################################
        ###########################################################################


        ###########################################################################
        ################ Program-Defined Class Variables ##########################

        ###########################################################################
        ###########################################################################


        ###########################################################################
        ###################### Initialization Routines ############################

        self.__dict__ = kwargs

        for key in ['ignorecase','ncomponents']:
            assert self.__dict__.has_key(key)



        ###########################################################################
        ###########################################################################

        return
####################################################################################################    

    def readEnergies(self,ifile):
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
            with open(ifile,'r') as f:
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
            ## self.atomtypes = set()
            ## for xyz in self.atoms1+self.atoms2:
            ##     self.atomtypes.add(xyz)
            ## self.atomtypes = list(self.atomtypes)

            # Use .sapt file to get eff_mu and eff_kt
            # TODO: Allow user to specify eff_kt and eff_mu directly
            etot_min = np.amin(self.qm_energy[6])
            self.eff_kt = -etot_min*self.scale_weighting_temperature
            self.eff_mu = 0.0

        return
####################################################################################################    




####################################################################################################    
####################################################################################################    


####################################################################################################    
####################################################################################################    
class Settings(object):
    '''Read settings files from user input.

    References
    ----------

    Attributes
    ----------

    Methods
    -------

    Known Issues
    ------------

    Units
    -----
    Atomic units are assumed throughout this module.

    '''
    
    def __init__(self):
        # Read in default POInter settings in case user doesn't specify
        # certain settings themselves
        self.settings_files = [default.__file__]
        self.settings = self.get_variables_from_module(default)

        # Set up list of recognized settings; these are primarily found in
        # methods/default.py, but some additional required/optional arguments
        # are explicitly listed below
        self.recognized_settings = self.settings.keys()
        self.required_user_settings = ['mon1','mon2']
        self.optional_user_settings = ['energy_file',
                                       'multipole_file1','multipole_file2',
                                       'drude_read_file',
                                       'ofile_prefix', 'ofile_suffix',
                                       'output_file','output_settings_file',
                                       'constraint_files']
        self.recognized_settings += self.required_user_settings
        self.recognized_settings += self.optional_user_settings

    def __str__(self):
        settings = ['Importing configuration settings from the following files:']
        for f in self.settings_files:
            settings.append(''.join(['\t',f]))
        settings.append('\nCurrent Program Settings:')
        template = '{:40s} {}'
        kwargs = [ template.format(k,v) for (k,v) in self.settings.items()]
        kwargs.sort()
        settings += kwargs
        settings.append('')
        return '\n'.join(settings)
####################################################################################################    


####################################################################################################    
    def getSettings(self,ifiles):
        '''Read in settings from a list of custom settings file, as specified by the
        user.

        '''
        sys.dont_write_bytecode = True
        
        for ifile in ifiles:
            ifile = ifile.replace('.py','')
            config_file = ifile.replace('/','.')

            try:
                config = importlib.import_module(config_file, package=None)
            except ImportError:
                try: #attempt to import settings from the default settings library
                    config_file = '.' + config_file
                    config = importlib.import_module(config_file, package='pointer.methods')
                except ImportError:
                    raise 

            self.settings_files.append(config.__file__)

            userconfigs = self.get_variables_from_module(config)
            self.settings.update(userconfigs)

        sys.dont_write_bytecode = False

        return self.settings
####################################################################################################    


####################################################################################################    
    def checkSettings(self):
        '''
        '''

        missing_error_message = '''Missing required setting: {0}
        Please specify this setting in the settings file, and re-run POInter.
        '''

        unrecognized_error_message = '''\n"{}" is not a recognized setting!
        Please correct/eliminate this setting in your settings file, and re-run POInter.

        Recognized settings in POInter are as follows: {}
        '''

        # Make sure all *non-optional* recognized keys have been defined somewhere, either in
        # methods/default.py, or by the user.
        for key in set(self.recognized_settings) - set(self.optional_user_settings):
            assert self.settings.has_key(key), missing_error_message.format(key)

        # If the user hasn't specified some of the *optional* user settings,
        # use default settings for these variables.
        inputdir = self.settings['inputdir']
        mon1 = self.settings['mon1']
        mon2 = self.settings['mon2']
        multipoles_suffix = self.settings['multipoles_suffix']
        for key in self.optional_user_settings:
            if not self.settings.has_key(key):
                if key in ['ofile_prefix','file_prefix']:
                    self.settings['ofile_prefix'] = ''
                elif key in ['ofile_suffix','file_suffix']:
                    self.settings['ofile_suffix'] = ''
                elif key == 'drude_read_file':
                    self.settings[key] = 'edrudes.dat'
                elif key == 'multipole_file1':
                    self.settings[key] = inputdir + mon1 + multipoles_suffix
                elif key == 'multipole_file2':
                    self.settings[key] = inputdir + mon2 + multipoles_suffix
                elif key == 'energy_file':
                    self.settings[key] = inputdir + mon1 + '_' + mon2 + '.sapt'
                elif key == 'output_file':
                    self.settings[key] = self.settings['ofile_prefix'] + 'coeffs' + self.settings['ofile_suffix'] + '.out'
                elif key == 'output_settings_file':
                    self.settings[key] = self.settings['ofile_prefix'] + 'settings' + self.settings['ofile_suffix'] + '.out'
                elif key == 'constraint_files':
                    self.settings[key] = []
                    for mon in [mon1,mon2]:
                        try:
                            monfile = inputdir + mon + _constraints_suffix
                            assert os.path.isfile(monfile)
                        except AssertionError:
                            continue
                        else:
                            self.settings[key].append(monfile) 
                else:
                    self.settings[key] = None

        # Make sure the user hasn't defined any non-recognized keys
        for key in self.settings:
            assert key in self.recognized_settings,\
                unrecognized_error_message.format(key,'\n\t\t'.join(['']+ sorted(self.recognized_settings)))

        return
####################################################################################################    


####################################################################################################    
    def processSettings(self):
        '''
        '''

        # Make sure all required settings have been entered, and that all
        # input settings are valid.
        self.checkSettings()

        # Convert settings from the config file to POInter settings; in most
        # cases, there is a 1-to-1 correspondence, but a few 'meta' settings
        # can be used in the input file that control multiple POInter
        # settings.
        self.pointer_settings = {}
        for k,v in self.settings.items():
            assert k in self.recognized_settings, error_message.format(k)
            # Handle dispersion settings
            if k == 'fit_dispersion':
                if v.lower() == 'anisotropic':
                    self.pointer_settings['fit_dispersion'] = True
                    self.pointer_settings['scale_isotropic_dispersion'] = False
                elif v.lower() == 'all':
                    self.pointer_settings['fit_dispersion'] = True
                    self.pointer_settings['scale_isotropic_dispersion'] = True
                elif v.lower() == 'none':
                    self.pointer_settings['fit_dispersion'] = False
                    self.pointer_settings['scale_isotropic_dispersion'] = False
                else:
                    assert False,\
                        '"{}" is not a valid option for {}. '.format(k,v) +\
                        'Valid options are "all", "anisotropic", and "none".'
            # Choose a functional form
            elif k == 'functional_form':
                self.pointer_settings['slater_correction'] = False
                self.pointer_settings['exp_scale'] = 1.00
                self.pointer_settings['exp_source'] = 'read'
                self.pointer_settings['functional_form'] = 'born-mayer'
                if v.lower() in ['slater','slater-isa']:
                    self.pointer_settings['slater_correction'] = True
                elif v.lower() == 'born-mayer-sisa':
                    self.pointer_settings['exp_scale'] = _bmsisa_exp_scale
                    print 'Exponents are being scaled from input values by a scale factor of', _bmsisa_exp_scale
                elif v.lower() == 'born-mayer-ip':
                    self.pointer_settings['exp_source'] = 'ip'
                else:
                    assert False,\
                        '"{}" is not a valid option for {}. '.format(v,k) +\
                        'Valid options are "slater-isa", "born-mayer-sisa", and "born-mayer-ip".'
            else:
                self.pointer_settings[k] = v



        return self.pointer_settings
####################################################################################################    


####################################################################################################    
    def get_variables_from_module(self,module):
        book = {}
        if module:
            book = {key: value for key, value in module.__dict__.iteritems() if
                    not (key.startswith('__') or key.startswith('_'))}
        return book
####################################################################################################    
####################################################################################################    
