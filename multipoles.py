# Standard Packages
#from __future__ import division
import numpy as np
import sys
import sympy as sp
import os
from sympy.utilities import lambdify
from warnings import warn

# Local Packages
from functional_forms import get_damping_factor
import rotations
# Numpy error message settings
#np.seterr(all='raise')

####################################################################################################    
####################################################################################################    

class Multipoles:
    '''Compute the electrostatic interaction between monomers whose
    electostatic potential is described by a distributed set of multipoles.

    References
    ----------
    (1) Stone, Anthony. The Theory of Intermolecular Forces, 2nd edition.
    Chapter 3 and Appendix F, in particular, are useful for defining and
    explaining the formula used in this module.

    Attributes
    ----------
    xyz1 : ndarray
        Positions of all the atoms in monomer 1. The shape of xyz1 should be of
        the form xyz1[datpt,atom,xyz_coord].
    xyz2 : ndarray
        Same as xyz2, but for monomer 2.
    multipole_file1 : string
        Filename containing multipole moments for monomer 1. Formatting should
        be the same as that output by MULFIT, see
        https://app.ph.qmul.ac.uk/wiki/ajm:camcasp:multipoles:mulfit
        https://app.ph.qmul.ac.uk/wiki/_media/ajm:camcasp:multipoles:mulfit.pdf
        for details.
    multipole_file2 : string
        Same as multipole_file1, but for monomer 2.
    exponents : ndarray, optional
        Array of shape (natoms1, natoms2) describing exponents (used in the
        short range portion of the force field potential) for each atom pair;
        these exponents are only needed for the Tang-Toennies damping
        functions used in this class. 
    slater_correction : bool, optional.
        If True, modifies the form of the standard Tang-Toennies damping function to
        account for the Slater form of the repulsive potential.

    Methods
    -------
    get_multipole_electrostatic_energy
        Given multipole moments for atoms on both monomers 1 and 2, computes
        the multipole component of the electrostatic interaction energy
        between the two monomers for all configurations given by xyz1 and
        xyz2.

    Known Issues
    ------------
    1. Only interaction functions up to rank 2 are included.
    2. Only distributed multipoles on atoms are supported; off-site
    interaction multipoles would require slight modifications of the current
    code.

    Units
    -----
    Atomic units are assumed throughout this module.

    '''
    
    def __init__(self, 
                   mon1, mon2,
                   xyz1, xyz2, 
                   multipole_file1, multipole_file2, 
                   axes1, axes2,
                   rigid_monomers,
                   exponents=np.array([]),
                   slater_correction=True,
                   damping_type='None',
                   damp_charges_only=True,
                   **kwargs
                   ):

        '''Initialize input variables and interaction function tensors.'''

        ###########################################################################
        ###################### Variable Initialization ############################
        self.mon1 = mon1
        self.mon2 = mon2
        self.xyz1 = xyz1
        self.xyz2 = xyz2
        self.multipole_file1 = multipole_file1
        self.multipole_file2 = multipole_file2
        self.axes1 = axes1
        self.axes2 = axes2
        self.exponents = exponents
        self.slater_correction = slater_correction
        self.rigid_monomers = rigid_monomers
        # Damping Type. Acceptable options are currently 'None' and
        # 'Tang-Toennies'
        self.damping_type = damping_type
        self.damp_charges_only = damp_charges_only
        if self.damping_type == 'Tang-Toennies':
            if not self.damp_charges_only:
                print '''WARNING: Haven't yet figured out TT damp for
                multipoles; currently using a formula that makes intuitive
                sense by that may not be entirely accurate. Make MVV check
                this.'''

        self.natoms1 = len(self.xyz1[0])
        self.natoms2 = len(self.xyz2[0])
        ###########################################################################
        ###########################################################################

        ###########################################################################
        ################ Program-Defined Class Variables ##########################
        # Verbosity settings (print each interaction function, useful for
        # debugging):
        self.verbose = False

        # Right now we are only using these derivatives for isotropic drude
        # particles; it makes sense to just initialize delT for functions
        # containing '00'
        self.limited_delT = True

        # Set file for storing pickle data
        self.fpik = 'multipoles.pik'
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path) + '/'
        self.fpik = dir_path + self.fpik

        ###########################################################################
        ###########################################################################


        ###########################################################################
        ###################### Initialization Routines ############################

        # Initiate dictionary of functions corresponding to interaction
        # functions for describing electrostatic interactions. See Appendix F
        # in Stone's book.
        self.initialize_interaction_tensor()
        self.delT={}

        # Read in dictionary settings
        for k,v in kwargs.items():
            setattr(self,k,v)

        ###########################################################################
        ###########################################################################

        return
####################################################################################################    


####################################################################################################    
    def get_local_axis_parameters(self,mon=1):
        '''Get the local axis system and associated multipoles for each
        monomer in the system.

        Two algorithms exist for computing the local axis parameters.
        The first strategy (rigid_monomers = True) uses multipole moments
        expressed in the global (Cartesian) coordinate frame, and calculates
        ea/eb (the unit vectors defining the local axis system of molecule
        a/b) as the rotation matrix that transforms the molecular coordinates
        from the .sapt file to the molecular coordinates of the .mom file.

        Alternately (and perhaps conceptually more simply), and given a local axis
        system defined for each atom in each molecule (via the .axes file),
        the multipole moments themselves can be rotated into the local axis
        frame. In this case, ea/eb is simply the local axis frame for each
        monomer, which has previously been calculated in fit_ff_parameters.py
        and stored in the self.axes1/self.axes2 variables.

        Parameters
        ----------
        None explicitly, though the routine relies on several class variables
        defined elsewhere.

        Limitations
        -----------
        1. Algorithm #1 will lead to errors in cases where the internal
        coordinates between the .mom and .sapt files are not self-consistent,
        and is not allowed to be used.
        2. Algorithm #2 can lead to errors in cases where the .axes file is
        underspecified (i.e. lacks a z- or x-axis for atoms with anisotropic
        multipole moments). Error checking exists to prevent POInter from
        using Algorithm #2 under these conditions.

        Returns
        -------
        self.multipoles1 : list of dictionaries
            Multipole moments for each atom in monomer 1; given in the local
            axis system for that monomer

        self.multipoles2 : list of dictionaries
            As above, but for monomer 2

        self.ea: 4darray, size (ndatpts x natoms1 x 3 x 3)
            Unit vectors defining the local axis system for each atom in
            monomer 1 with respect to the global (dimer) coordinate system.
            self.ea.shape()[0] == ndatpts as the coordinates for monomer 1
            differ for each dimer configuration in the .sapt file.

        self.eb: 4darray, size (ndatpts x natoms2 x 3 x 3)
            As above, but for monomer 2

        '''
        if self.rigid_monomers:
            if mon == 1:
                self.atoms1, self.multipoles1, self.local_coords1 = self.read_multipoles(self.multipole_file1)
                self.ea, transformation_success1 = rotations.get_local_to_global_rotation_matrix(self.xyz1,self.local_coords1)
                if not transformation_success1:
                    self.assess_rotation_failure(1,self.xyz1,self.local_coords1)
                # Broadcast ea for the number of atoms in each monomer;
                # this is in order to keep the shape of ea/eb consistant
                # regardless of whether or not the monomers are being treated as
                # rigid or flexible
                self.ea = np.tile(self.ea[:,np.newaxis,...],(1,self.natoms1,1,1))
            elif mon == 2:
                self.atoms2, self.multipoles2, self.local_coords2 = self.read_multipoles(self.multipole_file2)
                self.eb, transformation_success2  = rotations.get_local_to_global_rotation_matrix(self.xyz2,self.local_coords2)
                if not transformation_success2:
                    self.assess_rotation_failure(2,self.xyz2,self.local_coords2)
                self.eb = np.tile(self.eb[:,np.newaxis,...],(1,self.natoms2,1,1))
            else:
                sys.exit('Not a valid monomer number')
        else:
            if mon == 1:
                # Read in original multipole moments
                self.atoms1, self.multipoles1, self.local_coords1 = self.read_multipoles(self.multipole_file1)
                # Read in local axis definitions from relevant .axes file
                # TODO: Read in axes file(s) explicitly; make these files
                # independent of the monomer file names
                self.axis_file1 = self.inputdir + self.mon1 + '.axes'
                self.axis_definitions1, self.local_axes1 = rotations.read_local_axes(
                                                            self.atoms1,self.local_coords1,self.axis_file1)
                # Rotate multipole moments into the local axis frame defined above
                global_xyz = np.eye(3)
                self.multipoles1 = rotations.rotate_multipole_moments(
                                        self.multipoles1,self.local_axes1,global_xyz)
                # Make sure that the user sufficiently specified the local axis
                # system for each atom
                self.assert_good_multipole_transformations(
                        self.axis_file1,self.multipole_file1,
                        self.atoms1,self.multipoles1,self.axis_definitions1)
                # Express each unit vector of the local axis system in global
                # (Cartesian) coordinates; these unit vectors will differ for each
                # dimer configuration, but have already been calculated and stored
                # in the self.axes1/2 variables.
                self.ea = self.axes1
            elif mon == 2:
                self.atoms2, self.multipoles2, self.local_coords2 = self.read_multipoles(self.multipole_file2)

                self.axis_file2 = self.inputdir + self.mon2 + '.axes'
                self.axis_definitions2, self.local_axes2 = rotations.read_local_axes(
                                                            self.atoms2,self.local_coords2,self.axis_file2)
                global_xyz = np.eye(3)
                self.multipoles2 = rotations.rotate_multipole_moments(
                                        self.multipoles2,self.local_axes2,global_xyz)
                self.assert_good_multipole_transformations(
                        self.axis_file2,self.multipole_file2,
                        self.atoms2,self.multipoles2,self.axis_definitions2)
                self.eb = self.axes2
            else:
                sys.exit('Not a valid monomer number')

        return 
####################################################################################################    


####################################################################################################    
    def get_multipole_electrostatic_energy(self):
        '''Get the multipole component of the first order electrostatic energy
        between monomers for each input configuration.

        Parameters
        ----------
        None explicitly, though the routine relies on several class variables
        defined elsewhere.

        Returns
        -------
        self.multipole_energy : ndarray
            Multipole interaction energy for each dimer configuration as an
            array of size len(self.xyz1).

        '''

        # Using the local axis system for each monomer, get the local
        # multipoles and axis systems (ea/eb)
        self.get_local_axis_parameters(mon=1)
        self.get_local_axis_parameters(mon=2)

        # Use ea and eb to define related axis system variables, eab, ra, rb, cab, and
        # cba
        self.update_direction_vectors(init=True)

        # Calculate the multipole energy for the overall system
        self.multipole_energy = np.zeros_like(self.xyz1[:,0,0])
        for i in xrange(self.natoms1):
            for j in xrange(self.natoms2):
                for qi in self.multipoles1[i].keys():
                    for qj in self.multipoles2[j].keys():
                        int_type = (qi,qj)
                        self.multipole_energy += self.get_multipole_energy(i,j,int_type)

        return self.multipole_energy
####################################################################################################    


####################################################################################################    
    def assert_good_multipole_transformations(
            self,axes_file,multipole_file,atoms,multipoles,axis_definitions):
        '''

        Parameters
        ----------
        multipoles: list of dictionaries
            List of spherical harmonic moments for each atom in a molecule, with
            individual moments given as a dictionary with keys 
            ['Q00', 'Q10', 'Q11c', 'Q11s', 'Q20', 'Q21c', 'Q21s', 'Q22c', 'Q22s']
            All moments are given with respect to the global coordinate frame
            global_axes

        axis_definitions: list of lists (N x 2)
            Local axis definitions for each atom in a molecule, with the outer
            list corresponding to the atom number, and the inner list
            corresponding to either the z- or x-axis definition, respectively.
            See documentation for the .axes file for more details on how this
            list of lists is constructed.

        Returns
        ------
            True if the multipole transformation passes all checks, False
            otherwise

        '''
        tol = 1e-7
        warning_template = '''

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Error!!!! In {0}, you have not specified a {1}-axis for atom {2}.
        Nevertheless, upon rotating the multipole moments in {4} into the
        local coordinate frame specified by {0}, there is a non-zero value
        listed for the {3} moment. The full listing of multipole moments for
        {2}, expressed in the local coordinate frame, is as follows: 

        {5}

        Due to the non-zero {3} moment and the underspecified {1}-axis, 
        inaccuracies can arise in calculating the multipolar electrostatic energy.
        Rather than calculate an erroneous energy, POInter will quit now.

        To fix this problem, either:
            1. Specify a {1}-axis for atom {2} in {0}.
            2. (Assuming all the dimer configurations in your .sapt file have
                the same monomer configuration(s) as in your .mom file(s)), set
                rigid_monomers = True in the input/settings.py file.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        '''

        for atom,multipole,axis in zip(atoms,multipoles,axis_definitions):
            multipole_text = [ '{:4s}  :  {:+8.6f}'.format(k,v) 
                        for (k,v) in multipole.items()]
            multipole_text = '\n\t'.join(sorted(multipole_text))
            # If a z-axis has not been explicitly defined, make sure all y10,
            # y20 moments are zero
            if not axis[0]:
                moments = ['Q10','Q20']
                for sph in moments:
                    assert not (multipole.has_key(sph) and abs(multipole[sph] > tol)),\
                        warning_template.format(axes_file,'z',atom,sph,multipole_file,
                        multipole_text)
            # If an x-axis has not been explicitly defined, make sure all
            # other higher-order moments are zero
            if not axis[1]:
                moments = ['Q11c','Q11s','Q21c','Q21s','Q22c','Q22s']
                for sph in moments:
                    assert not (multipole.has_key(sph) and abs(multipole[sph] > tol)),\
                        warning_template.format(axes_file,'x',atom,sph,multipole_file,
                        multipole_text)

        return True
####################################################################################################    


####################################################################################################    
    def update_direction_vectors(self,init=False):
        '''Given ea and eb (the local axis system for each atom expressed in
        the global (dimer) coordinate system), define related direction
        vectors. See Appendix F in Stone's book for more details. 

        Parameters
        ----------
        init : boolean, optional
            Call to this subroutine should be slightly more involved for the
            first call; set init to True for this first call

        Returns
        -------
        (see Appendix F in Stone's book for more detailed notation)

        r : 3darray
            Inter-site distance array (ndatpts x natoms1 x natoms2)
        eab : 4darray
           Unit vector from site a to b (ndatpts x natoms 1 x natoms2 x 3)
           vector )
        ra : 4darray
            Inter-site distance vector expressed using the local axis of site
            a
        rb : 4darray
            Inter-site distance vector expressed using the local axis of site
            b
        cab : 3darray
            Direction cosines between sites a and b
        cba : 3darray
            Direction cosines between sites b and a

        '''

        # r, eab, ra, and rb need to be updated each time self.xyz1 or
        # self.xyz2 change.
        x1 = self.xyz1
        x2 = self.xyz2
        r = (x2[:,np.newaxis,:,:] - x1[:,:,np.newaxis,:])**2
        self.r = np.sqrt(np.sum(r,axis=-1))
        self.eab = (x2[:,np.newaxis,:,:] - x1[:,:,np.newaxis,:])/self.r[...,np.newaxis]
        ##### OLD CODE ########
        ## self.ra = np.sum(self.ea[:,np.newaxis,np.newaxis,:,:]*self.eab[:,:,:,np.newaxis,:],axis=-1)
        ## self.rb = -np.sum(self.eb[:,np.newaxis,np.newaxis,:,:]*self.eab[:,:,:,np.newaxis,:],axis=-1)
        ##### OLD CODE ########
        ## print 'shapes:'
        ## print self.ea.shape, self.eb.shape, self.eab.shape
        self.ra = np.sum(self.ea[:,:,np.newaxis,...]*self.eab[:,:,:,np.newaxis,:],axis=-1)
        self.rb = -np.sum(self.eb[:,np.newaxis,...]*self.eab[:,:,:,np.newaxis,:],axis=-1)
        if init: # TODO Maybe not init only now? Check.
            # cab and cba do not depend on self.xyz1 or self.xyz2, and only
            # need to be updated at the start of a computation.
            ##### OLD CODE ######################
            ## self.cab = np.sum(self.ea[:,:,np.newaxis,:]*self.eb[:,np.newaxis,:,:],axis=-1)
            ## self.cba = np.sum(self.eb[:,:,np.newaxis,:]*self.ea[:,np.newaxis,:,:],axis=-1)
            ##### OLD CODE ######################

            self.cab = np.sum(self.ea[:,:,np.newaxis,:,np.newaxis,:]*self.eb[:,np.newaxis,:,np.newaxis,:,:],axis=-1)
            self.cba = np.sum(self.eb[:,:,np.newaxis,:,np.newaxis,:]*self.ea[:,np.newaxis,:,np.newaxis,:,:],axis=-1)

        return self.r, self.eab, self.ra, self.rb, self.cab, self.cba
####################################################################################################    


####################################################################################################    
    def read_multipoles(self,multipole_file):
        '''Read in multipole moments for each atom in a given monomer.
        Return list of dictionaries for each atom.

        Parameters
        ----------
        multipole_file : string
            Filename for file containing multipole moments for the monomer.
            File formatting should be standard; for each atom, there is a line
            containing element and cartesian coordinates, followed by all
            specified multipole moments and finally a blank line to denote the
            end of the section. This formatting is also used by Orient and
            Mulfit. 

        Returns
        -------
        atomic_multipoles : List of dictionaries
           List of length natoms whose entries correspond to all relevant
           multipole moments for that atomic site. Multipole information is
           store as a dictionary whose keys are the type of multiple moment (ex.
           'Q11c') and whose values are the numerical value of said multipole
           moment.
        atomic_coordinates : ndarray (natoms x 3)
           Cartesian coordinates of each atom expressed in the local
           coordinate system of that monomer.

        '''
        with open(multipole_file) as f:
            lines = f.readlines()
            data = [line.split() for line in lines]

        new_element_flag = False
        atomic_coordinates = []
        atomic_multipoles = []
        atoms = []
        tmp = {}
        for line in data:
        #for line in data[2:]:
            if line == ['End']:
                break
            elif new_element_flag:
                atom = line[0]
                atoms.append(atom)
                atomic_coordinates.append([float(i) for i in line[1:4]])
                new_element_flag = False
            elif not line: 
                # One blank line denote new section; extra blank lines should be
                # ignored
                if tmp:
                    atomic_multipoles.append(tmp.copy())
                tmp = {}
                new_element_flag = True
            elif line[0] == '!' or line[0] == 'Units': # Comment lines
                continue
            else: # line contains multipole moment
                tmp[line[0]] = float(line[-1])

        #return atomic_multipoles, np.array([np.array(line) for line in atomic_coordinates])
        return atoms, atomic_multipoles, np.array(atomic_coordinates)
####################################################################################################    


####################################################################################################    
    def assess_rotation_failure(self,mon,local_xyz,global_xyz):
        trans_local_xyz = local_xyz[np.newaxis,:] - local_xyz[0]
        trans_global_xyz = global_xyz - global_xyz[:,0,np.newaxis]

        # First, check if we're only dealing with point charges. In this case,
        # the local-axis transformation doesn't actually matter, and we can
        # continue running POInter.
        multipoles = self.multipoles1 if mon == 1 else self.multipoles2
        point_charges_only = True
        for m in multipoles:
            if m.keys() == 'Q00':
                continue
            for k,v in m.items():
                if k != 'Q00' and v != 0:
                    point_charges_only = False
                    break
            if not point_charges_only:
                break
        if point_charges_only:
            print
            warn_text = '''
            The global-to-local axis rotation between the .sapt and .mom
            coordinate systems failed for monomer {}. However, since only
            point charges are listed for this monomer, the axis transformation
            in unimportant, and POInter will continue running.
            '''.format(mon)
            warn(warn_text,stacklevel=3)
            print
            return

        # Otherwise, the local axis rotation does effect the multipole energy, and an error is
        # raised for the user
        success = np.all(np.isclose(trans_local_xyz,trans_global_xyz,atol=1e-5),axis=(-2,-1))
        ngeometries = success.size
        nsuccesses = np.sum(success)

        error_text = '''
        Warning!!! The local-to-global rotation of multipole moments failed for monomer {0}.
        Of {1} configurations, {2} local axis transformation(s) succeeded, and {3} failed.

        This error commonly arises when one of the following conditions is met:
            1. For monomer {0}, not all internal coordinates in the .sapt file are self-consistent.
            2. For monomer {0}, internal coordinates are not self-consistent between the .mom file and .sapt files.

        To avoid this error in the future, please do one of the following:
            1. For each monomer, ensure that *all* internal coordinates are consistent between the .mom and .sapt files.
            2. Set rigid_monomers = False in the file input/settings.py.
        '''.format(mon,ngeometries,nsuccesses,ngeometries - nsuccesses)

        transformation_success = False

        print
        assert transformation_success, error_text
####################################################################################################    


####################################################################################################    
    def initialize_interaction_tensor(self):
        '''Create a dictionary of functions corresponding to the list of
        interaction functions that arise in the sphericla-tensor formulation
        of electrostatic interactions. 
        
        See Stone Chapter 3, Appendix F for details.

        Parameters
        ----------
        None

        Returns
        -------
        self.T : dict
            Interaction tensor T^{ab}_{tu} as a dictionary, whose keys are
            given as a tuple of t and u (i.e., self.T[('11c','00')]
            correspons to the formula for T^{ab}_{tu} where t=11c and u=00).

        '''
        self.T = {}

        self.T[('00','00')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r

        self.T[('10','00')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**2 * za
        self.T[('11c','00')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**2 * xa
        self.T[('11s','00')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**2 * ya

        self.T[('10','10')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**3 * (3*za*zb + czz)
        self.T[('11c','11c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**3 * (3*xa*xb + cxx)
        self.T[('11s','11s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**3 * (3*ya*yb + cyy)
        self.T[('11s','10')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**3 * (3*ya*zb + cyz)
        self.T[('11c','10')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**3 * (3*xa*zb + cxz)
        self.T[('11c','11s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**3 * (3*xa*yb + cxy)

        self.T[('20','00')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**3 * 0.5 * (3*za**2 - 1)
        self.T[('21c','00')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**3 * np.sqrt(3.0) * xa*za
        self.T[('21s','00')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**3 * np.sqrt(3.0) * ya*za
        self.T[('22c','00')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**3 * 0.5 * np.sqrt(3.0) * (xa**2 - ya**2)
        self.T[('22s','00')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**3 * np.sqrt(3.0) * xa*ya


        self.T[('20','10')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * 0.5 * \
                                (15*za**2*zb + 6*za*czz - 3*zb)
        self.T[('20','11c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * 0.5 * \
                                (15*za**2*xb + 6*za*czx - 3*xb)
        self.T[('20','11s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * 0.5 * \
                                (15*za**2*yb + 6*za*czy - 3*yb)
        self.T[('21c','10')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * np.sqrt(3) * \
                                (xa*czz + cxz*za + 5*xa*za*zb)
        self.T[('21c','11c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * np.sqrt(3) * \
                                 (xa*czx + cxx*za + 5*xa*za*xb)
        self.T[('21c','11s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * np.sqrt(3) * \
                                 (xa*czy + cxy*za + 5*xa*za*yb)
        self.T[('21s','10')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * np.sqrt(3) * \
                                 (ya*czz + cyz*za + 5*ya*za*zb)
        self.T[('21s','11c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * np.sqrt(3) * \
                                 (ya*czx + cyx*za + 5*ya*za*xb)
        self.T[('21s','11s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * np.sqrt(3) * \
                                 (ya*czy + cyy*za + 5*ya*za*yb)
        self.T[('22c','10')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * 0.5 * np.sqrt(3) * \
                                (5*(xa**2 - ya**2)*zb + 2*xa*cxz -
                                    2*ya*cyz)
        self.T[('22c','11c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * 0.5 * np.sqrt(3) * \
                                (5*(xa**2 - ya**2)*xb + 2*xa*cxx -
                                    2*ya*cyx)
        self.T[('22c','11s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * 0.5 * np.sqrt(3) * \
                                (5*(xa**2 - ya**2)*yb + 2*xa*cxy -
                                    2*ya*cyy)
        self.T[('22s','10')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * np.sqrt(3) * \
                                (5*xa*ya*zb + xa*cyz +
                                    ya*cxz)
        self.T[('22s','11c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * np.sqrt(3) * \
                                 (5*xa*ya*xb + xa*cyx +
                                    ya*cxx)
        self.T[('22s','11s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**4 * np.sqrt(3) * \
                                 (5*xa*ya*yb + xa*cyy +
                                    ya*cxy)

        self.T[('20','20')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * 0.75 * \
                               (35*za**2*zb**2 - 5*za**2 - 5*zb**2 +
                                 20*za*zb*czz + 2*czz**2 + 1)
        self.T[('20','21c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * 0.5 * np.sqrt(3) * \
                                (35*za**2*xb*zb - 5*xb*zb +
                                  10*za*xb*czz +
                                  10*za*zb*czx +
                                  2*czx*czz)
        self.T[('20','21s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * 0.5 * np.sqrt(3) * \
                                (35*za**2*yb*zb - 5*yb*zb +
                                  10*za*yb*czz +
                                  10*za*zb*czy +
                                  2*czy*czz)
        self.T[('20','22c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * 0.25 * np.sqrt(3) * \
                                (35*za**2*(xb**2 - yb**2) - 5*xb**2 +
                                  5*yb**2 + 20*za*xb*czx -
                                  20*za*yb*czy + 2*czx**2 -
                                  2*czy**2)
        self.T[('20','22s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * 0.5 * np.sqrt(3) * \
                                (35*za**2*xb*yb - 5*xb*yb +
                                  10*za*xb*czy +
                                  10*za*yb*czx +
                                  2*czx*czy)
        self.T[('21c','21c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * \
                                 (35*xa*za*xb*zb +
                                   5*xa*xb*czz +
                                   5*xa*zb*czx +
                                   5*za*xb*cxz +
                                   5*za*zb*cxx + cxx*czz +
                                   cxz*czx)
        self.T[('21c','21s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * \
                                 (35*xa*za*yb*zb +
                                   5*xa*yb*czz +
                                   5*xa*zb*czy +
                                   5*za*yb*cxz +
                                   5*za*zb*cxy + cxy*czz +
                                   cxz*czy)
        self.T[('21c','22c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * 0.5 * \
                                 (35*xa*za*(xb**2 - yb**2) +
                                   10*xa*xb*czx -
                                   10*xa*yb*czy +
                                   10*za*xb*cxx -
                                   10*za*yb*cxy +
                                   2*cxx*czx - 2*cxy*czy)
        self.T[('21c','22s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * \
                                 (35*xa*za*xb*yb +
                                   5*xa*xb*czy +
                                   5*xa*yb*czx +
                                   5*za*xb*cxy +
                                   5*za*yb*cxx + cxx*czy +
                                   cxy*czx)
        self.T[('21s','21s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * \
                                 (35*ya*za*yb*zb +
                                   5*ya*yb*czz +
                                   5*ya*zb*czy +
                                   5*za*yb*cyz +
                                   5*za*zb*cyy + cyy*czz +
                                   cyz*czy)
        self.T[('21s','22c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * 0.5 * \
                                 (35*ya*za*(xb**2 - yb**2) +
                                   10*ya*xb*czx -
                                   10*ya*yb*czy +
                                   10*za*xb*cyx -
                                   10*za*yb*cyy +
                                   2*cyx*czx - 2*cyy*czy)
        self.T[('21s','22s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * \
                                 (35*ya*za*xb*yb +
                                   5*ya*xb*czy +
                                   5*ya*yb*czx +
                                   5*za*xb*cyy +
                                   5*za*yb*cyx + cyx*czy +
                                   cyy*czx)
        self.T[('22c','22c')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * 0.25 * \
                                 (35*xa**2*xb**2 -
                                  35*xa**2*yb**2 -
                                  35*ya**2*xb**2 +
                                  35*ya**2*yb**2 +
                                  20*xa*xb*cxx - 
                                  20*xa*yb*cxy - 
                                  20*ya*xb*cyx + 
                                  20*ya*yb*cyy + 
                                  2*cxx**2 - 2*cxy**2 -
                                  2*cyx**2 + 2*cyy**2)
        self.T[('22c','22s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * 0.5 * \
                                 (35*xa**2*xb*yb - 
                                  35*ya**2*xb*yb +
                                  10*xa*xb*cxy +
                                  10*xa*yb*cxx -
                                  10*ya*xb*cyy -
                                  10*ya*yb*cyx +
                                  2*cxx*cxy -
                                  2*cyx*cyy)
        self.T[('22s','22s')] = lambda r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz: 1/r**5 * \
                                 (35*xa*ya*xb*yb +
                                   5*xa*xb*cyy +
                                   5*xa*yb*cyx +
                                   5*ya*xb*cxy +
                                   5*ya*yb*cxx + cxx*cyy +
                                   cxy*cyx)

        return self.T
####################################################################################################    


####################################################################################################    
    def initialize_del_interaction_tensor(self):
        '''Given a specified rank of multipole moments t and u (on sites a and
        b, respectively) separated by a distance r, where eab is a unit vector
        in the direction from site a to site b and ea/eb describe the local
        axis of that multipole moment, compute the interaction tensor
        T^{ab}_{tu}.

        Parameters
        ----------
        interaction_type : tuple of strings
            2-membered tuple containing, respectively, moments t and u. 
        r : 1darray
            Distance between sites a and b.
        eab : 2darray, size (ndatpts, 3)
            Unit vector from site a to b
        ea : 3darray, size (ndatpts,3,3)
            Unit vector for local axis of site a, where each axis vector is
            expressed in the global coordinate system
        eb : 3darray, size (ndatpts,3,3)
            Same as ea, but for site b

        Returns
        -------
        The numerical evaluation of T^{ab}_{tu} for a given r,eab,ea, and eb.

        '''

        print 'Initializing derivatives for multipole moments.'
        self.delT = {}

        if not self.T: #make sure T dictionary has already been created
            self.initialize_interaction_tensor()

        r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz =\
                sp.symbols(' r xa ya za xb yb zb cxx cxy cxz cyx cyy cyz czx czy czz' )
        for k,v in self.T.items():
            if self.limited_delT and '00' not in k:
                continue
            elif not self.limited_delT:
                raise NotTestedError, 'Code has not been tested for multipole derivatives aside from those using point charges!'

            # Expand r and the unit vectors of a in terms of xa,ya, and za
            # (the cartesian vector marking the distance and direction from site a to b)

            # First set up sympy expressions with respect to ra
            xa, ya, za = sp.symbols('xa ya za')

            r = (xa**2 + ya**2 + za**2)**.5
            exa = xa/r
            eya = ya/r
            eza = za/r

            exb = (cxx*exa + cyx*eya + czx*eza)
            eyb = (cxy*exa + cyy*eya + czy*eza)
            ezb = (cxz*exa + cyz*eya + czz*eza)

            xb = -exb*r
            yb = -eyb*r
            zb = -ezb*r

            args = (r,exa,eya,eza,exb,eyb,ezb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz)
            reduced_args = (xa,ya,za,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz)

            # Compute del_aT^{ab}_{tu}, where tu = k; here del_a indicates
            # that we're computing the del operator in the local coordinate
            # frame of monomer a (keep in mind this may be different than the
            # global coordinate frame)
            args = (r,exa,eya,eza,exb,eyb,ezb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz)
            reduced_args = (xa,ya,za,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz)

            delT = [ lambdify( reduced_args, sp.diff(v(*args), ia), modules='numexpr') 
                        for ia in (xa,ya,za) ]
            self.delT[k] = delT

        print 'Finished initializing derivatives for multipole moments.'

        # If dill module available, save derivatives to file
        try:
            import dill
            dill.settings['recurse'] = True
            print 'Saving multipoles to file:'
            print self.fpik
            with open(self.fpik,'wb') as f:
                dill.dump(self.delT, f)

        except ImportError:
            print 'For computational efficiency, download the dill module in order to serialize derivatives of multipole moments.'
            pass

        return
####################################################################################################    


####################################################################################################    
    def get_interaction_tensor(self,i,j,interaction_type):
        '''Given a specified rank of multipole moments t and u (on sites a and
        b, respectively) separated by a distance r, where eab is a unit vector
        in the direction from site a to site b and ea/eb describe the local
        axis of that multipole moment, compute the interaction tensor
        T^{ab}_{tu}.

        Parameters
        ----------
        interaction_type : tuple of strings
            2-membered tuple containing, respectively, moments t and u. 
        r : 1darray
            Distance between sites a and b.
        eab : 2darray, size (ndatpts, 3)
            Unit vector from site a to b
        ea : 3darray, size (ndatpts,3,3)
            Unit vector for local axis of site a, where each axis vector is
            expressed in the global coordinate system
        eb : 3darray, size (ndatpts,3,3)
            Same as ea, but for site b

        Returns
        -------
        The numerical evaluation of T^{ab}_{tu} for a given r,eab,ea, and eb.

        '''

        # R is inter-site distance (ndatpts)
        # eab is unit vector from site a to b (ndatpts by 3)
        # Ea is unit vector for local axis of site a (ndatpts by 3 by 3) (each
        # axis vector expressed in the global coordinate frame)
        # Eb is unit vector for local axis of site b (ndatpts by 3 by 3)
        # cab is rotation matrix from ea to eb (ndatpts by 3 by 3)

        r = self.r[:,i,j]
        ra = self.ra[:,i,j]
        rb = self.rb[:,i,j]
        ###### OLD COdE ########
        ## cab = self.cab
        ## cba = self.cba
        ###### OLD COdE ########
        cab = self.cab[:,i,j]
        cba = self.cba[:,j,i]

        # Flatten numpy arguments
        xa = ra[:,0]
        ya = ra[:,1]
        za = ra[:,2]

        xb = rb[:,0]
        yb = rb[:,1]
        zb = rb[:,2]

        #try:
        if interaction_type in self.T:
            cxx = cab[:,0,0]
            cxy = cab[:,0,1]
            cxz = cab[:,0,2]

            cyx = cab[:,1,0]
            cyy = cab[:,1,1]
            cyz = cab[:,1,2]

            czx = cab[:,2,0]
            czy = cab[:,2,1]
            czz = cab[:,2,2]
            args = (r,xa,ya,za,xb,yb,zb,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz)
            return self.T[interaction_type](*args)

        else: # T for A,B isn't listed; try reversing the components
            cxx = cba[:,0,0]
            cxy = cba[:,0,1]
            cxz = cba[:,0,2]

            cyx = cba[:,1,0]
            cyy = cba[:,1,1]
            cyz = cba[:,1,2]

            czx = cba[:,2,0]
            czy = cba[:,2,1]
            czz = cba[:,2,2]

            args = (r,xb,yb,zb,xa,ya,za,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz)
            interaction_type = (interaction_type[1],interaction_type[0])

            return self.T[interaction_type](*args)
####################################################################################################    


####################################################################################################    
    def get_del_interaction_tensor(self,i,j,interaction_type):
        '''Given a specified rank of multipole moments t and u (on sites a and
        b, respectively) separated by a distance r, where eab is a unit vector
        in the direction from site a to site b and ea/eb describe the local
        axis of that multipole moment, compute the interaction tensor
        T^{ab}_{tu}.

        Parameters
        ----------
        interaction_type : tuple of strings
            2-membered tuple containing, respectively, moments t and u. 
        r : 1darray
            Distance between sites a and b.
        eab : 2darray, size (ndatpts, 3)
            Unit vector from site a to b
        ea : 3darray, size (ndatpts,3,3)
            Unit vector for local axis of site a, where each axis vector is
            expressed in the global coordinate system
        eb : 3darray, size (ndatpts,3,3)
            Same as ea, but for site b

        Returns
        -------
        The numerical evaluation of T^{ab}_{tu} for a given r,eab,ea, and eb.

        '''

        # If delT hasn't yet been initialized, do so now
        if not self.delT:
            # Try and read in interaction tensors from file
            try:
                import dill
                dill.settings['recurse'] = True
                with open(self.fpik,'rb') as f:
                    self.delT = dill.load(f)
                if self.verbose:
                    print 'Read in multipole derivatives from the following file:'
                    print self.fpik
            except (ImportError, IOError):
                self.initialize_del_interaction_tensor()

        # Get r, ra, rb, and cab
        ea = self.ea[:,i]
        r = self.r[:,i,j]
        ra = self.ra[:,i,j]
        rb = self.rb[:,i,j]
        ###### OLD COdE ########
        ## cab = self.cab
        ## cba = self.cba
        ###### OLD COdE ########
        cab = self.cab[:,i,j]
        cba = self.cba[:,j,i]


        if interaction_type in self.delT:
            # Flatten numpy arguments
            xa = ra[:,0]
            ya = ra[:,1]
            za = ra[:,2]

            xb = rb[:,0]
            yb = rb[:,1]
            zb = rb[:,2]

            cxx = cab[:,0,0]
            cxy = cab[:,0,1]
            cxz = cab[:,0,2]

            cyx = cab[:,1,0]
            cyy = cab[:,1,1]
            cyz = cab[:,1,2]

            czx = cab[:,2,0]
            czy = cab[:,2,1]
            czz = cab[:,2,2]

            args = (r*xa,r*ya,r*za,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz)
            delT = np.array([d(*args) for d in self.delT[interaction_type]])

            # Transform to global coordinates
            delT = np.swapaxes(delT,0,1)
            eainv = np.linalg.inv(ea)
            ###### OLD CODE ###############
            #eainv = np.linalg.inv(self.ea)
            #delT = np.sum(eainv*delT[:,np.newaxis], axis=-1)
            ###### OLD CODE ###############
            delT = np.sum(eainv*delT[:,np.newaxis], axis=-1)
            #delT = np.sum(eainv*delT[:,np.newaxis,:,np.newaxis], axis=-1)

        else: 
            raise NotImplementedError, '''Higher order derivatives of multipole
            moments have not yet been implemented or sufficiently tested,
            however this should be possible to implement if later necessary.
            See source code for my initial guess as to the correct
            implementation.'''

            # WARNING! This code might be right, but I haven't tested it yet
            xa = rb[:,0]
            ya = rb[:,1]
            za = rb[:,2]

            xb = ra[:,0]
            yb = ra[:,1]
            zb = ra[:,2]

            cxx = cba[:,0,0]
            cxy = cba[:,0,1]
            cxz = cba[:,0,2]

            cyx = cba[:,1,0]
            cyy = cba[:,1,1]
            cyz = cba[:,1,2]

            czx = cba[:,2,0]
            czy = cba[:,2,1]
            czz = cba[:,2,2]

            interaction_type = (interaction_type[1],interaction_type[0])

            args = (r*xa,r*ya,r*za,cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz)
            delT = np.array([d(*args) for d in self.delT[interaction_type]])

            # Transform to global coordinates
            delT = np.swapaxes(delT,0,1)
            eainv = np.linalg.inv(-self.eb)
            delT = np.sum(eainv*delT[:,np.newaxis], axis=-1)

        return delT


####################################################################################################    


####################################################################################################    
    def get_multipole_energy(self,i,j,interaction_type):
        '''Given two multipole moments (specified by interaction_type) on
        atoms i and j, respectively, computes the multipole energy arising
        from the interaction of these two moments.

        Paramters
        ---------
        i : integer
            Index specifying a particular atom in monomer 1.
        j : integer
            Same as i, but for monomer 2.
        interaction_type : tuple
            Tuple specifying t and u in the evaluation of T^{ab}_{tu}.

        Returns
        -------
        multipole_energy : 1darray, size ndatpts
            Mutipole energy arising from the interaction of two multipole
            moments on atom i in monomer 1 and atom j in monomer 2.

        '''
        
        Qa = self.multipoles1[i][interaction_type[0]]
        Qb = self.multipoles2[j][interaction_type[1]]

        int_type = (interaction_type[0].strip('Q'),interaction_type[1].strip('Q'))
        T = self.get_interaction_tensor(i,j,int_type)

        rij = self.r[:,i,j]
        if self.damping_type == 'Tang-Toennies':
            if self.damp_charges_only and not interaction_type == ('Q00','Q00'):
                damp = 1
            else:
                bij = self.exponents[i][j]
                tt_order = int(interaction_type[0][1]) + int(interaction_type[1][1]) + 1
                if bij.shape != (1,1):
                    raise NotImplementedError,\
                            '''The mathematical form of the Tang-Toennies damping differs if
                            the repulsive potential is comprised of multiple
                            exponents (see Tang, K. T.; Toennies, J. P.  Surf.
                            Sci. 1992, 279, L203-L206 for details), and this
                            (more complicated) functional form has not yet
                            been included in this fitting program.'''
                # Only use the first exponent in calculating the damping
                # factor; see warning above for dealing with multiple
                # exponents.
                bij = bij[0][0]
                damp = get_damping_factor(None,rij,bij,tt_order,self.slater_correction)
        else:
            damp = 1

        if self.verbose:
            print 'Interaction Type, Rij, Multipole interaction energy (mH)'
            print interaction_type[0], interaction_type[1], rij[0], (Qa*T*Qb)[0]*1000

        return damp*Qa*T*Qb
####################################################################################################    


