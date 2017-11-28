# Standard Packages
#from __future__ import division
import numpy as np
import sys
import sympy as sp
import os
from sympy.utilities import lambdify

# Local Packages
from functional_forms import get_damping_factor
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
    
    def __init__(self, xyz1, xyz2, 
                   multipole_file1, multipole_file2, 
                   exponents=np.array([]),
                   slater_correction=True,
                   damping_type='None',
                   damp_charges_only=True,
                   ):

        '''Initialize input variables and interaction function tensors.'''

        ###########################################################################
        ###################### Variable Initialization ############################
        self.xyz1 = xyz1
        self.xyz2 = xyz2
        self.multipole_file1 = multipole_file1
        self.multipole_file2 = multipole_file2
        self.exponents = exponents
        self.slater_correction = slater_correction
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
        #self.damping_type = 'Tang-Toennies'

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

        ###########################################################################
        ###########################################################################

        return
####################################################################################################    


####################################################################################################    
    def get_multipole_electrostatic_energy(self):
        '''Get the multipole component of the first order electrostatic energy
        between monomers for each input configuration.

        Parameters
        ----------
        None

        Returns
        -------
        self.multipole_energy : ndarray
            Multipole interaction energy for each dimer configuration as an
            array of size len(self.xyz1).

        '''
        self.multipoles1, self.local_coords1 = self.read_multipoles(self.multipole_file1)
        self.multipoles2, self.local_coords2 = self.read_multipoles(self.multipole_file2)


        self.ea = self.get_local_to_global_rotation_matrix(self.xyz1,self.local_coords1)
        self.eb = self.get_local_to_global_rotation_matrix(self.xyz2,self.local_coords2)

        self.update_direction_vectors(init=True)

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
    def update_direction_vectors(self,init=False):
        '''

        Parameters
        ----------
        init : boolean, optional
            Call to this subroutine should be slightly more involved for the
            first fall; set init to True for this first call

        Returns
        -------
        (see Appendix F in Stone's book for more detailed notation)

        r : 3darray
            Inter-site distance array (ndatpts x natoms1 x natoms2)
        eab : 4darray
           Unit vector from site a to b (ndatpts x natoms 1 x natoms2 x 3)
           vector )
        ra : 3darray
            Inter-site distance vector expressed using the local axis of site
            a
        rb : 3darray
            Inter-site distance vector expressed using the local axis of site
            b
        cab : 3darray
            Direction cosines between sites a and b
        cba : 3darray
            Direction cosines between sites b and a

        '''
        #TODO: Provide clearer descriptions of return values

        # r, eab, ra, and rb need to be updated each time self.xyz1 or
        # self.xyz2 change.
        x1 = self.xyz1
        x2 = self.xyz2
        r = (x2[:,np.newaxis,:,:] - x1[:,:,np.newaxis,:])**2
        self.r = np.sqrt(np.sum(r,axis=-1))
        self.eab = (x2[:,np.newaxis,:,:] - x1[:,:,np.newaxis,:])/self.r[...,np.newaxis]
        self.ra = np.sum(self.ea[:,np.newaxis,np.newaxis,:,:]*self.eab[:,:,:,np.newaxis,:],axis=-1)
        self.rb = -np.sum(self.eb[:,np.newaxis,np.newaxis,:,:]*self.eab[:,:,:,np.newaxis,:],axis=-1)
        if init:
            # cab and cba do not depend on self.xyz1 or self.xyz2, and only
            # need to be updated at the start of a computation.
            self.cab = np.sum(self.ea[:,:,np.newaxis,:]*self.eb[:,np.newaxis,:,:],axis=-1)
            self.cba = np.sum(self.eb[:,:,np.newaxis,:]*self.ea[:,np.newaxis,:,:],axis=-1)

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
        tmp = {}
        for line in data:
        #for line in data[2:]:
            if line == ['End']:
                break
            elif new_element_flag:
                atom = line[0]
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
        return atomic_multipoles, np.array(atomic_coordinates)
####################################################################################################    


####################################################################################################    
    def get_local_to_global_rotation_matrix(self,global_xyz,local_xyz):
        '''Compute the rotation matrix that transforms coordinates from the
        local to global coordinate frame.

        Parameters
        ----------
        global_xyz : 3darray
            xyz coordinates of all atoms and all monomer configurations for which multipole
            interactions will later be computed, of size (ndatpts,natoms,3).
        local_xyz : 2darray
            xyz coordinates of each atom in the multipolar coordinate frame,
            of size (natoms,3).

        Returns
        -------
        rotation_matrix : 3x3 array
            Rotation matrix that transforms coordinates from the local (i.e.
            monomer multipole) frame to the global coordinate frame.

        '''
        # Get rotation vector from the global axis to the local axis
        # http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
        rotation_matrix = np.array([np.identity(3) for x in global_xyz])
        trans_local_xyz = local_xyz[np.newaxis,:] - local_xyz[0]
        trans_global_xyz = global_xyz - global_xyz[:,0,np.newaxis]
        if len(global_xyz[0]) == 1:
            #if xyz is an atom, don't need to rotate local axes
            assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
            return rotation_matrix

        v1 = trans_local_xyz[:,1] - trans_local_xyz[:,0]
        v2 = trans_global_xyz[:,1] - trans_global_xyz[:,0]
        q_w,q_vec = self.get_rotation_quaternion(v1,v2)

        np.seterr(all="ignore")
        trans_local_xyz = self.rotate_local_xyz(q_w, q_vec, trans_local_xyz)
        rotation_matrix = self.rotate_local_xyz(q_w,q_vec, rotation_matrix)
        np.seterr(all="warn")

        if len(global_xyz[0]) == 2:
            #if xyz is diatomic, don't need to rotate second set of axes
            assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
            return rotation_matrix

        v1 = trans_local_xyz[:,1] - trans_local_xyz[:,0]
        v2 = trans_global_xyz[:,1] - trans_global_xyz[:,0]
        for i in range(2,len(global_xyz[0])):
            # Search for vector in molecule that is not parallel to the first
            v1b = trans_local_xyz[:,i] - trans_local_xyz[:,0]
            v2b = trans_global_xyz[:,i] - trans_global_xyz[:,0]
            v3 = np.cross(v1,v1b)
            v4 = np.cross(v2,v2b)
            if not np.allclose(v3,np.zeros_like(v3),atol=1e-8):
                break
        else:
            # All vectors in molecule are parallel; hopefully molecules are
            # now aligned
            assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
            #assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-1)
            return rotation_matrix

        # Align an orthogonal vector to v1; once this vector is aligned, the
        # molecules should be perfectly aligned
        q_w,q_vec = self.get_rotation_quaternion(v3,v4,v_orth=v1)
        np.seterr(all="ignore")
        trans_local_xyz = self.rotate_local_xyz(q_w, q_vec, trans_local_xyz)
        rotation_matrix = self.rotate_local_xyz(q_w,q_vec, rotation_matrix)
        np.seterr(all="warn")

        try:
            transformation_success = np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
            #transformation_success = np.allclose(trans_local_xyz,trans_global_xyz,atol=1e1,rtol=1e1)
            assert transformation_success
        except AssertionError:
            # print trans_local_xyz - trans_global_xyz
            mon = 1 if np.array_equal(local_xyz,self.local_coords1) else 2
            print
            print 'Warning!!! Global-to-local rotation of multipole moments failed for monomer {} .'.format(mon)

            success = np.all(np.isclose(trans_local_xyz,trans_global_xyz,atol=1e-5),axis=(1,2))
            ngeometries = success.size
            nsuccesses = np.sum(success)
            template = 'Of {} configurations, {} local axis transformation(s) succeeded, and {} failed.'
            print template.format(ngeometries,nsuccesses,ngeometries - nsuccesses)

            print 'This error commonly arises from neglecting one of the following conditions:'
            print '1. For each monomer, all internal coordinates in the .sapt file MUST be self-consistent.'
            print '2. For each monomer, internal coordinates must be self-consistent between the .mom file and .sapt files.'


            # If we're only dealing with point charges, the local-axis
            # transformation doesn't actually matter, and we can continue
            # running POInter. Otherwise, raise an error.
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
                print 'However since only point charges are listed for this monomer, the axis transformation in unimportant, and POInter will continue running.'
                print 
            else:
                print 'Fix these errors and re-run POInter.'
                print 
                raise

        return rotation_matrix
####################################################################################################    


####################################################################################################    
    def get_rotation_quaternion(self,v1,v2,v_orth=np.array([1,0,0]),tol=1e-16):
        '''Given two vectors v1 and v2, compute the rotation quaternion
        necessary to align vector v1 with v2.

        In the event that v1 and v2 are antiparallel to within some tolold
        tol, v_orth serves as a default vector about which the rotation
        from v2 to v1 will occur.

        Parameters
        ----------
        v1 : 2darray
            Vector of size (ndatpts,3) to be rotated.
        v2 : 2darray
            Vector of size (ndatpts,3) that is to remain fixed.
        v_orth : 1darray, optional
            Cartesian vector providing a default rotation vector in the event
            that v1 and v2 are antiparallel.
        tol : float, optional
            Tolerance from zero at which two vectors are still considered
            antiparallel.

        Returns
        -------
        q_w : 1darray
            Rotation amount (in radians) for each of the ndatpts by which v1
            is to be rotated.
        q_vec : 2darray
            Vector for each of the ndatpts about which v1 is to be rotated.

        '''
        v1 /= np.sqrt(np.sum(v1*v1,axis=-1))[:,np.newaxis]
        v2 /= np.sqrt(np.sum(v2*v2,axis=-1))[:,np.newaxis]

        dot = np.sum(v1*v2,axis=-1)

        q_vec = np.where(dot[:,np.newaxis] > -1.0 + tol,
                np.cross(v1,v2), v_orth)
        q_w = np.sqrt(np.sum(v1*v1,axis=-1)*np.sum(v2*v2,axis=-1)) + dot

        # Normalize quaternion
        q_norm = np.sqrt(q_w**2 + np.sum(q_vec**2,axis=-1))
        q_w /= q_norm
        q_vec /= q_norm[:,np.newaxis]

        return q_w, q_vec
####################################################################################################    


####################################################################################################    
    def rotate_local_xyz(self,a,vector=np.array([0,0,1]),local_xyz=np.array([1,2,3]),thresh=1e-14):
        """Compute the new position of a set of points 'local_xyz' in 3-space (given as a
        3-membered list) after a rotation by 2*arccos(a) about the vector [b,c,d].
    
        This method uses quaternions to accomplish the transformation. For more
        information about the mathematics of quaternions, refer to 
        http://graphics.stanford.edu/courses/cs164-09-spring/Handouts/handout12.pdf

        Parameters
        ----------
        a : 1darray
            Array of size ndatpts indicating the angle by which local_xyz is to be
            rotated, where the rotation angle phi = 2*arccos(a).
        vector : 2darray
            Array of size (ndatpts,3) containing the vector about which local_xyz
            is to be rotated.
        local_xyz : 3darray
            Array of size (ndatpts,natoms,3) in cartesian space to be rotated by phi about
            vector.

        Returns
        -------
        new_local_xyz : 3darray
            Array of size (ndatpts,natoms,3) in cartesian space corresponding
            to the rotated set of points.

        """
    
        #Compute unit quaternion a+bi+cj+dk
        b = vector[:,0]
        c = vector[:,1]
        d = vector[:,2]
        norm = np.sqrt(np.sin(np.arccos(a))**2/(b**2+c**2+d**2))
        b *= norm
        c *= norm
        d *= norm
    
        # Compute quaternion rotation matrix:
        [a2,b2,c2,d2] = [a**2,b**2,c**2,d**2]
        [ab,ac,ad,bc,bd,cd] = [a*b,a*c,a*d,b*c,b*d,c*d]
    
        rotation = np.array([[ a2+b2-c2-d2 ,  2*bc-2*ad  ,  2*bd+2*ac  ],\
                             [  2*bc+2*ad  , a2-b2+c2-d2 ,  2*cd-2*ab  ],\
                             [  2*bd-2*ac  ,  2*cd+2*ab  , a2-b2-c2+d2 ]])
        rotation = np.rollaxis(rotation,-1,0)

        # Compute rotation of local_xyz about the axis
        new_local_xyz = np.sum(rotation[:,np.newaxis,:,:]*local_xyz[...,np.newaxis,:],axis=-1)
        # Return new local_xyz unless rotation vector is ill-defined (occurs for
        # zero rotation), in which case return the original local_xyz
        new_local_xyz = np.where(np.sqrt(np.sum(vector*vector,axis=-1))[:,np.newaxis,np.newaxis] > thresh, 
                            new_local_xyz, local_xyz)
        return new_local_xyz
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

            delT = [ lambdify( reduced_args, sp.diff(v(*args), ia), modules='numpy') 
                        for ia in (xa,ya,za) ]
            self.delT[k] = delT

        print 'Finished initializing derivatives for multipole moments.'

        # If cloudpickle module available, save derivatives to file
        try:
            import cloudpickle
            print 'Saving multipoles to file:'
            print self.fpik
            with open(self.fpik,'wb') as f:
                cloudpickle.dump(self.delT, f)

        except ImportError:
            print 'For computational efficiency, download the cloudpickle module in order to serialize derivatives of multipole moments.'
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
        cab = self.cab
        cba = self.cba

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

        #print 'interaction tensor 1!!!!!'

        # If delT hasn't yet been initialized, do so now
        if not self.delT:
            # Try and read in interaction tensors from file
            try:
                import cloudpickle
                with open(self.fpik,'rb') as f:
                    self.delT = cloudpickle.load(f)
                if self.verbose:
                    print 'Read in multipole derivatives from the following file:'
                    print self.fpik
            except (ImportError, IOError):
                self.initialize_del_interaction_tensor()

        # Get r, ra, rb, and cab
        r = self.r[:,i,j]
        ra = self.ra[:,i,j]
        rb = self.rb[:,i,j]
        cab = self.cab
        cba = self.cba


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
            eainv = np.linalg.inv(self.ea)
            delT = np.sum(eainv*delT[:,np.newaxis], axis=-1)

        else: 
            raise NotImplementedError, '''Higher order derivatives of multipole
            moments have not yet been implemented or sufficiently tested,
            however this should be possible if later necessary.'''

            # This code might be right, but I haven't tested it yet
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


