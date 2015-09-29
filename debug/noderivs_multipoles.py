# Standard Packages
import numpy as np
import sys

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
                   damping_type='None'):

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

        ###########################################################################
        ###########################################################################


        ###########################################################################
        ###################### Initialization Routines ############################

        # Initiate dictionary of functions corresponding to interaction
        # functions for describing electrostatic interactions. See Appendix F
        # in Stone's book.
        self.initialize_interaction_tensor()

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

        self.initialize_direction_vectors()

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
    def initialize_direction_vectors(self):
        '''

        Parameters
        ----------
        None

        Returns
        -------

        '''

        x1 = self.xyz1
        x2 = self.xyz2
        r = (x2[:,np.newaxis,:,:] - x1[:,:,np.newaxis,:])**2
        self.r = np.sqrt(np.sum(r,axis=-1))
        self.eab = (x2[:,np.newaxis,:,:] - x1[:,:,np.newaxis,:])/self.r[...,np.newaxis]
        self.ra = np.sum(self.ea[:,np.newaxis,np.newaxis,:,:]*self.eab[:,:,:,np.newaxis,:],axis=-1)
        self.rb = -np.sum(self.eb[:,np.newaxis,np.newaxis,:,:]*self.eab[:,:,:,np.newaxis,:],axis=-1)
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

        #assert data[0] == ['Units','bohr'] # Unit consistency check

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
            if not np.array_equal(v3,np.zeros_like(v3)):
                break
        else:
            # All vectors in molecule are parallel; hopefully molecules are
            # now aligned
            assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
            return rotation_matrix

        # Align an orthogonal vector to v1; once this vector is aligned, the
        # molecules should be perfectly aligned
        q_w,q_vec = self.get_rotation_quaternion(v3,v4,v_orth=v1)
        np.seterr(all="ignore")
        trans_local_xyz = self.rotate_local_xyz(q_w, q_vec, trans_local_xyz)
        rotation_matrix = self.rotate_local_xyz(q_w,q_vec, rotation_matrix)
        np.seterr(all="warn")

        assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
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
        #[b,c,d] = vector
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

        self.T[('00','00')] = lambda r,ra,rb,cab: 1/r

        self.T[('10','00')] = lambda r,ra,rb,cab: 1/r**2 * ra[:,2]
        self.T[('11c','00')] = lambda r,ra,rb,cab: 1/r**2 * ra[:,0]
        self.T[('11s','00')] = lambda r,ra,rb,cab: 1/r**2 * ra[:,1]

        self.T[('10','10')] = lambda r,ra,rb,cab: 1/r**3 * (3*ra[:,2]*rb[:,2] + cab[:,2,2])
        self.T[('11c','11c')] = lambda r,ra,rb,cab: 1/r**3 * (3*ra[:,0]*rb[:,0] + cab[:,0,0])
        self.T[('11s','11s')] = lambda r,ra,rb,cab: 1/r**3 * (3*ra[:,1]*rb[:,1] + cab[:,1,1])
        self.T[('11s','10')] = lambda r,ra,rb,cab: 1/r**3 * (3*ra[:,1]*rb[:,2] + cab[:,1,2])
        self.T[('11c','10')] = lambda r,ra,rb,cab: 1/r**3 * (3*ra[:,0]*rb[:,2] + cab[:,0,2])
        self.T[('11c','11s')] = lambda r,ra,rb,cab: 1/r**3 * (3*ra[:,0]*rb[:,1] + cab[:,0,1])

        self.T[('20','00')] = lambda r,ra,rb,cab: 1/r**3 * 0.5 * (3*ra[:,2]**2 - 1)
        self.T[('21c','00')] = lambda r,ra,rb,cab: 1/r**3 * np.sqrt(3.0) * ra[:,0]*ra[:,2]
        self.T[('21s','00')] = lambda r,ra,rb,cab: 1/r**3 * np.sqrt(3.0) * ra[:,1]*ra[:,2]
        self.T[('22c','00')] = lambda r,ra,rb,cab: 1/r**3 * 0.5 * np.sqrt(3.0) * (ra[:,0]**2 - ra[:,1]**2)
        self.T[('22s','00')] = lambda r,ra,rb,cab: 1/r**3 * np.sqrt(3.0) * ra[:,0]*ra[:,1]


        self.T[('20','10')] = lambda r,ra,rb,cab: 1/r**4 * 0.5 * \
                                (15*ra[:,2]**2*rb[:,2] + 6*ra[:,2]*cab[:,2,2] - 3*rb[:,2])
        self.T[('20','11c')] = lambda r,ra,rb,cab: 1/r**4 * 0.5 * \
                                (15*ra[:,2]**2*rb[:,0] + 6*ra[:,2]*cab[:,2,0] - 3*rb[:,0])
        self.T[('20','11s')] = lambda r,ra,rb,cab: 1/r**4 * 0.5 * \
                                (15*ra[:,2]**2*rb[:,1] + 6*ra[:,2]*cab[:,2,1] - 3*rb[:,1])
        self.T[('21c','10')] = lambda r,ra,rb,cab: 1/r**4 * np.sqrt(3) * \
                                (ra[:,0]*cab[:,2,2] + cab[:,0,2]*ra[:,2] + 5*ra[:,0]*ra[:,2]*rb[:,2])
        self.T[('21c','11c')] = lambda r,ra,rb,cab: 1/r**4 * np.sqrt(3) * \
                                 (ra[:,0]*cab[:,2,0] + cab[:,0,0]*ra[:,2] + 5*ra[:,0]*ra[:,2]*rb[:,0])
        self.T[('21c','11s')] = lambda r,ra,rb,cab: 1/r**4 * np.sqrt(3) * \
                                 (ra[:,0]*cab[:,2,1] + cab[:,0,1]*ra[:,2] + 5*ra[:,0]*ra[:,2]*rb[:,1])
        self.T[('21s','10')] = lambda r,ra,rb,cab: 1/r**4 * np.sqrt(3) * \
                                 (ra[:,1]*cab[:,2,2] + cab[:,1,2]*ra[:,2] + 5*ra[:,1]*ra[:,2]*rb[:,2])
        self.T[('21s','11c')] = lambda r,ra,rb,cab: 1/r**4 * np.sqrt(3) * \
                                 (ra[:,1]*cab[:,2,0] + cab[:,1,0]*ra[:,2] + 5*ra[:,1]*ra[:,2]*rb[:,0])
        self.T[('21s','11s')] = lambda r,ra,rb,cab: 1/r**4 * np.sqrt(3) * \
                                 (ra[:,1]*cab[:,2,1] + cab[:,1,1]*ra[:,2] + 5*ra[:,1]*ra[:,2]*rb[:,1])
        self.T[('22c','10')] = lambda r,ra,rb,cab: 1/r**4 * 0.5 * np.sqrt(3) * \
                                (5*(ra[:,0]**2 - ra[:,1]**2)*rb[:,2] + 2*ra[:,0]*cab[:,0,2] -
                                    2*ra[:,1]*cab[:,1,2])
        self.T[('22c','11c')] = lambda r,ra,rb,cab: 1/r**4 * 0.5 * np.sqrt(3) * \
                                (5*(ra[:,0]**2 - ra[:,1]**2)*rb[:,0] + 2*ra[:,0]*cab[:,0,0] -
                                    2*ra[:,1]*cab[:,1,0])
        self.T[('22c','11s')] = lambda r,ra,rb,cab: 1/r**4 * 0.5 * np.sqrt(3) * \
                                (5*(ra[:,0]**2 - ra[:,1]**2)*rb[:,1] + 2*ra[:,0]*cab[:,0,1] -
                                    2*ra[:,1]*cab[:,1,1])
        self.T[('22s','10')] = lambda r,ra,rb,cab: 1/r**4 * np.sqrt(3) * \
                                (5*ra[:,0]*ra[:,1]*rb[:,2] + ra[:,0]*cab[:,1,2] +
                                    ra[:,1]*cab[:,0,2])
        self.T[('22s','11c')] = lambda r,ra,rb,cab: 1/r**4 * np.sqrt(3) * \
                                 (5*ra[:,0]*ra[:,1]*rb[:,0] + ra[:,0]*cab[:,1,0] +
                                    ra[:,1]*cab[:,0,0])
        self.T[('22s','11s')] = lambda r,ra,rb,cab: 1/r**4 * np.sqrt(3) * \
                                 (5*ra[:,0]*ra[:,1]*rb[:,1] + ra[:,0]*cab[:,1,1] +
                                    ra[:,1]*cab[:,0,1])

        self.T[('20','20')] = lambda r,ra,rb,cab: 1/r**5 * 0.75 * \
                               (35*ra[:,2]**2*rb[:,2]**2 - 5*ra[:,2]**2 - 5*rb[:,2]**2 +
                                 20*ra[:,2]*rb[:,2]*cab[:,2,2] + 2*cab[:,2,2]**2 + 1)
        self.T[('20','21c')] = lambda r,ra,rb,cab: 1/r**5 * 0.5 * np.sqrt(3) * \
                                (35*ra[:,2]**2*rb[:,0]*rb[:,2] - 5*rb[:,0]*rb[:,2] +
                                  10*ra[:,2]*rb[:,0]*cab[:,2,2] +
                                  10*ra[:,2]*rb[:,2]*cab[:,2,0] +
                                  2*cab[:,2,0]*cab[:,2,2])
        self.T[('20','21s')] = lambda r,ra,rb,cab: 1/r**5 * 0.5 * np.sqrt(3) * \
                                (35*ra[:,2]**2*rb[:,1]*rb[:,2] - 5*rb[:,1]*rb[:,2] +
                                  10*ra[:,2]*rb[:,1]*cab[:,2,2] +
                                  10*ra[:,2]*rb[:,2]*cab[:,2,1] +
                                  2*cab[:,2,1]*cab[:,2,2])
        self.T[('20','22c')] = lambda r,ra,rb,cab: 1/r**5 * 0.25 * np.sqrt(3) * \
                                (35*ra[:,2]**2*(rb[:,0]**2 - rb[:,1]**2) - 5*rb[:,0]**2 +
                                  5*rb[:,1]**2 + 20*ra[:,2]*rb[:,0]*cab[:,2,0] -
                                  20*ra[:,2]*rb[:,1]*cab[:,2,1] + 2*cab[:,2,0]**2 -
                                  2*cab[:,2,1]**2)
        self.T[('20','22s')] = lambda r,ra,rb,cab: 1/r**5 * 0.5 * np.sqrt(3) * \
                                (35*ra[:,2]**2*rb[:,0]*rb[:,1] - 5*rb[:,0]*rb[:,1] +
                                  10*ra[:,2]*rb[:,0]*cab[:,2,1] +
                                  10*ra[:,2]*rb[:,1]*cab[:,2,0] +
                                  2*cab[:,2,0]*cab[:,2,1])
        self.T[('21c','21c')] = lambda r,ra,rb,cab: 1/r**5 * \
                                 (35*ra[:,0]*ra[:,2]*rb[:,0]*rb[:,2] +
                                   5*ra[:,0]*rb[:,0]*cab[:,2,2] +
                                   5*ra[:,0]*rb[:,2]*cab[:,2,0] +
                                   5*ra[:,2]*rb[:,0]*cab[:,0,2] +
                                   5*ra[:,2]*rb[:,2]*cab[:,0,0] + cab[:,0,0]*cab[:,2,2] +
                                   cab[:,0,2]*cab[:,2,0])
        self.T[('21c','21s')] = lambda r,ra,rb,cab: 1/r**5 * \
                                 (35*ra[:,0]*ra[:,2]*rb[:,1]*rb[:,2] +
                                   5*ra[:,0]*rb[:,1]*cab[:,2,2] +
                                   5*ra[:,0]*rb[:,2]*cab[:,2,1] +
                                   5*ra[:,2]*rb[:,1]*cab[:,0,2] +
                                   5*ra[:,2]*rb[:,2]*cab[:,0,1] + cab[:,0,1]*cab[:,2,2] +
                                   cab[:,0,2]*cab[:,2,1])
        self.T[('21c','22c')] = lambda r,ra,rb,cab: 1/r**5 * 0.5 * \
                                 (35*ra[:,0]*ra[:,2]*(rb[:,0]**2 - rb[:,1]**2) +
                                   10*ra[:,0]*rb[:,0]*cab[:,2,0] -
                                   10*ra[:,0]*rb[:,1]*cab[:,2,1] +
                                   10*ra[:,2]*rb[:,0]*cab[:,0,0] -
                                   10*ra[:,2]*rb[:,1]*cab[:,0,1] +
                                   2*cab[:,0,0]*cab[:,2,0] - 2*cab[:,0,1]*cab[:,2,1])
        self.T[('21c','22s')] = lambda r,ra,rb,cab: 1/r**5 * \
                                 (35*ra[:,0]*ra[:,2]*rb[:,0]*rb[:,1] +
                                   5*ra[:,0]*rb[:,0]*cab[:,2,1] +
                                   5*ra[:,0]*rb[:,1]*cab[:,2,0] +
                                   5*ra[:,2]*rb[:,0]*cab[:,0,1] +
                                   5*ra[:,2]*rb[:,1]*cab[:,0,0] + cab[:,0,0]*cab[:,2,1] +
                                   cab[:,0,1]*cab[:,2,0])
        self.T[('21s','21s')] = lambda r,ra,rb,cab: 1/r**5 * \
                                 (35*ra[:,1]*ra[:,2]*rb[:,1]*rb[:,2] +
                                   5*ra[:,1]*rb[:,1]*cab[:,2,2] +
                                   5*ra[:,1]*rb[:,2]*cab[:,2,1] +
                                   5*ra[:,2]*rb[:,1]*cab[:,1,2] +
                                   5*ra[:,2]*rb[:,2]*cab[:,1,1] + cab[:,1,1]*cab[:,2,2] +
                                   cab[:,1,2]*cab[:,2,1])
        self.T[('21s','22c')] = lambda r,ra,rb,cab: 1/r**5 * 0.5 * \
                                 (35*ra[:,1]*ra[:,2]*(rb[:,0]**2 - rb[:,1]**2) +
                                   10*ra[:,1]*rb[:,0]*cab[:,2,0] -
                                   10*ra[:,1]*rb[:,1]*cab[:,2,1] +
                                   10*ra[:,2]*rb[:,0]*cab[:,1,0] -
                                   10*ra[:,2]*rb[:,1]*cab[:,1,1] +
                                   2*cab[:,1,0]*cab[:,2,0] - 2*cab[:,1,1]*cab[:,2,1])
        self.T[('21s','22s')] = lambda r,ra,rb,cab: 1/r**5 * \
                                 (35*ra[:,1]*ra[:,2]*rb[:,0]*rb[:,1] +
                                   5*ra[:,1]*rb[:,0]*cab[:,2,1] +
                                   5*ra[:,1]*rb[:,1]*cab[:,2,0] +
                                   5*ra[:,2]*rb[:,0]*cab[:,1,1] +
                                   5*ra[:,2]*rb[:,1]*cab[:,1,0] + cab[:,1,0]*cab[:,2,1] +
                                   cab[:,1,1]*cab[:,2,0])
        self.T[('22c','22c')] = lambda r,ra,rb,cab: 1/r**5 * 0.25 * \
                                 (35*ra[:,0]**2*rb[:,0]**2 -
                                  35*ra[:,0]**2*rb[:,1]**2 -
                                  35*ra[:,1]**2*rb[:,0]**2 +
                                  35*ra[:,1]**2*rb[:,1]**2 +
                                  20*ra[:,0]*rb[:,0]*cab[:,0,0] - 
                                  20*ra[:,0]*rb[:,1]*cab[:,0,1] - 
                                  20*ra[:,1]*rb[:,0]*cab[:,1,0] + 
                                  20*ra[:,1]*rb[:,1]*cab[:,1,1] + 
                                  2*cab[:,0,0]**2 - 2*cab[:,0,1]**2 -
                                  2*cab[:,1,0]**2 + 2*cab[:,1,1]**2)
        self.T[('22c','22s')] = lambda r,ra,rb,cab: 1/r**5 * 0.5 * \
                                 (35*ra[:,0]**2*rb[:,0]*rb[:,1] - 
                                  35*ra[:,1]**2*rb[:,0]*rb[:,1] +
                                  10*ra[:,0]*rb[:,0]*cab[:,0,1] +
                                  10*ra[:,0]*rb[:,1]*cab[:,0,0] -
                                  10*ra[:,1]*rb[:,0]*cab[:,1,1] -
                                  10*ra[:,1]*rb[:,1]*cab[:,1,0] +
                                  2*cab[:,0,0]*cab[:,0,1] -
                                  2*cab[:,1,0]*cab[:,1,1])
        self.T[('22s','22s')] = lambda r,ra,rb,cab: 1/r**5 * \
                                 (35*ra[:,0]*ra[:,1]*rb[:,0]*rb[:,1] +
                                   5*ra[:,0]*rb[:,0]*cab[:,1,1] +
                                   5*ra[:,0]*rb[:,1]*cab[:,1,0] +
                                   5*ra[:,1]*rb[:,0]*cab[:,0,1] +
                                   5*ra[:,1]*rb[:,1]*cab[:,0,0] + cab[:,0,0]*cab[:,1,1] +
                                   cab[:,0,1]*cab[:,1,0])

                                  
                                  
        return self.T
####################################################################################################    


####################################################################################################    
    def get_interaction_tensor(self,i,j,interaction_type):
    #def get_interaction_tensor(self,interaction_type,r,eab,ea,eb):
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

        ## assert r.ndim == 1
        ## assert eab.ndim == 2
        ## assert ea.ndim == eb.ndim == 3

        # R is inter-site distance (ndatpts)
        # eab is unit vector from site a to b (ndatpts by 3)
        # Ea is unit vector for local axis of site a (ndatpts by 3 by 3) (each
        # axis vector expressed in the global coordinate frame)
        # Eb is unit vector for local axis of site b (ndatpts by 3 by 3)
        # cab is rotation matrix from ea to eb (ndatpts by 3 by 3)

        ## ra = np.sum(ea*eab[:,np.newaxis,:],axis=-1)
        ## rb = -np.sum(eb*eab[:,np.newaxis,:],axis=-1)
        ## cab = np.sum(ea[:,:,np.newaxis,:]*eb[:,np.newaxis,:,:],axis=-1)
        r = self.r[:,i,j]
        ra = self.ra[:,i,j]
        rb = self.rb[:,i,j]
        cab = self.cab
        cba = self.cba

        ## i = 0
        ## j = 0
        ## print ra.shape
        ## print self.ra[:,i,j].shape
        ## print rb.shape
        ## print self.rb[:,i,j].shape

        ## assert np.array_equal(ra,self.ra[:,i,j])
        ## assert np.array_equal(rb,self.rb[:,i,j])
        ## assert np.array_equal(cab,self.cab)

        try:
            return self.T[interaction_type](r,ra,rb,cab)
        except KeyError: # T for A,B isn't listed; try reversing the components
            interaction_type = (interaction_type[1],interaction_type[0])
            #cba = np.sum(eb[:,:,np.newaxis,:]*ea[:,np.newaxis,:,:],axis=-1)
            try:
                return self.T[interaction_type](r,rb,ra,cba)
            except ValueError:
                print ra.shape
                print rb.shape
                print cba.shape
                print cba[0].shape
                print cba[:,0,0].shape
                print cba[:,0,1].shape
                raise
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
        
        ## xi = self.xyz1[:,i]
        ## xj = self.xyz2[:,j]
        ## rij = (xj - xi)**2
        ## rij = np.sqrt(np.sum(rij,axis=1))
        ## eab = (xj - xi)/rij[:,np.newaxis]
        Qa = self.multipoles1[i][interaction_type[0]]
        Qb = self.multipoles2[j][interaction_type[1]]

        ## assert np.array_equal(rij, self.r[:,i,j])
        ## assert np.array_equal(eab, self.eab[:,i,j])

        int_type = (interaction_type[0].strip('Q'),interaction_type[1].strip('Q'))
        #T = self.get_interaction_tensor(int_type,rij,eab,self.ea,self.eb)
        T = self.get_interaction_tensor(i,j,int_type)

        rij = self.r[:,i,j]
        if self.damping_type == 'Tang-Toennies':
            bij = self.exponents[i][j]
            tt_order = int(interaction_type[0][1]) + int(interaction_type[1][1]) + 1
            damp = get_damping_factor(rij,bij,tt_order,self.slater_correction)
        else:
            damp = 1

        if self.verbose:
            print 'Interaction Type, Rij, Multipole interaction energy (mH)'
            print interaction_type[0], interaction_type[1], rij[0], (Qa*T*Qb)[0]*1000

        return damp*Qa*T*Qb
####################################################################################################    


