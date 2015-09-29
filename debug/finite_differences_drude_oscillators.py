# Standard Packages
import numpy as np
import sys
import os
import sympy as sp
from sympy.utilities import lambdify
from scipy.optimize import minimize

# Numpy error message settings
#np.seterr(all='raise')
from drude_oscillators import Drudes
from multipoles import Multipoles

####################################################################################################    
####################################################################################################    

class FDDrudes(Drudes):
    '''Locate optimal positions for, and evaluate energies of, drude oscillators (charge on a spring model).

    Drude oscillators are used to model polarizability in molecular systems.
    In this model, small charges (parameterized, and usually < 1.5e-) are placed on springs and
    connected to each atom in a molecule, where they are allowed to relax with
    respect to the electric field of both monomers. The purpose of this class
    is to determine relaxed positions for these drude oscillators and to
    evaluate their potential energies.

    References
    ----------
    (1) Lindan, P. J. D.; Gillan, M. J. J. Phys Condens. Matter 1993, 5, 1019.
    (2) Rick, S. W.; Stuart, S. J. Potentials and Algorithms for Incorporating
    Polarizability in Computer Simulations; 2002; Vol. 18.
    (3) Previous implementation work by Jesse McDaniel (fortran code located
    in svn repository)

    Attributes
    ----------
    xyz1 : ndarray
        Positions of all the atoms in monomer 1. The shape of xyz1 should be of
        the form xyz1[datpt,atom,xyz_coord].
    xyz2 : ndarray
        Same as xyz2, but for monomer 2.
    charges1 : 1darray
        Partial charges of all the atoms in monomer 1, one charge per atom.
    charges2 : 1darray
        Same as charges1, but for monomer 2.
    qshell1 : 1darray
        Charges of each drude oscillator in monomer1, one charge per atom.
    qshell2 : 1darray
        Same as qshell1, but for monomer 2.
    exponents : ndarray
        Array of shape (natoms1, natoms2) describing exponents (used in the
        short range portion of the force field potential) for each atom pair;
        these exponents are only needed for the Tang-Toennies damping
        functions used in this class.
    screenlength : float, optional.
        Thole parameter for Thole screening functions, defaults to 2.0.
    springcon : float, optional.
        Spring constant for the drude oscillators; only one spring constant is
        allowed for all oscillators. Defaults to 0.1.
    slater_correction : bool, optional.
        If True, modifies the form of the standard Tang-Toennies damping function to
        account for the Slater form of the repulsive potential.

    Methods
    -------
    find_drude_positions
        Given drude charges and core positions, use a conjugate gradient
        method to locate the lowest-energy configuration of the drude
        particles.
    get_induction_and_dhf_drude_energy
        After converging oscillators, compute the total drude oscillator
        energy and break this energy into 2nd order and higher order terms.

    '''
    
    def __init__(self, xyz1, xyz2, 
                   multipole_file1, multipole_file2, 
                   qshell1, qshell2, 
                   exponents,
                   screenlength=2.0, springcon=0.1, slater_correction=True,
                   damping_type='None'):
                   #charges1, charges2, 

        '''Initilialize input variables and drude positions.'''

        # Inherit initialization routines from original drudes class
        charges1 = 0 # Treat as dummy variable
        charges2 = 0
        Drudes.__init__(self, xyz1, xyz2, 
                   charges1, charges2, 
                   qshell1, qshell2, 
                   exponents,
                   screenlength, springcon, 
                   slater_correction,
                   damping_type)

        self.multipole_file1 = multipole_file1
        self.multipole_file2 = multipole_file2


        ###########################################################################
        ################ Program-Defined Class Variables ##########################

        # Initialize drude positions slightly off each core center if set to
        # True. Normally has little effect on convergence, but may matter in
        # some cases.
        self.initialize_off_center = False

        # Provide cutoffs for when to treat charges and distances as
        # effectively zero:
        self.small_q = 1e-7
        self.small_r = 1e-7

        # Verbosity settings:
        self.verbose = True

        ###########################################################################
        ###########################################################################


        ###########################################################################
        ###################### Initialization Routines ############################

        # Initialize drude oscillator positions
        self.initialize_shell_positions()

        ## # Create numerical subroutines to compute gradients for the Thole
        ## # and Tang-Toennies damping functions. Note that, for all
        ## # intramolecular contacts, Thole screening will be used, while all
        ## # intermolecular contacts will be damped via Tang-Toennies screening.
        ## print 'Creating numerical subroutines for damping functions.'
        ## bij, qi, qj, xij, yij, zij = sp.symbols("bij qi qj xij yij zij")
        
        self.damp_inter = lambda bij, xij, yij, zij : 1

        ##                         self.get_tt_damping_factor(bij,xij,yij,zij), modules='numpy')
        ## diff_damp_inter = [ sp.diff(self.get_tt_damping_factor(bij,xij,yij,zij),x)
        ##                         for x in [xij,yij,zij] ]
        ## self.del_damp_inter = [ lambdify((bij,xij,yij,zij),\
        ##                              sp.diff(self.get_tt_damping_factor(bij,xij,yij,zij),x),\
        ##                              modules='numpy') \
        ##                         for x in [xij,yij,zij] ]
        ## self.damp_intra = lambdify((qi,qj,xij,yij,zij),\
        ##                        self.get_thole_damping_factor(qi,qj,xij,yij,zij), modules='numpy')
        ## diff_damp_intra = [ sp.diff(self.get_thole_damping_factor(qi,qj,xij,yij,zij),x)
        ##                                 for x in [xij,yij,zij] ]
        ## self.del_damp_intra = [ lambdify((qi,qj,xij,yij,zij), ddamp, modules='numpy')
        ##                                    for ddamp in diff_damp_intra ]

        ## ###########################################################################
        ## ###########################################################################
        
        #self.main()

        return
####################################################################################################    


####################################################################################################    
    def find_drude_positions(self,itermax=100,thresh=1e-8):
        '''Use a finite differences method to find lowest-energy positions for drude oscillators.

        Parameters
        ----------
        itermax : int, optional.
            Maximum number of allowed iterations, defaults to 100.
        thresh : float, optional.
            Largest acceptable force (in any direction) on an oscillator.
            Defaults to 1e-8.

        Returns
        -------
        self.shell_xyz1 : ndarray
            Drude oscillator positions for monomer 1.
        self.shell_xyz2 : ndarray
            Drude oscillator positions for monomer 2.

        '''
        converged=converged1=converged2 = False
        iterno=0

        save_xyz1 = self.xyz1
        save_xyz2 = self.xyz2
        save_shell_xyz1 = self.shell_xyz1
        save_shell_xyz2 = self.shell_xyz2

        first = True
        # Initialize multipoles class in order to deal with higher-order
        # interactions. 
        # M1 class instance for the interaction of mon2 drudes with mon1
        # multipoles
        m1 = Multipoles(self.xyz1,self.shell_xyz2, self.multipole_file1,
                self.multipole_file2,
                self.exponents, self.slater_correction)
        m1.multipoles1, m1.local_coords1 = m1.read_multipoles(self.multipole_file1)
        m1.ea = m1.get_local_to_global_rotation_matrix(m1.xyz1,m1.local_coords1)
        save_m1_ea = m1.ea
        m1.multipoles2 = [ {'Q00' : q } for q in self.qshell2 ]
        # Here we're assuming that the drude charges are simple point charges;
        # thus is doesn't matter what we consider the local coordinate system
        # for these shell charges
        m1.eb = np.array([ np.identity(3) for xyz in m1.xyz2])
        save_m1_eb = m1.eb

        # M2 class instance for the interaction of mon1 drudes with mon2
        # multipoles
        m2 = Multipoles(self.shell_xyz1,self.xyz2, self.multipole_file1,
                self.multipole_file2,
                self.exponents, self.slater_correction)
        m2.multipoles2, m2.local_coords2 = m2.read_multipoles(self.multipole_file2)
        m2.eb = m2.get_local_to_global_rotation_matrix(m2.xyz2,m2.local_coords2)
        save_m2_eb = m2.eb
        m2.multipoles1 = [ {'Q00' : q } for q in self.qshell1 ]
        m2.ea = np.array([ np.identity(3) for xyz in m2.xyz1])
        save_m2_ea = m2.ea

        # Save multipole class instances as Drude oscillator class
        # variables.
        self.m1 = m1
        self.m2 = m2

        for i,(xyz1,xyz2,shell_xyz1,shell_xyz2) in \
                enumerate(zip(save_xyz1,save_xyz2,save_shell_xyz1,save_shell_xyz2)):
            self.xyz1 = xyz1[np.newaxis,:]
            self.xyz2 = xyz2[np.newaxis,:]
            shell_xyz1 = shell_xyz1[np.newaxis,:]
            shell_xyz2 = shell_xyz2[np.newaxis,:]

            # Update multipole class variables to the specific configuration
            # we're interested in.
            self.m1.ea = save_m1_ea[i:i+1]
            self.m1.eb = save_m1_eb[i:i+1]
            self.m2.ea = save_m2_ea[i:i+1]
            self.m2.eb = save_m2_eb[i:i+1]
            self.m1.xyz1 = self.xyz1
            self.m1.xyz2 = shell_xyz2
            self.m2.xyz1 = shell_xyz1
            self.m2.xyz2 = self.xyz2

            shell_xyz = shell_xyz1.flatten()
            shell_xyz = np.append(shell_xyz,shell_xyz2.flatten())

            pgtol=1e-13
            ftol=1e-15
            maxiter=500
            maxfun=10000

            res = minimize(self.get_drude_energy,shell_xyz,method='L-BFGS-B',\
                    options={'disp':True,'gtol':pgtol,'ftol':ftol,'maxiter':maxiter,'maxfun':maxfun})
            shell_xyz = res.x
            (ndatpts,natoms1,nxyz) = self.xyz1.shape
            (ndatpts,natoms2,nxyz) = self.xyz2.shape
            shift = ndatpts*natoms1*nxyz
            shell_xyz1 = np.reshape(shell_xyz[:shift],self.xyz1.shape)
            shell_xyz2 = np.reshape(shell_xyz[shift:],self.xyz2.shape)

            if first:
                self.shell_xyz1 = shell_xyz1
                self.shell_xyz2 = shell_xyz2
                first = False
            else:
                self.shell_xyz1 = np.append(self.shell_xyz1,shell_xyz1,axis=0)
                self.shell_xyz2 = np.append(self.shell_xyz2,shell_xyz2,axis=0)

        ## (ndatpts,natoms1,nxyz) = self.xyz1.shape
        ## (ndatpts,natoms2,nxyz) = self.xyz2.shape

        ## shift = ndatpts*natoms1*nxyz
        ## self.shell_xyz1 = np.reshape(shell_xyz[:shift],self.xyz1.shape)
        ## self.shell_xyz2 = np.reshape(shell_xyz[shift:],self.xyz2.shape)

        self.xyz1 = save_xyz1
        self.xyz2 = save_xyz2
        shell_xyz = self.shell_xyz1.flatten()
        shell_xyz = np.append(shell_xyz,self.shell_xyz2.flatten())

        m1.xyz2 = self.shell_xyz2
        m2.xyz1 = self.shell_xyz1

        m1 = Multipoles(self.xyz1,self.shell_xyz2, self.multipole_file1,
                self.multipole_file2,
                self.exponents, self.slater_correction)
        m1.multipoles1, m1.local_coords1 = m1.read_multipoles(self.multipole_file1)
        m1.ea = m1.get_local_to_global_rotation_matrix(m1.xyz1,m1.local_coords1)
        m1.multipoles2 = [ {'Q00' : q } for q in self.qshell2 ]
        # Here we're assuming that the drude charges are simple point charges;
        # thus is doesn't matter what we consider the local coordinate system
        # for these shell charges
        m1.eb = np.array([ np.identity(3) for xyz in m1.xyz2])

        # M2 class instance for the interaction of mon1 drudes with mon2
        # multipoles
        m2 = Multipoles(self.shell_xyz1,self.xyz2, self.multipole_file1,
                self.multipole_file2,
                self.exponents, self.slater_correction)
        m2.multipoles2, m2.local_coords2 = m2.read_multipoles(self.multipole_file2)
        m2.eb = m2.get_local_to_global_rotation_matrix(m2.xyz2,m2.local_coords2)
        m2.multipoles1 = [ {'Q00' : q } for q in self.qshell1 ]
        m2.ea = np.array([ np.identity(3) for xyz in m2.xyz1])

        print m1.multipoles1
        print m2.multipoles2

        # Re-save multipole class instances as Drude oscillator class
        # variables.
        self.m1 = m1
        self.m2 = m2

        self.drude_energy = self.get_drude_energy(shell_xyz, return_total=False)

        return self.shell_xyz1, self.shell_xyz2
####################################################################################################    
## 
## 
####################################################################################################    
    def get_induction_and_dhf_drude_energy(self):
        '''Converge drude oscillator positions; compute the potential energy
        arising from drude oscillators; return this energy as a sum of 2nd and
        higher order terms.

            Note that intramolecular permanent charges do not contribute to
            this energy, as they form part of the intramolecular energy that we
            are *not* describing.

        Parameters
        ----------

        Returns
        -------
        edrude_ind : ndarray
            Array with the same shape as self.xyz{1,2}[:] containing the 2nd
            order drude oscillator potential energy for each data point.
        edrude_high_order : ndarray
            Array with the same shape as self.xyz{1,2}[:] containing the
            higher order (that is, total drude oscillator energy minus
            edrude_ind) drude oscillator potential energy for each data point.
            
        '''
        qshell1_save = np.copy(self.qshell1)
        qshell2_save = np.copy(self.qshell2)

        # Get total drude oscillator energy
        self.find_drude_positions()
        shell_xyz = self.shell_xyz1.flatten()
        shell_xyz = np.append(shell_xyz,self.shell_xyz2.flatten())
        edrude_total = self.get_drude_energy(shell_xyz,return_total=False)
        print 'DHF shell displacements'
        print self.shell_xyz1[0] - self.xyz1[0]
        print '---'
        print self.shell_xyz2[0] - self.xyz2[0]
        print '---'

        # Set each monomer's drude charges to zero and get drude energy in
        # order to get 2nd order induction energy
        self.qshell2 = np.zeros_like(self.qshell2)
        self.find_drude_positions()
        shell_xyz = self.shell_xyz1.flatten()
        shell_xyz = np.append(shell_xyz,self.shell_xyz2.flatten())
        edrude_ind1 = self.get_drude_energy(shell_xyz,return_total=False)
        print 'Ind qshell1 shell displacements'
        print self.shell_xyz1[0] - self.xyz1[0]
        print '---'
        print self.shell_xyz2[0] - self.xyz2[0]
        print '---'

        self.qshell1 = np.zeros_like(self.qshell1)
        self.qshell2 = qshell2_save
        self.find_drude_positions()
        shell_xyz = self.shell_xyz1.flatten()
        shell_xyz = np.append(shell_xyz,self.shell_xyz2.flatten())
        edrude_ind2 = self.get_drude_energy(shell_xyz,return_total=False)
        print 'Ind qshell2 shell displacements'
        print self.shell_xyz1[0] - self.xyz1[0]
        print '---'
        print self.shell_xyz2[0] - self.xyz2[0]
        print '---'

        edrude_ind = edrude_ind1 + edrude_ind2
        edrude_high_order = edrude_total - edrude_ind

        return edrude_ind, edrude_high_order
####################################################################################################    


####################################################################################################    
    def get_drude_energy(self, shell_xyz, return_total=True):
        '''Compute the potential energy arising from drude oscillators.

            Note that intramolecular permanent charges do not contribute to
            this energy, as they form part of the intramolecular energy that we
            are *not* describing.

            If the flag include_electrostatics is set to True,
            get_drude_energy returns the intermolecular electrostatic energy
            (which arises from intermolecular interactions between permanent
            charges on either monomer). In most cases, however, we're only
            interested in the polarization energy, and include_electrostatics
            should be set to False.

        Parameters
        ----------
        include_electrostatics : boolean, optional.
            If True, includes the intermolecular electrostatic energy in the
            returned potential energy, otherwise excludes this energy
            contribution. Defaults to False.

        Returns
        -------
        edrude : ndarray
            Array with the same shape as self.xyz{1,2}[:] containing the
            drude oscillator potential energy for each data point.
            
        '''
        edrude = np.zeros_like(self.xyz1[:,0,0])

        #print shell_xyz[0]


        # Map flattened parameters onto shell_xyz1 and shell_xyz2
        ## print shell_xyz
        (ndatpts,natoms1,nxyz) = self.xyz1.shape
        (ndatpts,natoms2,nxyz) = self.xyz2.shape

        shift = ndatpts*natoms1*nxyz
        try:
            shell_xyz1 = np.reshape(shell_xyz[:shift],self.xyz1.shape)
            shell_xyz2 = np.reshape(shell_xyz[shift:],self.xyz2.shape)
        except ValueError:
            print 'ndatpts = ', ndatpts
            print 'natoms1 = ', natoms1
            print 'natoms2 = ', natoms2
            print 'nxyz = ', nxyz
            print 'shell_xyz ', shell_xyz.size
            print 'xyz ', self.xyz1.size, self.xyz2.size, self.xyz1.size + self.xyz2.size
            raise

        # Intramolecular drude energy from monomer 1
        for i,qi in enumerate(self.qshell1):
            if abs(qi) < self.small_q: continue
            for k,qj in enumerate(self.qshell1[i+1:]):
                j = k + i + 1
                if abs(qj) < self.small_q: continue

                # Shell-shell interactions
                xi = shell_xyz1[:,i,:]
                xj = shell_xyz1[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*qi*qj/rij

                # Core-shell interactions
                xi = shell_xyz1[:,i,:]
                xj = self.xyz1[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*qi*(-qj)/rij

                xi = self.xyz1[:,i,:]
                xj = shell_xyz1[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*(-qi)*qj/rij

                # Core-core interactions
                xi = self.xyz1[:,i,:]
                xj = self.xyz1[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*(-qi)*(-qj)/rij

        # Intramolecular drude energy from monomer 2
        for i,qi in enumerate(self.qshell2):
            if abs(qi) < self.small_q: continue
            for k,qj in enumerate(self.qshell2[i+1:]):
                j = k + i + 1
                if abs(qj) < self.small_q: continue

                # Shell-shell interactions
                xi = shell_xyz2[:,i,:]
                xj = shell_xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*qi*qj/rij

                # Core-shell interactions
                xi = shell_xyz2[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*qi*(-qj)/rij

                xi = self.xyz2[:,i,:]
                xj = shell_xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*(-qi)*qj/rij

                # Core-core interactions
                xi = self.xyz2[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*(-qi)*(-qj)/rij

        # Intermolecular drude energy between monomers 1 and 2
        for i,qi in enumerate(self.qshell1):
            for j,qj in enumerate(self.qshell2):
                bij = self.exponents[i,j]

                # Shell-shell interactions
                xi = shell_xyz1[:,i,:]
                xj = shell_xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                if abs(qi) > self.small_q and abs(qj) > self.small_q:
                    edrude += \
                        self.damp_inter(bij,dx[:,0],dx[:,1],dx[:,2])*qi*qj/rij

                # Core-shell interactions (does not include permanent charges on opposite monomer)
                xi = shell_xyz1[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                qcore_j = -qj 
                if abs(qi) > self.small_q and abs(qcore_j) > self.small_q:
                    edrude += \
                        self.damp_inter(bij,dx[:,0],dx[:,1],dx[:,2])*qi*(qcore_j)/rij

                xi = self.xyz1[:,i,:]
                xj = shell_xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                qcore_i = -qi 
                if abs(qcore_i) > self.small_q and abs(qj) > self.small_q:
                    edrude += \
                        self.damp_inter(bij,dx[:,0],dx[:,1],dx[:,2])*(qcore_i)*qj/rij

                # Core-core interactions
                xi = self.xyz1[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                qcore_i = -qi #+ self.charges1[i]
                qcore_j = -qj #+ self.charges2[j]
                if abs(qcore_i) > self.small_q and abs(qcore_j) > self.small_q:
                    edrude += \
                        self.damp_inter(bij,dx[:,0],dx[:,1],dx[:,2])*qcore_i*qcore_j/rij


                # Shell - permanent multipole interactions
                self.m1.xyz2 = shell_xyz2 # Update shell positions in case these have changed
                self.m1.initialize_direction_vectors()
                for mi in self.m1.multipoles1[i].keys():
                    for mj in self.m1.multipoles2[j].keys():
                        int_type = (mi,mj)
                        edrude += self.m1.get_multipole_energy(i,j,int_type)

                self.m2.xyz1 = shell_xyz1 # Update shell positions in case these have changed
                self.m2.initialize_direction_vectors()
                for mi in self.m2.multipoles1[i].keys():
                    for mj in self.m2.multipoles2[j].keys():
                        int_type = (mi,mj)
                        edrude += self.m2.get_multipole_energy(i,j,int_type)

                # Core - permanent multipole interactions
                self.m1.xyz2 = self.xyz2 # Update shell positions in case these have changed
                self.m1.initialize_direction_vectors()
                for mi in self.m1.multipoles1[i].keys():
                    for mj in self.m1.multipoles2[j].keys():
                        int_type = (mi,mj)
                        edrude -= self.m1.get_multipole_energy(i,j,int_type)
                        # Minus sign accounts for the fact that all core
                        # charges have the opposite sign of the shell charges

                self.m2.xyz1 = self.xyz1 # Update shell positions in case these have changed
                self.m2.initialize_direction_vectors()
                for mi in self.m2.multipoles1[i].keys():
                    for mj in self.m2.multipoles2[j].keys():
                        int_type = (mi,mj)
                        edrude -= self.m2.get_multipole_energy(i,j,int_type)

        # Include spring energy:
        # Spring Energy Monomer1
        dr2 = np.sum((self.xyz1 - shell_xyz1)**2, axis=-1)
        edrude += 0.5*np.sum(self.springcon*dr2, axis=-1)
        # Spring Energy Monomer 2
        dr2 = np.sum((self.xyz2 - shell_xyz2)**2, axis=-1)
        edrude += 0.5*np.sum(self.springcon*dr2, axis=-1)

        if return_total:
            #print 'Edrude = ', edrude[0]
            return np.sum(edrude)/ndatpts
        else:
            #print 'Edrude = ', edrude[0]
            return edrude
####################################################################################################    


