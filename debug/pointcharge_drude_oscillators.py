# Standard Packages
import numpy as np
import sys
import os
import sympy as sp
from sympy.utilities import lambdify

# Numpy error message settings
#np.seterr(all='raise')

####################################################################################################    
####################################################################################################    

class Drudes:
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
                   charges1, charges2, 
                   qshell1, qshell2, 
                   exponents,
                   screenlength=2.0, 
                   springcon=0.1, 
                   slater_correction=True,
                   inter_damping_type='None'):

        '''Initilialize input variables and drude positions.'''

        ###########################################################################
        ###################### Variable Initialization ############################
        self.xyz1 = xyz1
        self.xyz2 = xyz2
        self.charges1 = charges1
        self.charges2 = charges2
        self.qshell1 = qshell1
        self.qshell2 = qshell2
        self.exponents = exponents
        self.screenlength = screenlength
        self.springcon = springcon
        self.slater_correction = slater_correction

        self.natoms1 = len(self.qshell1)
        self.natoms2 = len(self.qshell2)
        ###########################################################################
        ###########################################################################

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

        # Create numerical subroutines to compute gradients for the Thole
        # and Tang-Toennies damping functions. Note that, for all
        # intramolecular contacts, Thole screening will be used, while all
        # intermolecular contacts will be damped via Tang-Toennies screening.
        print 'Creating numerical subroutines for damping functions.'
        bij, qi, qj, xij, yij, zij = sp.symbols("bij qi qj xij yij zij")
        if inter_damping_type == 'Tang-Toennies':
            self.damp_inter = lambdify((bij,xij,yij,zij),\
                                    self.get_tt_damping_factor(bij,xij,yij,zij), modules='numpy')
            diff_damp_inter = [ sp.diff(self.get_tt_damping_factor(bij,xij,yij,zij),x)
                                    for x in [xij,yij,zij] ]
            self.del_damp_inter = [ lambdify((bij,xij,yij,zij),\
                                         sp.diff(self.get_tt_damping_factor(bij,xij,yij,zij),x),\
                                         modules='numpy') \
                                    for x in [xij,yij,zij] ]
        elif inter_damping_type == 'None':
            self.damp_inter = lambda bij, xij, yij, zij : 1
            self.del_damp_inter = [ lambda bij, xij, yij, zij : 0
                                    for x in [xij,yij,zij] ]
        else:
            sys.exit('Unknown Damping Type ' + inter_damping_type)

        self.damp_intra = lambdify((qi,qj,xij,yij,zij),\
                               self.get_thole_damping_factor(qi,qj,xij,yij,zij), modules='numpy')
        diff_damp_intra = [ sp.diff(self.get_thole_damping_factor(qi,qj,xij,yij,zij),x)
                                        for x in [xij,yij,zij] ]
        self.del_damp_intra = [ lambdify((qi,qj,xij,yij,zij), ddamp, modules='numpy')
                                           for ddamp in diff_damp_intra ]

        ###########################################################################
        ###########################################################################

        return
####################################################################################################    


####################################################################################################    
    def initialize_shell_positions(self,drude_initial=0.005):
        '''Provide an initial guess for the positions of the drude oscillators.

        Parameters
        ----------
        drude_initial : float, optional.
            If class variable initialize_off_center is set to true, controls
            the magnitude of the displacement of the oscillator positions.
            Defaults to 0.005.

        Returns
        -------
        Nothing, though self.shell_xyz{1,2} are updated.

        '''
        # Initialize drude positions slightly off each core center
        if self.initialize_off_center:
            self.shell_xyz1 = np.copy(self.xyz1) + \
                    drude_initial*self.get_random_unit_vec(self.xyz1)
            self.shell_xyz2 = np.copy(self.xyz2) + \
                    drude_initial*self.get_random_unit_vec(self.xyz2)
        else:
            self.shell_xyz1 = np.copy(self.xyz1)
            self.shell_xyz2 = np.copy(self.xyz2)
        return
####################################################################################################    


####################################################################################################    
    def find_drude_positions(self,itermax=100,thresh=1e-8):
        '''Use a conjugate gradient method to find lowest-energy positions for drude oscillators.

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
        old_forces1=old_forces2 = 0.0 #values here are placeholders only
        old_search_vec1=old_search_vec2 = 0.0

        while not converged:
            if iterno > itermax:
                sys.exit('Too many iterations to find drude oscillator positions!')

            print '----'+str(iterno)+'-----'
            print 'shell positions'
            print self.shell_xyz1[0] - self.xyz1[0]

            forces1 = np.zeros_like(self.xyz1)
            for i in xrange(self.natoms1):
                print 'Monomer 1, Atom #'+str(i)
                # Compute forces on drude particles due to surrounding efield
                if np.abs(self.qshell1[i]) > self.small_q:
                    # Avoid unnecessary computation of get_efield if qshell is
                    # sufficiently small.
                    forces1[:,i,:] = self.qshell1[i]*self.get_efield(i,mon=1) 
            # For each particle, determine if forces are balanced out by the
            # spring tension in the oscillator. If any forces are not
            # cancelled out by the spring force, flag system as unconverged.
            x1 = self.shell_xyz1
            x2 = self.xyz1
            dx = x1 - x2
            forces1 = forces1 - self.springcon*dx
            converged1 = np.all(np.abs(forces1) < thresh)

            print 'forces'
            print forces1[0]
            print '----'

            if not converged1:
                lambda1, search_vec1 = \
                        self.compute_next_step(iterno,forces1,old_forces1,old_search_vec1)
                # Update drude positions a distance lambda in the direction of the
                # search vector
                self.shell_xyz1 += lambda1*search_vec1

            # Save the current forces and search vectors for the next
            # iteration
            old_forces1 = np.copy(forces1)
            old_search_vec1 = np.copy(forces1)

            # Repeat above procedure for monomer 2

            print 'shell positions 2'
            print self.shell_xyz2[0] - self.xyz2[0]

            forces2 = np.zeros_like(self.xyz2)
            for i in xrange(self.natoms2):
                if np.abs(self.qshell2[i]) > self.small_q:
                    forces2[:,i,:] = self.qshell2[i]*self.get_efield(i,mon=2) 
            x1 = self.shell_xyz2
            x2 = self.xyz2
            dx = x1 - x2
            forces2 = forces2 - self.springcon*dx
            converged2 = np.all(np.abs(forces2) < thresh)

            if not converged2:
                lambda2, search_vec2 = \
                        self.compute_next_step(iterno,forces2,old_forces2,old_search_vec2)
                self.shell_xyz2 += lambda2*search_vec2

            old_forces2 = np.copy(forces2)
            old_search_vec2 = np.copy(forces2)

            iterno += 1
            converged = converged1 and converged2

        if self.verbose:
            print 'Drude oscillators converged in iterno ',iterno, ' with maximum forces ',\
                    np.amax(forces1), ' and ', np.amax(forces2), 'in any direction.'
        return self.shell_xyz1, self.shell_xyz2
####################################################################################################    


####################################################################################################    
    def compute_next_step(self,iterno,forces,old_forces,old_search_vec,
                            small_lambda=1e-30, small_f=1e-15):
        '''Compute the positions of drude oscillators for the next iteration of a conjugate gradient descent.

        Parameters
        ----------
        iterno : int
            Iteration number. Next step computed differently if iterno == 0.
        forces : ndarray
            Current forces on drude particles.
        old_forces : ndarray
            Forces on drude particles from previous iteration.
        old_search_vec : ndarray
            Search direction from previous iteration.
        small_lambda : float, optional.
            Cutoff for when to treat lambda values as being effectively zero.
            Necessary for avoiding divergent behavior.
        small_f : float, optional.
            Cutoff for when to treat forces as being effectively zero.

        Returns
        -------
        lambda : ndarray
            Array with same shape as self.xyz{1,2}[:] containing values
            corresponding to the magnitude of the step to take in the
            search_vec direction.
        search_vec : ndarray
            Array with the same shape as self.xyz{1,2}[:,0,:] containing the
            new search direction for each data point.

        '''
        # For first iteration, compute next step via steepest descent:
        if iterno == 0:
            search_vec = np.copy(forces)
            assert not np.may_share_memory(search_vec, forces)
            lambdai = 1/self.springcon
        # For subsequent steps, using a conjugate gradient method, as
        # described in 
        # Lindan, P. J. D.; Gillan, M. J. J. Phys Condens. Matter # 1993, 5, 1019
        # to determine the next step:
        else:
            sum_f_old = np.sum(old_forces*old_forces,axis=(-1,-2))
            sum_f_new = np.sum(forces*forces,axis=(-1,-2))
            beta = np.where(sum_f_old > small_f , sum_f_new/sum_f_old, 0)

            search_vec = forces + beta[:,np.newaxis,np.newaxis]*old_search_vec

            lambdai = np.sum(forces*search_vec,axis=(-1,-2))
            lambda_denom = self.springcon*np.sum(search_vec*search_vec,axis=(-1,-2))
            # Here we have to be careful to avoid zero division errors, if
            # lambda is too close to zero
            lambdai /= np.where(abs(lambdai) > small_lambda,
                            lambda_denom, np.inf ) 
            # Broadcast lambda into the correct shape
            lambdai = lambdai[:,np.newaxis,np.newaxis]


        return lambdai, search_vec
####################################################################################################    


####################################################################################################    
    def get_efield(self,ishell,mon=1):
        '''Compute the electrostatic field on a drude oscillator with index ishell.

        Electrostatic field contributions arise from the following sources:
            1. Intramolecular drude oscillator contributions
            2. Intermolecular permanent charges 
            3. Intermolecular drude oscillator contributions

            Note that intramolecular permanent charges do not contribute to
            this field, as they form part of the intramolecular energy that we
            are *not* describing.

        Parameters
        ----------
        ishell : integer
            The index corresponding the drude shell in question, whose
            position should be given by self.shell_xyz{1,2}[datpt,ishell] for
            a given input data point.
        mon : {1,2}, optional.
            Specifies which monomer the ishell'th drude is a part of.

        Returns
        -------
        efield : ndarray
            The electric field on the ishell'th drude for each data point,
            with the same shape as self.shell_xyz{1,2}[:,ishell,:].
            
        '''
        # To keep these formulas as general as possible, we'll write
        # intramolecular contributions as belonging to monomer 'i', and
        # intermolecular contributions as belonging to monomer 'j'.
        if mon == 1:
            natoms_i = self.natoms1
            natoms_j = self.natoms2

            xyz_i = self.xyz1
            xyz_j = self.xyz2
            shell_xyz_i = self.shell_xyz1
            shell_xyz_j = self.shell_xyz2

            qshell_i = self.qshell1
            qshell_j = self.qshell2
            
            qcore_i = self.charges1 - self.qshell1
            qcore_j = self.charges2 - self.qshell2

            exponents = self.exponents

        elif mon == 2:
            natoms_i = self.natoms2
            natoms_j = self.natoms1

            xyz_i = self.xyz2
            xyz_j = self.xyz1
            shell_xyz_i = self.shell_xyz2
            shell_xyz_j = self.shell_xyz1

            qshell_i = self.qshell2
            qshell_j = self.qshell1

            qcore_i = self.charges2 - self.qshell2
            qcore_j = self.charges1 - self.qshell1

            exponents = np.transpose(self.exponents)

        else:
            sys.exit('Must set mon == 1 or mon == 2.')


        # Start calculation of the electric field at the location of atom ishell
        # in monomer i.
        efield = np.zeros_like(xyz_i[:,ishell,:])

        # Keep track of the charge of the ishell'th drude
        q1 = qshell_i[ishell]

        # First, compute field due to intramolecular drude oscillators
        for i in xrange(natoms_i):
            if i == ishell: # Ignore self-interaction, zero-charge oscillators
                continue
            q2 = qshell_i[i]
            if abs(q2) < self.small_q:
                # If q2 == 0, we can ignore this drude pair
                continue

            # Shell-shell interactions
            x1 = shell_xyz_i[:,ishell]
            x2 = shell_xyz_i[:,i]
            xvec = x1 - x2
            efield += self.get_efield_from_thole_charge(q1,q2,xvec)

            # Shell-core interaction
            q2 *= -1 # Core charge negative of shell charge
            x2 = xyz_i[:,i]
            xvec = x1 - x2
            efield += self.get_efield_from_thole_charge(q1,q2,xvec)

        print 'Intramolecular efield'
        print efield[0]

        # Second, compute field due to intermolecular permanent charges and
        # drude oscillators:
        for j in xrange(natoms_j):
            ## # Shell-core interactions
            ## q2 = qcore_j[j]
            ## x1 = shell_xyz_i[:,ishell]
            ## x2 = xyz_j[:,j]
            ## xvec = x1 - x2
            ## bij = exponents[ishell,j]
            ## efield += self.get_efield_from_tangtoennies_charge(q2,bij,xvec)

            # Shell-multipole interactions
            q2 = qcore_j[j] + qshell_j[j]
            x1 = shell_xyz_i[:,ishell]
            x2 = xyz_j[:,j]
            xvec = x1 - x2
            bij = exponents[ishell,j]
            efield += self.get_efield_from_tangtoennies_charge(q2,bij,xvec)
            print 'shell-multipole'
            print self.get_efield_from_tangtoennies_charge(q2,bij,xvec)[0]

            # Shell-core interactions
            q2 = -qshell_j[j]
            x1 = shell_xyz_i[:,ishell]
            x2 = xyz_j[:,j]
            xvec = x1 - x2
            bij = exponents[ishell,j]
            efield += self.get_efield_from_tangtoennies_charge(q2,bij,xvec)
            print 'shell-core'
            print self.get_efield_from_tangtoennies_charge(q2,bij,xvec)[0]

            # Shell-shell interactions
            q2 = qshell_j[j]
            if abs(q2) < self.small_q: continue
            x2 = shell_xyz_j[:,j]
            xvec = x1 - x2
            efield += self.get_efield_from_tangtoennies_charge(q2,bij,xvec)
            print 'shell-shell'
            print self.get_efield_from_tangtoennies_charge(q2,bij,xvec)[0]

        print 'Total efield' 
        print efield[0]

        return efield
####################################################################################################    


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
        edrude_total = self.get_drude_energy()

        # Set each monomer's drude charges to zero and get drude energy in
        # order to get 2nd order induction energy
        self.qshell2 = np.zeros_like(self.qshell2)
        self.find_drude_positions()
        edrude_ind1 = self.get_drude_energy()

        self.qshell1 = np.zeros_like(self.qshell1)
        self.qshell2 = qshell2_save
        self.find_drude_positions()
        edrude_ind2 = self.get_drude_energy()

        edrude_ind = edrude_ind1 + edrude_ind2
        edrude_high_order = edrude_total - edrude_ind

        return edrude_ind, edrude_high_order
####################################################################################################    


####################################################################################################    
    def get_drude_energy(self,include_electrostatics=False):
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

        # Intramolecular drude energy from monomer 1
        for i,qi in enumerate(self.qshell1):
            if abs(qi) < self.small_q: continue
            for k,qj in enumerate(self.qshell1[i+1:]):
                j = k + i + 1
                if abs(qj) < self.small_q: continue

                # Shell-shell interactions
                xi = self.shell_xyz1[:,i,:]
                xj = self.shell_xyz1[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*qi*qj/rij

                # Core-shell interactions
                xi = self.shell_xyz1[:,i,:]
                xj = self.xyz1[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*qi*(-qj)/rij

                xi = self.xyz1[:,i,:]
                xj = self.shell_xyz1[:,j,:]
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
                xi = self.shell_xyz2[:,i,:]
                xj = self.shell_xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*qi*qj/rij

                # Core-shell interactions
                xi = self.shell_xyz2[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*qi*(-qj)/rij

                xi = self.xyz2[:,i,:]
                xj = self.shell_xyz2[:,j,:]
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
                xi = self.shell_xyz1[:,i,:]
                xj = self.shell_xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                if abs(qi) > self.small_q and abs(qj) > self.small_q:
                    edrude += \
                        self.damp_inter(bij,dx[:,0],dx[:,1],dx[:,2])*qi*qj/rij

                # Core-shell interactions (includes permanent charges on opposite monomer)
                xi = self.shell_xyz1[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                qcore_j = -qj + self.charges2[j]
                if abs(qi) > self.small_q and abs(qcore_j) > self.small_q:
                    edrude += \
                        self.damp_inter(bij,dx[:,0],dx[:,1],dx[:,2])*qi*(qcore_j)/rij

                xi = self.xyz1[:,i,:]
                xj = self.shell_xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                qcore_i = -qi + self.charges1[i]
                if abs(qcore_i) > self.small_q and abs(qj) > self.small_q:
                    edrude += \
                        self.damp_inter(bij,dx[:,0],dx[:,1],dx[:,2])*(qcore_i)*qj/rij

                # Core-core interactions
                xi = self.xyz1[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                qcore_i = -qi + self.charges1[i]
                qcore_j = -qj + self.charges2[j]
                if abs(qcore_i) > self.small_q and abs(qcore_j) > self.small_q:
                    edrude += \
                        self.damp_inter(bij,dx[:,0],dx[:,1],dx[:,2])*qcore_i*qcore_j/rij

                # Exclude energy from static charges, if not including
                # intermolecular electrostatic energy
                if not include_electrostatics:
                    qcore_i =  self.charges1[i]
                    qcore_j =  self.charges2[j]
                    if abs(qcore_i) > self.small_q and abs(qcore_j) > self.small_q:
                        edrude -= \
                            self.damp_inter(bij,dx[:,0],dx[:,1],dx[:,2])*qcore_i*qcore_j/rij


        # Include spring energy:
        # Spring Energy Monomer1
        dr2 = np.sum((self.xyz1 - self.shell_xyz1)**2, axis=-1)
        edrude += 0.5*np.sum(self.springcon*dr2, axis=-1)
        # Spring Energy Monomer 2
        dr2 = np.sum((self.xyz2 - self.shell_xyz2)**2, axis=-1)
        edrude += 0.5*np.sum(self.springcon*dr2, axis=-1)

        #print 'Edrude = ', edrude[0]
        return edrude
####################################################################################################    


####################################################################################################    
    def get_efield_from_tangtoennies_charge(self,q,bij,xvec): 
        '''Compute the electric field due to a screened point charge q located at xvec.

        More specifically, here it is assumed that the potential from the
        point charge is given by
             V = f_damp*q/r
        where f_damp is given by the standard Tang-Toennies damping function,
        and r is the norm of xvec.
        yielding an electric field
             E = - del V
               = - ( del(f_damp)*q/r + f_damp*q/r^3*xvec )
        Note that, in the limit where q or r are sufficiently small, E = 0.

        Parameters
        ----------
        q : float
            Magnitude of the point charge
        bij : float
            Screening parameter; should correspond to the exponent used in the
            repulsive potential (described in more detail elsewhere).
        xvec: ndarray
            Array containing the relative position of q for each data point
            with respect to a point x1 where we are calculating the electric
            field.

        Returns
        -------
        efield : ndarray
            Array with the same shape as xvec that contains the electric field
            due to a Tang-Toennies screened point charge q.
        
        '''
        xij = xvec[:,0]
        yij = xvec[:,1]
        zij = xvec[:,2]
        r = np.sqrt(np.sum(xvec**2,axis=1))

        damp = np.where( r > self.small_r, self.damp_inter(bij, xij, yij, zij)/r**3, 0)
        efield = damp[:,np.newaxis]*q*xvec

        ddamp = np.array([ np.where( r > self.small_r, \
                                         - dcharge(bij,xij,yij,zij)*q/r,\
                                         0) for dcharge in self.del_damp_inter ])
        efield += np.swapaxes(ddamp,0,1)

        return efield
####################################################################################################    


####################################################################################################    
    def get_efield_from_thole_charge(self,q1,q2,xvec): 
        '''Compute the electric field due to a screened point charge q2 located at xvec.

        More specifically, here it is assumed that the potential from the
        point charge is given by
             V = f_damp*q/r
        where f_damp is given by the standard Thole damping function,
        and r is the norm of xvec.
        yielding an electric field
             E = - del V
               = - ( del(f_damp)*q/r + f_damp*q/r^3*xvec )
        Note that, in the limit where q or r are sufficiently small, E = 0.

        Parameters
        ----------
        q1 : float
            Magnitude of the point charge where the electric field is being
            calculated; only needed to calculate the damping function.
        q2 : float
            Magnitude of the point charge located at a position xvec away from
            q1.
        xvec: ndarray
            Array containing the relative position of q2 for each data point
            with respect to a point x1 where we are calculating the electric
            field.

        Returns
        -------
        efield : ndarray
            Array with the same shape as xvec that contains the electric field
            due to a Thole-screened point charge q2.
        
        '''
        xij = xvec[:,0]
        yij = xvec[:,1]
        zij = xvec[:,2]
        r = np.sqrt(np.sum(xvec**2,axis=1))
        damp = np.where( r > self.small_r, self.damp_intra(q1, q2, xij, yij, zij)/r**3, 0)
        efield = damp[:,np.newaxis]*q2*xvec

        ddamp = np.array([ np.where( r > self.small_r, \
                                         - ddrude(q1,q2,xij,yij,zij)*q2/r,\
                                         0) for ddrude in self.del_damp_intra ])
        efield += np.swapaxes(ddamp,0,1)

        return efield
####################################################################################################    


####################################################################################################    
    def get_thole_damping_factor(self,qi,qj,xij,yij,zij):
        '''Compute the Thole damping factor.

        References:
        (1) Yu, K.; McDaniel, J. G.; Schmidt, J. R. J. Phys. Chem. B 2011, 115, 10054-10063.

        Parameters
        ----------
        qi : Symbol
            Charge of the first drude oscillator.
        qj : Symbol
            Charge of the second drude oscillator.
        xij : Symbol
            Cartesian distance between oscillators in the x-direction.
        yij : Symbol
            Cartesian distance between oscillators in the y-direction.
        zij : Symbol
            Cartesian distance between oscillators in the z-direction.

        Returns
        -------
        damping_factor : Sympy expression

        '''
        rij = sp.sqrt(xij**2 + yij**2 + zij**2)

        ai = qi**2/self.springcon
        aj = qj**2/self.springcon
        p = self.screenlength

        prefactor = 1.0 + p*rij/(2*(ai*aj)**(1.0/6))
        exponent = p*rij/(ai*aj)**(1.0/6)

        damping_factor = 1.0 - sp.exp(-exponent)*prefactor

        return damping_factor
####################################################################################################    


####################################################################################################    
    def get_tt_damping_factor(self, bij, xij, yij, zij):
        '''Compute the Tang-Toennies damping factor.

        This damping factor depends (see ref. 3) on the form of the repulsive part
        of the potential, with x=y*r and y = -d/dr(ln V_repulsive).

        References:
        (1) McDaniel, J. G.; Schmidt, J. R. J. Phys. Chem. A 2013, 117, 2053-066.
        (2) Tang, K. T.; Toennies, J. P. J. Chem. Phys. 1984, 80, 3726-3741.
        (3) Tang, K. T.; Peter Toennies, J. Surf. Sci. 1992, 279, L203-206.

        Parameters
        ----------
        bij : Symbol
            Exponent between the two oscillators.
        xij : Symbol
            Cartesian distance between oscillators in the x-direction.
        yij : Symbol
            Cartesian distance between oscillators in the y-direction.
        zij : Symbol
            Cartesian distance between oscillators in the z-direction.

        Returns
        -------
        damping_factor : Sympy expression

        '''
        rij = sp.sqrt(xij**2 + yij**2 + zij**2)

        if self.slater_correction:
            y = bij - (2*bij**2*rij + 3*bij)/(bij**2*rij**2 + 3*bij*rij + 3)
        else:
            y = bij

        return 1.0 - sp.exp(-y*rij)*(1 + y*rij)
####################################################################################################    


####################################################################################################    
    def get_random_unit_vec(self,ivec):
        '''Get a random unit vector of the same shape as the input vector.
        
        Parameters
        ----------
        ivec : 3darray
            Input vector, assumed to have length 3 in the last dimension so as
            to correspond to 3d cartesian space.

        Returns
        -------
        random_vec : 3darray
            Randomly generated unit vector of the same shape as ivec.
        '''

        assert ivec.ndim == 3
        assert ivec.shape[-1] == 3
        
        # Normal distribution of points yields uniform distribution of
        # directions:
        # http://stackoverflow.com/questions/9750908/how-to-generate-a-unit-vector-pointing-in-a-random-direction-with-isotropic-dist
        random_vec = np.random.normal(size=ivec.shape)
        random_vec /= np.linalg.norm(random_vec,axis=-1)[:,:,np.newaxis]

        return random_vec
####################################################################################################    


