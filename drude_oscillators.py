# Standard Packages
import numpy as np
import sys
import os
import sympy as sp
from sympy.utilities import lambdify

# Local Packages
from multipoles import Multipoles

# Numpy error message settings
np.seterr(under='ignore')

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

    Known Issues
    ------------
    1. Multipole derivatives have only been tested assuming the drude
    oscillator is a point charge; if drude oscillators are ever treated as
    a higher-order multipole, these derivatives may need to be re-evaluated.

    Units
    -----
    Atomic units are assumed throughout this module.

    '''
    
    def __init__(self, 
                   mon1, mon2, 
                   xyz1, xyz2, 
                   multipole_file1, multipole_file2, 
                   axes1,axes2,
                   qshell1, qshell2, 
                   springcon1,springcon2,
                   exponents,
                   rigid_monomers,
                   screenlength1=2.0, screenlength2=2.0, 
                   slater_correction=True,
                   intra_damping_type='thole_tinker',
                   inter_damping_type='None',
                   damp_charges_only=True
                   ):

        '''Initilialize input variables and drude positions.'''

        ###########################################################################
        ################ Program-Defined Class Variables ##########################

        # Initialize drude positions slightly off each core center if set to
        # True. Normally has little effect on convergence, but may matter in
        # some cases.
        self.initialize_off_center = True
        #self.initialize_off_center = False

        # Provide cutoffs for when to treat charges and distances as
        # effectively zero:
        self.small_q = 1e-7
        self.small_r = 1e-7

        # Verbosity settings:
        self.verbose = True

        # Scale the electric fields (those induced by multipole moments) by
        # this amount (primarily for debugging purposes):
        self.multipole_efield_scale_factor = 1.00

        # Filename to store damping functions
        self.fpik = 'drude_oscillators.pik'
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path) + '/'
        self.fpik = dir_path + self.fpik

        # Use average isotropic Thole damping factors for now based on average
        # spring constant; this is what OpenMM does, but one could imagine
        # wanting to change this in the future.
        self.avg_springcon = 0.1

        ###########################################################################
        ###########################################################################


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
        self.qshell1 = qshell1
        self.qshell2 = qshell2
        self.rigid_monomers = rigid_monomers
        self.springcon1 = springcon1
        self.springcon2 = springcon2
        self.exponents = exponents
        if type(screenlength1) == np.float:
            self.screenlength1 = screenlength1*np.ones_like(self.qshell1)
        else:
            self.screenlength1 = screenlength1
        if type(screenlength2) == np.float:
            self.screenlength2 = screenlength2*np.ones_like(self.qshell2)
        else:
            self.screenlength2 = screenlength2
        self.slater_correction = slater_correction

        self.natoms1 = len(self.qshell1)
        self.natoms2 = len(self.qshell2)

        # Transform spring constants to the global coordinate system
        self.springcon1 = self.axes1*self.springcon1[np.newaxis,:,np.newaxis,:]
        self.springcon1 = np.sqrt(np.sum(self.springcon1**2,-1))
        self.springcon2 = self.axes2*self.springcon2[np.newaxis,:,np.newaxis,:]
        self.springcon2 = np.sqrt(np.sum(self.springcon2**2,-1))

        # Compute polarizabilities from shell charges
        self.polarizability1 = self.qshell1**2/self.avg_springcon
        self.polarizability2 = self.qshell2**2/self.avg_springcon

        # Set non-polarizable atoms to have a default polarizability (this
        # allows for computation of intermolecular Thole damping factors where
        # necessary)
        default_pol = 1.0
        self.polarizability1[self.polarizability1 == 0] = default_pol
        self.polarizability2[self.polarizability2 == 0] = default_pol

        self.intra_damping_type = intra_damping_type
        self.inter_damping_type = inter_damping_type
        self.damp_charges_only = damp_charges_only
        if self.inter_damping_type == 'Tang-Toennies':
            if self.exponents.shape[-2:] != (1,1):
                raise NotImplementedError,\
                        '''The mathematical form of the Tang-Toennies damping differs if
                        the repulsive potential is comprised of multiple
                        exponents (see Tang, K. T.; Toennies, J. P.  Surf.
                        Sci. 1992, 279, L203-L206 for details), and this
                        (more complicated) functional form has not yet
                        been included in this fitting program.'''
        # For now, only use first (and due to check above, only) exponent
        # for each atomtype
        self.exponents = np.squeeze(self.exponents,axis=(-2,-1))

        ###########################################################################
        ###########################################################################

        ###########################################################################
        ###################### Initialization Routines ############################

        # Initialize drude oscillator positions
        self.initialize_shell_positions()

        # Initialize multipole moment class
        self.update_multipole_moments(init=True)

        # Initialize damping functions, either by reading functions in from
        # file or generating them on the fly.
        self.get_damping_functions()


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
    def update_multipole_moments(self,init=False):
        '''Update energies due to shell positions due to electrostatic
        interactions.

        Parameters
        ----------
        init : bool, optional.
            If first call to this function, set to True (which initializes
            several variables and only needs to be done once). Default is
            False.

        Returns
        -------
        Nothing, though self.Mon[1,2]Multipoles classes are updated to reflect
        new shell positions.

        '''
        if init:
            # M1 class instance for the interaction of mon2 drudes with mon1
            # multipoles
            m1 = Multipoles(self.mon1, self.mon2,
                            self.xyz1,self.shell_xyz2, 
                            self.multipole_file1, self.multipole_file2,
                            self.axes1, self.axes2,
                            self.rigid_monomers,
                            self.exponents, self.slater_correction)
            ## m1.atoms1, m1.multipoles1, m1.local_coords1 = m1.read_multipoles(self.multipole_file1)
            ## m1.ea = m1.get_local_to_global_rotation_matrix(m1.xyz1,m1.local_coords1)
            # Get multipoles1 and ea
            m1.get_local_axis_parameters(mon=1)
            # Here we're assuming that the drude charges are simple point charges;
            # thus is doesn't matter what we consider the local coordinate system
            # for these shell charges
            m1.multipoles2 = [ {'Q00' : q } for q in self.qshell2 ]
            m1.eb = np.array([ np.identity(3) for xyz in m1.xyz2])
            m1.eb = np.tile(m1.eb[:,np.newaxis,...],(1,self.natoms2,1,1))

            # M2 class instance for the interaction of mon1 drudes with mon2
            # multipoles
            m2 = Multipoles(self.mon2, self.mon1,
                            self.xyz2,self.shell_xyz1, 
                            self.multipole_file2, self.multipole_file1,
                            self.axes2, self.axes1,
                            self.rigid_monomers,
                            self.exponents, self.slater_correction)
            # TODO: Should exponents be reversed?
            ## m2.atoms2, m2.multipoles1, m2.local_coords1 = m2.read_multipoles(self.multipole_file2)
            ## m2.ea = m2.get_local_to_global_rotation_matrix(m2.xyz1,m2.local_coords1)
            m2.get_local_axis_parameters(mon=1)
            m2.multipoles2 = [ {'Q00' : q } for q in self.qshell1 ]
            m2.eb = np.array([ np.identity(3) for xyz in m2.xyz1])
            m2.eb = np.tile(m2.eb[:,np.newaxis,...],(1,self.natoms1,1,1))

            self.Mon1Multipoles = m1
            self.Mon2Multipoles = m2
            self.Mon1Multipoles.update_direction_vectors(init=True)
            self.Mon2Multipoles.update_direction_vectors(init=True)

        else:
            # Update shell positions and vectors
            self.Mon1Multipoles.xyz2 = self.shell_xyz2
            self.Mon1Multipoles.update_direction_vectors()
            self.Mon2Multipoles.xyz2 = self.shell_xyz1
            self.Mon2Multipoles.update_direction_vectors()


        return
####################################################################################################    


####################################################################################################    
    def get_damping_functions(self):
        '''Creates numerical functions and derivatives for the various damping
        functions that can be utilized in this module, and (if dill is
        available) stores them for later reference.

        Parameters
        ----------
        None

        Returns
        -------
        None, though class variables for both intra- and inter-molecular
        damping functions are set up.

        '''
        # First, try and unpack serialized damping functions
        try:
            #import dill
            import dill 
            import pickle
            dill.settings['recurse'] = True
            print 'loaded dill functionality'
            with open(self.fpik,'rb') as f:
                slater_tt_damp_inter = dill.load(f)
                slater_tt_del_damp_inter = dill.load(f)
                noslater_tt_damp_inter = dill.load(f)
                noslater_tt_del_damp_inter = dill.load(f)
                no_damp_inter = dill.load(f)
                no_del_damp_inter = dill.load(f)
                linear_damp_intra = dill.load(f)
                linear_del_damp_intra = dill.load(f)
                tinker_damp_intra = dill.load(f)
                tinker_del_damp_intra = dill.load(f)

        # If dill module not available, or data not previously
        # serialized, recreate damping functions.
        except (ImportError,IOError,EOFError):
            # Create numerical subroutines to compute gradients for the Thole
            # and Tang-Toennies damping functions. Note that, for all
            # intramolecular contacts, Thole screening will be used, while all
            # intermolecular contacts will be damped via Tang-Toennies screening.
            print 'Creating numerical subroutines for damping functions.'
            bij, ai, aj, p, xij, yij, zij = sp.symbols("bij ai aj p xij yij zij")
            noslater_tt_damp_inter = lambdify((bij,xij,yij,zij),\
                                    self.get_tt_damping_factor(bij,xij,yij,zij,
                                        slater_correction=False), modules='numexpr')
            noslater_tt_del_damp_inter = [ lambdify((bij,xij,yij,zij),\
                                         sp.diff(self.get_tt_damping_factor(bij,xij,yij,zij,
                                             slater_correction=False),x),\
                                         modules='numexpr') \
                                    for x in [xij,yij,zij] ]
            slater_tt_damp_inter = lambdify((bij,xij,yij,zij),\
                                    self.get_tt_damping_factor(bij,xij,yij,zij,
                                        slater_correction=True), modules='numexpr')
            slater_tt_del_damp_inter = [ lambdify((bij,xij,yij,zij),\
                                         sp.diff(self.get_tt_damping_factor(bij,xij,yij,zij,
                                             slater_correction=True),x),\
                                         modules='numexpr') \
                                    for x in [xij,yij,zij] ]
            no_damp_inter = lambda bij, xij, yij, zij : 1
            no_del_damp_inter = [ lambda bij, xij, yij, zij : 0
                                    for x in [xij,yij,zij] ]

            thole_args = (ai, aj, p, xij, yij, zij)

            thole_style = ('linear',)
            linear_damp_intra = lambdify(thole_args,\
                                   self.get_thole_damping_factor(*(thole_args+thole_style)), modules='numexpr')
            diff_damp_intra = [ sp.diff(self.get_thole_damping_factor(*thole_args+ thole_style),x)
                                            for x in [xij,yij,zij] ]
            linear_del_damp_intra = [ lambdify(thole_args, ddamp, modules='numexpr')
                                               for ddamp in diff_damp_intra ]

            thole_style = ('tinker',)
            tinker_damp_intra = lambdify(thole_args,\
                                   self.get_thole_damping_factor(*(thole_args+thole_style)), modules='numexpr')
            diff_damp_intra = [ sp.diff(self.get_thole_damping_factor(*thole_args+thole_style),x)
                                            for x in [xij,yij,zij] ]

            tinker_del_damp_intra = [ lambdify(thole_args, ddamp, modules='numexpr')
                                               for ddamp in diff_damp_intra ]

            try:
                #import dill
                import dill
                import pickle
                dill.settings['recurse'] = True
            except ImportError:
                pass
            else:
                with open(self.fpik,'wb') as f:
                    print 'Saving numerical subroutines for damping functions to file:'
                    print self.fpik

                    dill.dump(noslater_tt_damp_inter, f)
                    dill.dump(noslater_tt_del_damp_inter, f)
                    dill.dump(slater_tt_damp_inter, f)
                    dill.dump(slater_tt_del_damp_inter, f)
                    dill.dump(no_damp_inter, f)
                    dill.dump(no_del_damp_inter, f)
                    dill.dump(linear_damp_intra, f)
                    dill.dump(linear_del_damp_intra, f)
                    dill.dump(tinker_damp_intra, f)
                    dill.dump(tinker_del_damp_intra, f)


        # Set appropriate damping functions as class variables
        if self.intra_damping_type.lower() == 'thole_linear':
            print 'Thole damping type set to: linear'
            self.damp_intra = linear_damp_intra
            self.del_damp_intra = linear_del_damp_intra
        elif self.intra_damping_type.lower() == 'thole_tinker':
            print 'Thole damping type set to: Tinker-style'
            self.damp_intra = tinker_damp_intra
            self.del_damp_intra = tinker_del_damp_intra
        else:
            sys.exit('Unknown Intramolecular Damping Type ' + self.intra_damping_type)

        if self.inter_damping_type == 'Tang-Toennies':
            if self.slater_correction:
                self.damp_inter = slater_tt_damp_inter
                self.del_damp_inter = slater_tt_del_damp_inter
            else:
                self.damp_inter = noslater_tt_damp_inter
                self.del_damp_inter = noslater_tt_del_damp_inter
        elif self.inter_damping_type == 'None':
            self.damp_inter = no_damp_inter
            self.del_damp_inter = no_del_damp_inter
        elif self.inter_damping_type == 'Thole':
            self.damp_inter = self.damp_intra
            self.del_damp_inter = self.del_damp_intra
        else:
            sys.exit('Unknown Intermolecular Damping Type ' + self.inter_damping_type)

        return
####################################################################################################    


####################################################################################################    
    def combine_thole_damping_parameter(self,
            p1,p2,combination_rule='arithmetic_mean'):
        '''Use a combination rule to determine the Thole screening length p for
        an atom pair based on p1 and p2, the Thole screening lengths of the individual
        atoms.

        Parameters
        ----------
        p1 : float
            Thole screening length for atom 1
        p2 : float
            Thole screening length for atom 2

        combination_rule : string, optional.
            Combination rule for Thole damping. Currently accepted values are
            arithmetic_mean and geometric_mean

        Returns
        -------
        p : float
            Thole screening length for the atom pair

        '''
        if combination_rule == 'arithmetic_mean':
            p = 0.5*(p1 + p2)
        elif combination_rule == 'geometric_mean':
            p = np.sqrt(p1*p2)
        else:
            raise InputError,\
                combination_rule + ' is not a valid combination rule for the Thole screening parameter.'

        return p
####################################################################################################    


####################################################################################################    
    def find_drude_positions(self,itermax=500,thresh=1e-5):
    #def find_drude_positions(self,itermax=100,thresh=1e-5):
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

        print 'Converging drude oscillator positions:',

        while not converged:
            #sys.stdout.write('.')
            sys.stdout.flush()
            if iterno > itermax:
                error = '''Too many iterations to find drude oscillator positions!
                
                Maximum force on drude oscillators is {} on monomer 1 and {}
                on monomer 2. 

                Exiting.
                '''
                sys.exit(error.format(np.max(np.abs(forces1)), np.max(np.abs(forces2))))

            self.update_multipole_moments()

            forces1 = np.zeros_like(self.xyz1)
            for i in xrange(self.natoms1):
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
            forces1 = forces1 - self.springcon1*dx
            #converged1 = np.all(np.abs(forces1) < thresh)
            converged1 = np.mean(np.abs(forces1)) < thresh

            if not converged1:
                lambda1, search_vec1 = \
                        self.compute_next_step(iterno,self.springcon1,forces1,old_forces1,old_search_vec1)
                # Update drude positions a distance lambda in the direction of the
                # search vector
                self.shell_xyz1 += lambda1*search_vec1

            # Save the current forces and search vectors for the next
            # iteration
            old_forces1 = np.copy(forces1)
            old_search_vec1 = np.copy(forces1)

            # Repeat above procedure for monomer 2
            forces2 = np.zeros_like(self.xyz2)
            for i in xrange(self.natoms2):
                if np.abs(self.qshell2[i]) > self.small_q:
                    forces2[:,i,:] = self.qshell2[i]*self.get_efield(i,mon=2) 
            x1 = self.shell_xyz2
            x2 = self.xyz2
            dx = x1 - x2
            forces2 = forces2 - self.springcon2*dx
            #converged2 = np.all(np.abs(forces2) < thresh)
            converged2 = np.mean(np.abs(forces2)) < thresh

            if not converged2:
                lambda2, search_vec2 = \
                        self.compute_next_step(iterno,self.springcon2,forces2,old_forces2,old_search_vec2)

                self.shell_xyz2 += lambda2*search_vec2

            old_forces2 = np.copy(forces2)
            old_search_vec2 = np.copy(forces2)

            iterno += 1
            converged = converged1 and converged2

            dxyz1 = np.sqrt(np.sum((self.shell_xyz1 - self.xyz1)**2,axis=2))
            dxyz2 = np.sqrt(np.sum((self.shell_xyz2 - self.xyz2)**2,axis=2))
            ## sys.stdout.write(str((dxyz1 > 2).nonzero()))
            ## sys.stdout.write(str((dxyz2 > 2).nonzero()))
            bad_convergence_points = set( list((dxyz1 > 2).nonzero()[0]) + list((dxyz2 > 2).nonzero()[0]) )
            #bad_convergence_points = set( (dxyz1 > 2).nonzero()[0] + (dxyz2 > 2).nonzero()[0] )
            if bad_convergence_points:
                sys.stdout.write('The following data points drude oscillators with large displacements: ')
                sys.stdout.write(str(bad_convergence_points))
            # sys.stdout.write(str(bad_convergence_points))
            sys.stdout.write('\n')

            template = '{:10s}\t{:16s}\t{:16s}\n'
            sys.stdout.write(template.format('Iteration','|Max Drude Force|','|Mean Drude Force|'))

            template = '{:10d}\t{:16.8e}\t{:16.8e}\n'
            max_force1 = np.max(np.abs(forces1))
            max_force2 = np.max(np.abs(forces2))
            max_force = np.amax([max_force1,max_force2])
            mean_force1 = np.mean(np.abs(forces1))
            mean_force2 = np.mean(np.abs(forces2))
            mean_force = np.mean([mean_force1,mean_force2])
            sys.stdout.write(template.format(iterno,max_force,mean_force))
            sys.stdout.flush()

        sys.stdout.write('\n')
        if self.verbose:
            print 'Drude oscillators converged in iterno ',iterno, ' with maximum forces ',\
                    np.amax(forces1), ' and ', np.amax(forces2), 'in any direction.'
        return self.shell_xyz1, self.shell_xyz2
####################################################################################################    


####################################################################################################    
    def compute_next_step(self,iterno,springcon,forces,old_forces,old_search_vec,
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
            lambdai = 1/springcon

        # For subsequent steps, using a conjugate gradient method, as
        # described in 
        # Lindan, P. J. D.; Gillan, M. J. J. Phys Condens. Matter 1993, 5, 1019
        # to determine the next step:
        else:
            sum_f_old = np.sum(old_forces*old_forces,axis=(-1,-2))
            sum_f_new = np.sum(forces*forces,axis=(-1,-2))
            beta = np.where(sum_f_old > small_f , sum_f_new/sum_f_old, 0)

            search_vec = forces + beta[:,np.newaxis,np.newaxis]*old_search_vec

            lambdai = np.sum(forces*search_vec,axis=(-1,-2))
            lambda_denom = np.sum(springcon*search_vec*search_vec,axis=(-1,-2))
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

            ai = self.polarizability1
            aj = self.polarizability2

            pi = self.screenlength1
            pj = self.screenlength2
            
            exponents = self.exponents

            Multipoles_i = self.Mon1Multipoles
            Multipoles_j = self.Mon2Multipoles


        elif mon == 2:

            natoms_i = self.natoms2
            natoms_j = self.natoms1

            xyz_i = self.xyz2
            xyz_j = self.xyz1
            shell_xyz_i = self.shell_xyz2
            shell_xyz_j = self.shell_xyz1

            qshell_i = self.qshell2
            qshell_j = self.qshell1

            ai = self.polarizability2
            aj = self.polarizability1

            pi = self.screenlength2
            pj = self.screenlength1

            exponents = np.transpose(self.exponents)

            Multipoles_i = self.Mon2Multipoles
            Multipoles_j = self.Mon1Multipoles

        else:
            sys.exit('Must set mon == 1 or mon == 2.')


        # Start calculation of the electric field at the location of atom ishell
        # in monomer i.
        efield = np.zeros_like(xyz_i[:,ishell,:])

        # Keep track of the charge of the ishell'th drude
        q1 = qshell_i[ishell]
        a1 = ai[ishell]
        p1 = pi[ishell]

        # First, compute field due to intramolecular drude oscillators
        for i in xrange(natoms_i):
            if i == ishell: # Ignore self-interaction, zero-charge oscillators
                continue
            q2 = qshell_i[i]
            a2 = ai[i]
            p2 = pi[i]
            p = self.combine_thole_damping_parameter(p1,p2)
            if abs(q2) < self.small_q:
                # If q2 == 0, we can ignore this drude pair
                continue

            # Shell-shell interactions
            x1 = shell_xyz_i[:,ishell]
            x2 = shell_xyz_i[:,i]
            xvec = x1 - x2
            efield += self.get_efield_from_thole_charge(q1,q2,xvec,a1,a2,p)

            # Shell-core interaction
            q2 *= -1 # Core charge negative of shell charge
            x2 = xyz_i[:,i]
            xvec = x1 - x2
            efield += self.get_efield_from_thole_charge(q1,q2,xvec,a1,a2,p)

        # Second, compute field due to intermolecular permanent charges and
        # drude oscillators:
        multipole_efield = np.zeros_like(efield)
        for j in xrange(natoms_j):

            # Get parameters for damping functions
            a1 = ai[ishell]
            a2 = aj[j]
            p1 = pi[ishell]
            p2 = pj[j]
            p = self.combine_thole_damping_parameter(p1,p2)
            bij = exponents[ishell,j]

            # Compute shell-permanent multipole interactions
            x1 = shell_xyz_i[:,ishell]
            x2 = xyz_j[:,j]
            xvec = x1 - x2
            # TODO: Fix TT damping here
            efield += self.get_efield_from_multipole_charge(j,ishell,
                    Multipoles_j,xvec,bij,a1,a2,p)

            # Compute shell-core interactions
            q2 = - qshell_j[j]
            x1 = shell_xyz_i[:,ishell]
            x2 = xyz_j[:,j]
            xvec = x1 - x2
            efield += self.get_efield_from_point_charge(q2,xvec,bij,a1,a2,p)

            # Compute shell-shell interactions
            q2 = qshell_j[j]
            if abs(q2) < self.small_q: continue
            x2 = shell_xyz_j[:,j]
            xvec = x1 - x2
            efield += self.get_efield_from_point_charge(q2,xvec,bij,a1,a2,p)

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
    def get_drude_energy(self):
        '''Compute the potential energy arising from drude oscillators.

            Note that intramolecular permanent charges do not contribute to
            this energy, as they form part of the intramolecular energy that we
            are *not* describing.

        Parameters
        ----------
        None

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

                # Thole damping parameters
                ai = self.polarizability1[i]
                aj = self.polarizability1[j]
                pi = self.screenlength1[i]
                pj = self.screenlength1[j]
                p = self.combine_thole_damping_parameter(pi,pj)


                # Shell-shell interactions
                xi = self.shell_xyz1[:,i,:]
                xj = self.shell_xyz1[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(ai,aj,p,dx[:,0],dx[:,1],dx[:,2])*qi*qj/rij

                # Core-shell interactions
                xi = self.shell_xyz1[:,i,:]
                xj = self.xyz1[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(ai,aj,p,dx[:,0],dx[:,1],dx[:,2])*qi*(-qj)/rij

                xi = self.xyz1[:,i,:]
                xj = self.shell_xyz1[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(ai,aj,p,dx[:,0],dx[:,1],dx[:,2])*(-qi)*qj/rij

                # Core-core interactions
                xi = self.xyz1[:,i,:]
                xj = self.xyz1[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(ai,aj,p,dx[:,0],dx[:,1],dx[:,2])*(-qi)*(-qj)/rij

        # Intramolecular drude energy from monomer 2
        for i,qi in enumerate(self.qshell2):
            if abs(qi) < self.small_q: continue
            for k,qj in enumerate(self.qshell2[i+1:]):
                j = k + i + 1
                if abs(qj) < self.small_q: continue

                # Thole damping parameters
                ai = self.polarizability2[i]
                aj = self.polarizability2[j]
                pi = self.screenlength2[i]
                pj = self.screenlength2[j]
                p = self.combine_thole_damping_parameter(pi,pj)

                # Shell-shell interactions
                xi = self.shell_xyz2[:,i,:]
                xj = self.shell_xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(ai,aj,p,dx[:,0],dx[:,1],dx[:,2])*qi*qj/rij

                # Core-shell interactions
                xi = self.shell_xyz2[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                # edrude += self.damp_intra(qi,qj,dx[:,0],dx[:,1],dx[:,2])*qi*(-qj)/rij
                edrude += self.damp_intra(ai,aj,p,dx[:,0],dx[:,1],dx[:,2])*qi*(-qj)/rij

                xi = self.xyz2[:,i,:]
                xj = self.shell_xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(ai,aj,p,dx[:,0],dx[:,1],dx[:,2])*(-qi)*qj/rij

                # Core-core interactions
                xi = self.xyz2[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                edrude += self.damp_intra(ai,aj,p,dx[:,0],dx[:,1],dx[:,2])*(-qi)*(-qj)/rij

        # Intermolecular drude energy between monomers 1 and 2
        for i,qi in enumerate(self.qshell1):
            for j,qj in enumerate(self.qshell2):

                # print 'shell charges:',i,j,qi,qj

                # Damping parameters
                bij = self.exponents[i,j]
                ai = self.polarizability1[i]
                aj = self.polarizability2[j]
                pi = self.screenlength1[i]
                pj = self.screenlength2[j]
                p = self.combine_thole_damping_parameter(pi,pj)

                # Shell-shell interactions
                xi = self.shell_xyz1[:,i,:]
                xj = self.shell_xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                if abs(qi) > self.small_q and abs(qj) > self.small_q:
                    if self.inter_damping_type == 'Thole':
                        args = (ai,aj,p,dx[:,0],dx[:,1],dx[:,2])
                    else:
                        args = (bij,dx[:,0],dx[:,1],dx[:,2])
                    edrude += self.damp_inter(*args)*qi*qj/rij

                # Core-shell interactions (does not include permanent charges on opposite monomer)
                xi = self.shell_xyz1[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                qcore_j = -qj 
                if abs(qi) > self.small_q and abs(qcore_j) > self.small_q:
                    if self.inter_damping_type == 'Thole':
                        args = (ai,aj,p,dx[:,0],dx[:,1],dx[:,2])
                    else:
                        args = (bij,dx[:,0],dx[:,1],dx[:,2])
                    edrude += self.damp_inter(*args)*qi*(qcore_j)/rij

                xi = self.xyz1[:,i,:]
                xj = self.shell_xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                qcore_i = -qi 
                if abs(qcore_i) > self.small_q and abs(qj) > self.small_q:
                    if self.inter_damping_type == 'Thole':
                        args = (ai,aj,p,dx[:,0],dx[:,1],dx[:,2])
                    else:
                        args = (bij,dx[:,0],dx[:,1],dx[:,2])
                    edrude += self.damp_inter(*args)*(qcore_i)*qj/rij

                # Core-core interactions
                xi = self.xyz1[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                rij = np.sqrt(np.sum((xi-xj)**2,axis=1))
                qcore_i = -qi #+ self.charges1[i]
                qcore_j = -qj #+ self.charges2[j]
                if abs(qcore_i) > self.small_q and abs(qcore_j) > self.small_q:
                    if self.inter_damping_type == 'Thole':
                        args = (ai,aj,p,dx[:,0],dx[:,1],dx[:,2])
                    else:
                        args = (bij,dx[:,0],dx[:,1],dx[:,2])
                    edrude += self.damp_inter(*args)*qcore_i*qcore_j/rij

                # Shell - permanent multipole interactions
                # Mon1 multipoles with Mon2 drude shells
                self.Mon1Multipoles.xyz2 = self.shell_xyz2 # Update shell positions in case these have changed
                self.Mon1Multipoles.update_direction_vectors()
                if self.inter_damping_type == 'Thole':
                    args = (ai,aj,p,dx[:,0],dx[:,1],dx[:,2])
                else:
                    args = (bij,dx[:,0],dx[:,1],dx[:,2])
                xi = self.xyz1[:,i,:]
                xj = self.shell_xyz2[:,j,:]
                for mi in self.Mon1Multipoles.multipoles1[i].keys():
                    int_type = (mi,'Q00')
                    damp = self.damp_inter(*args)
                    edrude += damp*self.Mon1Multipoles.get_multipole_energy(i,j,int_type)

                # Mon2 multipoles with Mon1 drude shells
                self.Mon2Multipoles.xyz2 = self.shell_xyz1 # Update shell positions in case these have changed
                self.Mon2Multipoles.update_direction_vectors()
                xi = self.shell_xyz1[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                if self.inter_damping_type == 'Thole':
                    args = (ai,aj,p,dx[:,0],dx[:,1],dx[:,2])
                else:
                    args = (bij,dx[:,0],dx[:,1],dx[:,2])
                for mj in self.Mon2Multipoles.multipoles1[j].keys():
                    int_type = (mj,'Q00')
                    damp = self.damp_inter(*args)
                    edrude += damp*self.Mon2Multipoles.get_multipole_energy(j,i,int_type)

                # Core - permanent multipole interactions
                # Mon1 multipoles with Mon2 drude core
                self.Mon1Multipoles.xyz2 = self.xyz2 # Update shell positions in case these have changed
                self.Mon1Multipoles.update_direction_vectors()
                xi = self.xyz1[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                if self.inter_damping_type == 'Thole':
                    args = (ai,aj,p,dx[:,0],dx[:,1],dx[:,2])
                else:
                    args = (bij,dx[:,0],dx[:,1],dx[:,2])
                for mi in self.Mon1Multipoles.multipoles1[i].keys():
                    int_type = (mi,'Q00')
                    damp = self.damp_inter(*args)
                    edrude -= damp*self.Mon1Multipoles.get_multipole_energy(i,j,int_type)
                    # Minus sign accounts for the fact that all core
                    # charges have the opposite sign of the shell charges

                # Mon2 multipoles with Mon1 drude core
                self.Mon2Multipoles.xyz2 = self.xyz1 # Update shell positions in case these have changed
                self.Mon2Multipoles.update_direction_vectors()
                xi = self.xyz1[:,i,:]
                xj = self.xyz2[:,j,:]
                dx = xi - xj
                if self.inter_damping_type == 'Thole':
                    args = (ai,aj,p,dx[:,0],dx[:,1],dx[:,2])
                else:
                    args = (bij,dx[:,0],dx[:,1],dx[:,2])
                for mj in self.Mon2Multipoles.multipoles1[j].keys():
                    int_type = (mj,'Q00')
                    damp = self.damp_inter(*args)
                    edrude -= damp*self.Mon2Multipoles.get_multipole_energy(j,i,int_type)
                    # Minus sign accounts for the fact that all core
                    # charges have the opposite sign of the shell charges

        # Include spring energy:
        # Spring Energy Monomer1
        kdr2 = np.sum(self.springcon1*(self.xyz1 - self.shell_xyz1)**2, axis=-1)
        edrude += 0.5*np.sum(kdr2, axis=-1)
        # Spring Energy Monomer 2
        kdr2 = np.sum(self.springcon2*(self.xyz2 - self.shell_xyz2)**2, axis=-1)
        edrude += 0.5*np.sum(kdr2, axis=-1)

        return edrude
####################################################################################################    


####################################################################################################    
    def get_efield_from_multipole_charge(
            self,j,ishell,Multipoles_j,xvec,bij=1.0,ai=1.0,aj=1.0,p=2.0,smallq=1e-6): 
        '''Compute the electric field due to a screened electric field
        centered at xvec.

        More specifically, here it is assumed that the potential, for a given
        multipole moment Tij, is given by
             V = f_damp*(Qi Tij Qj)
        where f_damp is either 1 or given by the standard Tang-Toennies damping function,
        and Tij is the interaction tensor described in Appendix F of Stone's
        book.
        This yields an electric field
             E = - del V
               = - ( del(f_damp)*(Qi Tij Qj)  + f_damp d(Qi Tij Qj) )
        Note that, in the limit where Q or r are sufficiently small, E = 0.

        Parameters
        ----------
        Multipoles_j : class
            Class containing information about the multipole moments at site j
        xvec: ndarray
            Array containing the relative position of q for each data point
            with respect to a point x1 where we are calculating the electric
            field.
        bij : float
            Screening parameter corresponding to the exponent used in the
            repulsive potential (described in more detail elsewhere). Only
            needed if f_damp is Tang-Toennies
        ai, aj : floats
            Screening parameters corresponding to the polarizabilities of
            the atoms with indices ishell and j. Only needed if f_damp is Thole.
        p : float
            Screening parameter corresponding to the dimensionless Thole
            screenlength. Only needed if f_damp is Thole.
        smallq : float, optional.
            Parameter to determine magnitude of a multipole moment Q that is
            sufficiently small so as to set E = 0 for that moment.

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

        qt = np.zeros_like(r)
        delqt = np.zeros(r.shape + (3,))
        efield = np.zeros_like(delqt)

        for mj,qj in Multipoles_j.multipoles1[j].items():
            if abs(qj) < smallq:
                continue
            mishell = '00'
            int_type = (mj.lstrip('Q'),mishell)

            # Compute multipole interaction as well as the derivative of the
            # multipole field
            qt = qj*Multipoles_j.get_interaction_tensor(j,ishell,int_type)
            delqt = qj*Multipoles_j.get_del_interaction_tensor(j,ishell,int_type)
            
            if self.damp_charges_only and mj.lstrip('Q') != '00':
                damp = np.ones_like(qt)
                ddamp = np.zeros_like(delqt)
            ## elif not self.damp_charges_only:
            ##     if self.inter_damping_type in ['Tang-Toennies', 'Thole']:
            ##         raise NotImplementedError,\
            ##         'TT damping for higher order multipole moments not yet implemented!'
            #FIXME: Put in correct damping functions for higher order multipole moments
            else:
                if self.inter_damping_type == 'Thole':
                    args = (ai,aj,p,xij,yij,zij)
                else:
                    args = (bij,xij,yij,zij)
                damp = np.where( r > self.small_r, self.damp_inter(*args), 0)

                ddamp = np.array([ np.where( r > self.small_r, 
                                                 dcharge(*args), 0) 
                                            for dcharge in self.del_damp_inter ])
                ddamp = np.swapaxes(ddamp,0,1) # Do we need this?

            # Get rid of data points where r is too small
            qt[r < self.small_r] = 0
            delqt[r < self.small_r,:] = np.zeros(3)

            # Compute efield
            ## print 'efield shapes:'
            ## print efield.shape
            ## print damp.shape
            ## print delqt.shape
            ## print ddamp.shape
            ## print qt.shape
            efield += -1*( damp[:,np.newaxis]*delqt + ddamp*qt[:,np.newaxis])
            #efield += -1*( damp[:,np.newaxis]*delqt[:,j,:] + ddamp*qt[:,np.newaxis])


        efield *= self.multipole_efield_scale_factor
        return efield
####################################################################################################    


####################################################################################################    
    def get_efield_from_point_charge(self,q,xvec,bij=1.0,ai=1.0,aj=1.0,p=1.0): 
        '''Compute the electric field due to a screened point charge qj located at xvec.

        More specifically, here it is assumed that the potential from the
        point charge is given by
             V = f_damp*qj/r
        where f_damp is given by a user-specified damping function (which may
        depend on bij or qi),
        and r is the norm of xvec.
        yielding an electric field
             E = - del V
               = - ( del(f_damp)*q/r - f_damp*q/r^3*xvec )
        Note that, in the limit where q or r are sufficiently small, E = 0.

        Parameters
        ----------
        q : float
            Magnitude of the point charge located at xvec
        bij : float
            Screening parameter; should correspond to the exponent used in the
            repulsive potential (described in more detail elsewhere).
        xvec: ndarray
            Array containing the relative position of q for each data point
            with respect to a point x1 where we are calculating the electric
            field.
        ai, aj : floats
            Screening parameters corresponding to the polarizabilities of
            the atoms with indices ishell and j. Only needed if f_damp is Thole.
        p : float
            Screening parameter corresponding to the dimensionless Thole
            screenlength. Only needed if f_damp is Thole.

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

        if self.inter_damping_type == 'Thole':
            args = (ai,aj,p,xij,yij,zij)
        else:
            args = (bij,xij,yij,zij)

        damp = np.where( r > self.small_r, self.damp_inter(*args)/r**3, 0)
        efield = damp[:,np.newaxis]*q*xvec

        ddamp = np.array([ np.where( r > self.small_r, \
                                         - dcharge(*args)*q/r,\
                                         0) for dcharge in self.del_damp_inter ])
        efield += np.swapaxes(ddamp,0,1)

        return efield
####################################################################################################    


####################################################################################################    
    def get_efield_from_thole_charge(self,q1,q2,xvec,a1,a2,p): 
        '''Compute the electric field due to a screened point charge q2 located at xvec.

        More specifically, here it is assumed that the potential from the
        point charge is given by
             V = f_damp*q/r
        where f_damp is given by the standard Thole damping function,
        and r is the norm of xvec.
        yielding an electric field
             E = - del V
               = - ( del(f_damp)*q/r - f_damp*q/r^3*xvec )
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
        a1, a2 : floats
            Screening parameters corresponding to the polarizabilities of
            atoms 1 and 2.
        p : float
            Dimensionless Thole screenlength. 

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
        args = (a1, a2, p, xij, yij, zij)
        damp = np.where( r > self.small_r, self.damp_intra(*args)/r**3, 0)
        efield = damp[:,np.newaxis]*q2*xvec

        ddamp = np.array([ np.where( r > self.small_r, \
                                         - ddrude(*args)*q2/r,\
                                         0) for ddrude in self.del_damp_intra ])
        efield += np.swapaxes(ddamp,0,1)

        return efield
####################################################################################################    


####################################################################################################    
    def get_thole_damping_factor(self,ai,aj,p,xij,yij,zij,thole_style='linear'):
        '''Compute the Thole damping factor.

        References:
        (1) Yu, K.; McDaniel, J. G.; Schmidt, J. R. J. Phys. Chem. B 2011, 115, 10054-10063.

        Parameters
        ----------
        ai : Symbol
            Polarizability of the first drude oscillator.
        aj : Symbol
            Polarizability of the second drude oscillator.
        p  : Symbol
            Dimensionless Thole screening parameter.
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

        if thole_style == 'linear':
            prefactor = 1.0 + p*rij/(2*(ai*aj)**(1.0/6))
            exponent = p*rij/(ai*aj)**(1.0/6)
            damping_factor = 1.0 - sp.exp(-exponent)*prefactor
        elif thole_style == 'tinker':
            a=p 
            uij = rij/(ai*aj)**(1.0/6.0)
            au3 = a*uij**3
            expau3 = sp.exp(-au3)
            approx_gamma = expau3*((4.0/9.0) + 2*(1+au3)**2) / (2*(1+ au3)**(7.0/3.0))
            damping_factor = 1.0-expau3 + uij*a**(1.0/3.0)*approx_gamma
        else:
            raise NotImplementedError('Only Linear and Tinker-style Thole damping is currently implemented.')

        return damping_factor
####################################################################################################    


####################################################################################################    
    def get_thole_multipole_damp(self,ai,aj,p,xij,yij,zij,m=0,thole_style='linear'):
        '''Compute the Thole damping factor.

        References:
        (1) Yu, K.; McDaniel, J. G.; Schmidt, J. R. J. Phys. Chem. B 2011, 115, 10054-10063.

        Parameters
        ----------
        ai : Symbol
            Polarizability of the first drude oscillator.
        aj : Symbol
            Polarizability of the second drude oscillator.
        p  : Symbol
            Dimensionless Thole screening parameter.
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
        rij = np.sqrt(xij**2 + yij**2 + zij**2)

        if thole_style == 'linear':
            if m==0:
                prefactor = 1.0 + p*rij/(2*(ai*aj)**(1.0/6))
                exponent = p*rij/(ai*aj)**(1.0/6)
                damping_factor = 1.0 - np.exp(-exponent)*prefactor
            else:
                raise NotImplementedError('Linear-style Thole damping only implemented up to rank 0')
        elif thole_style == 'tinker':
            a=p 
            uij = rij/(ai*aj)**(1.0/6.0)
            au3 = a*uij**3
            expau3 = np.exp(-au3)
            if m==0:
                approx_gamma = expau3*((4.0/9.0) + 2*(1+au3)**2) / (2*(1+ au3)**(7.0/3.0))
                damping_factor = 1.0-expau3 + uij*a**(1.0/3.0)*approx_gamma
            elif m==1:
                damping_factor = 1 - expau3
            elif m==2:
                #FIXME: This formula is incorrect, and needs to be correctly derived for the m == 2 case
                damping_factor = 1 - expau3
            else:
                raise NotImplementedError('Tinker-style Thole damping only implemented up to rank 1')
        else:
            raise NotImplementedError('Only Linear and Tinker-style Thole damping is currently implemented.')

        return damping_factor
####################################################################################################    


####################################################################################################    
    def get_tt_damping_factor(self, bij, xij, yij, zij,slater_correction=True):
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

        if slater_correction:
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


