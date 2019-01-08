#!/usr/bin/env python
"""
Takes as input a multipole file (perhaps from an ISA calculation), averages
the multipole moments, and returns a new .mom file for use in the force field
fitting program.

Last Updated: 10/6/16
"""

# Standard modules
import numpy as np
import sys
import os
import argparse
# mvanvleet specific modules
#from chemistry import io

###########################################################################
####################### Global Variables ##################################
error_message='''
---------------------------------------------------------------------------
Improperly formatted arguments. Proper usage is as follows:

$ averave_mom.py <ifile> <ofile>

(<...> indicates required argument, [...] indicates optional argument)
---------------------------------------------------------------------------
    '''

###########################################################################
###########################################################################


###########################################################################
def rotate_sph_harm(moments, r):
    '''
    Takes a set of spherical tensor moments and rotation matrix R for ordinary
    3-d vectors and returns the transformed set of spherical harmonics. 

    References
    ----------
    subroutine "wigner" from rotations.f90 in Anthony Stone's ORIENT code; this
    function borrows heavily from his code

    Wigner D-matrices:
    https://en.wikipedia.org/wiki/Wigner_D-matrix
    Stone, Theory of Intermolecular Forces, 2nd ed. p. 274

    Real and complex Spherical Harmonics:
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form

    '''


    # Set up Wigner transformation matrix D. Note that D is block-diagonal, so
    # we can set up each block (D0, D1, and D2) separately.
    xx=r[0,0]
    xy=r[0,1]
    xz=r[0,2]
    yx=r[1,0]
    yy=r[1,1]
    yz=r[1,2]
    zx=r[2,0]
    zy=r[2,1]
    zz=r[2,2]

    # D0
    D0 = np.ones(1)

    # D1
    D1 = np.array([[ zz, zx, zy ],
                   [ xz, xx, xy ],
                   [ yz, yx, yy ]])

    # D2
    D2 = np.zeros((5,5))
    rt3 = np.sqrt(3)
    D2[0,0] = (3*zz**2-1)/2
    D2[0,1] = rt3*zx*zz
    D2[0,2] = rt3*zy*zz
    D2[0,3] = (rt3*(-2*zy**2-zz**2+1))/2
    D2[0,4] = rt3*zx*zy
    D2[1,0] = rt3*xz*zz
    D2[1,1] = 2*xx*zz-yy
    D2[1,2] = yx+2*xy*zz
    D2[1,3] = -2*xy*zy-xz*zz
    D2[1,4] = xx*zy+zx*xy
    D2[2,0] = rt3*yz*zz
    D2[2,1] = 2*yx*zz+xy
    D2[2,2] = -xx+2*yy*zz
    D2[2,3] = -2*yy*zy-yz*zz
    D2[2,4] = yx*zy+zx*yy
    D2[3,0] = rt3*(-2*yz**2-zz**2+1)/2
    D2[3,1] = -2*yx*yz-zx*zz
    D2[3,2] = -2*yy*yz-zy*zz
    D2[3,3] = (4*yy**2+2*zy**2+2*yz**2+zz**2-3)/2
    D2[3,4] = -2*yx*yy-zx*zy
    D2[4,0] = rt3*xz*yz
    D2[4,1] = xx*yz+yx*xz
    D2[4,2] = xy*yz+yy*xz
    D2[4,3] = -2*xy*yy-xz*yz
    D2[4,4] = xx*yy+yx*xy

    # Create D matrix from D0, D1, and D2
    D = np.zeros((9,9))
    D[0,0] = D0
    D[1:4,1:4] = D1
    D[4:9,4:9] = D2

    # Perform the rotation
    rotated_moments = np.dot(D,moments)

    return rotated_moments
###########################################################################


####################################################################################################    
def get_local_to_global_rotation_matrix(global_xyz,local_xyz):
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
    trans_local_xyz = local_xyz[np.newaxis,:]
    trans_global_xyz = global_xyz
    # if xyz is an atom, don't need to rotate local axes
    if len(global_xyz[0]) == 1:
        assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
        return rotation_matrix

    # Otherwise, rotate coordinate frame in order to properly align x-axis
    v1 = trans_local_xyz[:,0]
    v2 = trans_global_xyz[:,0] 
    q_w,q_vec = get_rotation_quaternion(v1,v2)

    np.seterr(all="ignore")
    trans_local_xyz = rotate_local_xyz(q_w, q_vec, trans_local_xyz)
    rotation_matrix = rotate_local_xyz(q_w,q_vec, rotation_matrix)
    np.seterr(all="warn")

    # If xyz is diatomic, don't need to rotate second set of axes
    if len(global_xyz[0]) == 2:
        assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
        return rotation_matrix

    # Otherwise, rotate coordinate frame to properly align y- and z-axes
    v1 = trans_local_xyz[:,0]
    v2 = trans_global_xyz[:,0]
    for i in range(2,len(global_xyz[0])):
        # Search for vector in molecule that is not parallel to the first
        v1b = trans_local_xyz[:,i] #- trans_local_xyz[:,0]
        v2b = trans_global_xyz[:,i] #- trans_global_xyz[:,0]
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
    q_w,q_vec = get_rotation_quaternion(v3,v4,v_orth=v1)
    np.seterr(all="ignore")
    trans_local_xyz = rotate_local_xyz(q_w, q_vec, trans_local_xyz)
    rotation_matrix = rotate_local_xyz(q_w,q_vec, rotation_matrix)
    np.seterr(all="warn")

    try:
        assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
    except AssertionError:
        print 'bad rotation!!'
        print np.max(trans_local_xyz-trans_global_xyz)

        print trans_local_xyz[0]
        print '---'
        print trans_global_xyz[0]
        sys.exit()

    # Check that rotation matrix actually transforms local axis into global
    # axis
    if not np.allclose(np.dot(rotation_matrix,local_xyz)[0],global_xyz):
        print rotation_matrix
        print local_xyz
        print np.dot(rotation_matrix,local_xyz)
        print 
        print 'Rotation matrix does not actually transform local axis into global reference frame!'
        sys.exit()

    # Convert rotation matrix to Euler angles (need later for Wigner
    # transform)
    # See staff.city.ac.uk/~sbbh653/publications/euler.pdf for a tutorial on
    # how this is done. Note, however, that unlike the above article, we need
    # to use the z-y-z Euler representation (which rotate_sph_harm depends
    # on), and so the equations I've derived for the transformation differ
    # slightly.

    # We'll eventually be transforming the global axis to the local axis, so
    # we actually need alpha, beta, and gamma for R^-1:
    ## rotation_matrix = np.linalg.inv(rotation_matrix)
    ## r21 = rotation_matrix[0,2,1]
    ## r22 = rotation_matrix[0,2,2]
    ## r20 = rotation_matrix[0,2,0]
    ## r02 = rotation_matrix[0,0,2]
    ## r12 = rotation_matrix[0,1,2]
    ## r01 = rotation_matrix[0,0,1]
    ## r10 = rotation_matrix[0,1,0]
    ## r00 = rotation_matrix[0,0,0]

    ## beta = np.arccos(r22)
    ## alpha = np.arctan2(r12,r02)
    ## gamma = np.arctan2(-r21,r20)

    return rotation_matrix
####################################################################################################    


####################################################################################################    
def get_rotation_quaternion(v1,v2,v_orth=np.array([1,0,0]),tol=1e-16):
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

    # For antiparallel vectors, rotate 180 degrees
    q_w = np.sqrt(np.sum(v1*v1,axis=-1)*np.sum(v2*v2,axis=-1)) + dot

    # Normalize quaternion
    q_norm = np.sqrt(q_w**2 + np.sum(q_vec**2,axis=-1))
    q_w /= q_norm
    q_vec /= q_norm[:,np.newaxis]

    return q_w, q_vec
####################################################################################################    


####################################################################################################    
def read_local_axis_information(atoms,global_xyz,ifile):
    '''
    '''

    with open(ifile,'r') as f:
        axes_lines = [ line.split() for line in f.readlines()]

    start = axes_lines.index(['Axes']) + 2

    axes = [ [ [],[] ] for i in atoms ]
    for line in axes_lines[start:]:
        try:
            iatom = int(line[0])
        except (IndexError,TypeError):
            continue
        iaxis = 0 if line[1] == 'z' else 1 # list x and z axes seperately
        if axes[iatom][iaxis] != []:
            print 'The '+line[1]+' axis for atom '+line[0]+\
                    ' in monomer 2 has already been specified.'
            print 'Please only use one axis specification line per axis per atom.'
            sys.exit()
        else:
            axes[iatom][iaxis] = [ int(i) for i in line[2:] ]


    local_xyz = np.zeros((len(atoms),3,3))
    for iatom, (atom, axis) in enumerate(zip(atoms,axes)):

        # Get z-axis from axes information
        try:
            i = axes[iatom][0][0]
        except IndexError:
            print 'Can\'t find complete axis system for atom ',iatom
            print 'Did you forget to specify a local axis system for this atom?'
            raise
        z1 = global_xyz[i]
        z2 = np.mean([global_xyz[j] for j in axes[iatom][0][1:]],axis=0)
        z_axis = z2 - z1
        z_axis /= np.sqrt((z_axis ** 2).sum(-1))[..., np.newaxis] #Normalize

        try:
            # Get x-axis from axes information unless blank
            i = axes[iatom][1][0]
            x1 = global_xyz[i]
            x2 = np.mean([global_xyz[j] for j in axes[iatom][1][1:]],axis=0)
            vec = x2 - x1
        except IndexError:
            vec = np.array([1,0,0])
        # Project x-axis onto plane
        direction = np.cross(z_axis,np.cross(vec,z_axis))
        # In the case where a vector is perfectly in line with the z-axis,
        # return some default value for the x-axis
        if np.allclose(direction,[0,0,0]):
            direction = np.array([1,0,0])
        direction /= np.linalg.norm(direction)
        cos_angle = np.sum(vec*z_axis) / np.sqrt((vec ** 2).sum())
        angle = np.arccos(cos_angle)
        magnitude = np.sqrt((vec ** 2).sum(-1))*np.sin(angle)

        x_axis = magnitude*direction 
        x_axis /= np.linalg.norm(x_axis)
        if np.dot(z_axis,x_axis) > 1e-7:
            print 'not normalized!'
            print np.dot(z_axis,x_axis)
            sys.exit()

        y_axis = np.cross(z_axis,x_axis)
        y_axis /= np.linalg.norm(y_axis)

        local_xyz[iatom] = np.array([x_axis, y_axis, z_axis])

    return local_xyz
####################################################################################################    


####################################################################################################    
def rotate_local_xyz(a,vector=np.array([0,0,1]),local_xyz=np.array([1,2,3]),thresh=1e-14):
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
def write_mom_file(template_file,multipoles,ofile):
    '''
    '''
    with open(template_file,'r') as f:
        lines = f.readlines()
        data = [ line.split() for line in lines ]

    with open(ofile,'w') as f:
        collect_flag = True
        multipole_count = 0
        iatom = -1 
        for i,line in enumerate(data):
            if not line:
                collect_flag = True
                f.write('\n')
            elif line[0] == '!':
                # Skip comments
                f.write(lines[i])
            elif collect_flag == True:
                # Write atom header line
                iatom += 1
                f.write(lines[i])
                multipole_count = 0
                collect_flag = False
            else:
                # Lines containing multipoles
                q = line[0]
                m = multipoles[iatom][multipole_count]
                f.write('\t\t{:5s} = {:8.6f} \n'.format(q,m))
                multipole_count += 1
####################################################################################################    


###########################################################################
########################## Main Code ######################################

def average_mom(ifile,iaxes,average=True,set_equal_to=False,scale=1,cutoff=1e-3,trim=True):

    # Read in data from mom file
    with open(ifile,'r') as f:
        mom_lines = [ line.split() for line in f.readlines() ]

    # Obtain point charge, dipole, and quadrupole information for each atom
    tag = 'Type'
    atoms = []
    labels = []
    moments = []
    xyz = []
    atomtypes = []
    for line in mom_lines:
        if len(line) > 5 and line[4] == tag:
            atoms.append(line[0])
            atomtypes.append(line[5])
            labels.append([])
            moments.append([])
            xyz.append([float(i) for i in line[1:4]])
        elif not line or '!' in line[0] or line[0] in ['End','Units']:
            continue
        else:
            labels[-1].append(line[0])
            moments[-1].append(float(line[2]))

    # Ensure labels are ordered normally
    label_order = ['Q00', 'Q10', 'Q11c', 'Q11s', 'Q20', 'Q21c', 'Q21s', 'Q22c', 'Q22s']
    for l in labels:
        assert l == label_order

    # Read in axis information for each atom
    local_axis = read_local_axis_information(atoms,xyz,iaxes)
    global_xyz = np.eye(3)

    # Compute quaternion to rotate global axis frame to local one
    rotated_moments = []
    for iatom in xrange(len(atoms)):
        R = get_local_to_global_rotation_matrix(global_xyz[np.newaxis,...],local_axis[iatom])
        # Since we're transforming from the global to the local frame, we actually
        # need Rinv:
        Rinv = np.linalg.inv(R)
        rotated_moments.append(rotate_sph_harm(moments[iatom],Rinv[0]))

    # Average together moments of the same atomtype
    rotated_moments = np.array(rotated_moments)
    averaged_moments = {}
    if not (average or set_equal_to):
        atomtypes = atoms
    for atomtype in set(atomtypes):
        iatomtypes = [ i for i,a in enumerate(atomtypes) if a == atomtype ]
        # Use numpy's masking ability to only keep rotated moments of the
        # specified atomtype
        all_moments = rotated_moments[iatomtypes]
        if average:
            averaged_moments[atomtype] = np.average(all_moments,axis=0)
        elif set_equal_to:
            averaged_moments[atomtype] = all_moments[0]
        else:
            averaged_moments[atomtype] = all_moments[0]

    # Get rid of too-small moments if the trim option has been selected
    if trim:
        for atom,mom in averaged_moments.items():
            max_l0 = np.max(np.abs(mom[0]))
            max_l1 = np.max(np.abs(mom[1:4]))
            max_l2 = np.max(np.abs(mom[4:9]))
            cutoff = cutoff*np.array([max_l0] + [max_l1]*3 + [max_l2]*5)
            new_mom = np.where( np.abs(mom) > np.abs(cutoff),
                                    mom,
                                    0 )
            averaged_moments[atom] = new_mom

    # Multiply all moments by specified scale factor (default 1)
    for atom,mom in averaged_moments.items():
        averaged_moments[atom] = scale*mom


    rotated_moments = []
    for iatom in xrange(len(atoms)):
        #R = get_local_to_global_rotation_matrix(global_xyz[np.newaxis,...],local_axis[iatom])
        R = get_local_to_global_rotation_matrix(global_xyz[np.newaxis,...],local_axis[iatom])
        ## # For the backwards rotation, we actually need Rinv
        ## R = np.linalg.inv(R)

        atomtype = atomtypes[iatom]
        rotated_moments.append(rotate_sph_harm(averaged_moments[atomtype],R[0]))


    return averaged_moments, rotated_moments


###########################################################################


###########################################################################
def convert_mom_to_xml(moments,kz='DEFAULT',kx='DEFAULT',atomtype_numbers=[]):
    '''
    '''
    # Rotate each multipole according to the local axis information provided in
    # the .axes file
    au_to_nm = 0.0529177
    template = '<Multipole type="{}" kz="{}" kx="{}" c0="{:.6g}" d1="{:.6g}" d2="{:.6g}" d3="{:.6g}" q11="{:.6g}" q21="{:.6g}" q22="{:.6g}" q31="{:.6g}" q32="{:.6g}" q33="{:.6g}"  />'


    # Set default parameters
    if kz=='DEFAULT':
        kz = [0 for i in moments]
    if kx=='DEFAULT':
        kx = [0 for i in moments]
    if not atomtype_numbers:
        atomtype_numbers = range(1,len(moments)+1)
    assert len(atomtype_numbers) == len(kz) == len(kx) == len(moments)

    # Print labels in Cartesian coordinates and OpenMM units
    xml = ''
    #for atom, m in zip(atoms,moments):
    for i,(atom, m) in enumerate(moments.items()):
        c0 = m[0] #point charge
        d1 = m[2] # Q11c = dx = d1
        d2 = m[3] # Q11s = dy = d2
        d3 = m[1] # Q10  = dz = d3
        # OpenMM prints uses values for the quadrupole moments that are 3x the
        # value given by CamCASP (presumably due to normalization)
        q11 = (1.0/3)*(np.sqrt(3)/2*m[7] - 0.5*m[4])
        q21 = (1.0/3)*(np.sqrt(3)/2*m[8])
        q22 = (1.0/3)*(-np.sqrt(3)/2*m[7] - 0.5*m[4])
        q31 = (1.0/3)*(np.sqrt(3)/2*m[5])
        q32 = (1.0/3)*(np.sqrt(3)/2*m[6])
        q33 = (1.0/3)*(m[4])

        d1 *= au_to_nm
        d2 *= au_to_nm
        d3 *= au_to_nm

        q11 *= au_to_nm**2
        q21 *= au_to_nm**2
        q22 *= au_to_nm**2
        q31 *= au_to_nm**2
        q32 *= au_to_nm**2
        q33 *= au_to_nm**2


        atom_xml = template.format(i+1,kz[i],kx[i],c0,d1,d2,d3,q11,q21,q22,q31,q32,q33)
        xml += ' '.join(atom_xml.split()) + '\n'

    return xml

###########################################################################

###########################################################################
###########################################################################


###########################################################################
######################## Command Line Parser ###########################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    averagehelp="Averages the multipole moments on an atomtype basis. Defaults to True."
    trimhelp="""Eliminates multipole moments below a certain relative cutoff
    compared to the largest multipole moment of a given order. Defaults to
    False."""

    #parser.add_argument("energy_file", type=str, help=energyhelp)
    parser.add_argument("ifile", help="Input .mom file")
    parser.add_argument("iaxes", help="Input axis file")
    parser.add_argument("-o","--ofile",
            help="Output .mom file (defaults to ifile_averaged.mom if left unspecified", 
            default='FILL')
    parser.add_argument("-t","--trim", help=trimhelp,\
             action="store_true", default=False)
    parser.add_argument("--cutoff", 
            type=np.float,
            help="Relative cutoff below which multipole moments are truncated.",
             default=0.001)
    parser.add_argument("--scale", 
            type=np.float,
            help="Scale factor for multipole moments; default 1",
             default=1.0)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a","--average", help=averagehelp,\
             action="store_true", default=False)
    group.add_argument("-s","--set_equal_to", help=averagehelp,\
             action="store_true", default=False)


    args = parser.parse_args()

    local_moments, global_moments = average_mom(args.ifile,args.iaxes,
            average=args.average,set_equal_to=args.set_equal_to,
            scale=args.scale,cutoff=args.cutoff,trim=args.trim)

    print convert_mom_to_xml(local_moments)

    # Write .mom file
    if args.ofile == 'FILL':
        args.ofile = args.ifile.replace('.mom','_averaged.mom')
    write_mom_file(args.ifile,global_moments,args.ofile)
###########################################################################
###########################################################################


