#!/usr/bin/env python
"""

Last Updated:
"""

# Standard modules
import numpy as np
import sys
import os
# mvanvleet specific modules
from chemistry.constants import bohr2ang, ang2bohr

###########################################################################
####################### Global Variables ##################################
error_message='''
---------------------------------------------------------------------------
Improperly formatted arguments. Proper usage is as follows:

$ 

(<...> indicates required argument, [...] indicates optional argument)
---------------------------------------------------------------------------
    '''


###########################################################################
###########################################################################


###########################################################################
######################## Command Line Arguments ###########################
try:
    ifile = sys.argv[1]
    ofile = sys.argv[2]
except IndexError:
    print error_message
    sys.exit()

res1 = 'foo'
res2 = 'bar'


###########################################################################
###########################################################################


###########################################################################
########################## Main Code ######################################

###########################################################################
def read_sapt(file):
    with open(file,'r') as f:
        data = [line.split() for line in f.readlines()]

    imon1 = int(data[0][0])
    imon2 = int(data[1+imon1][0])
    nsapt_lines = data.index([]) + 1

    atoms_mon1 = [ i[0] for i in data[1:imon1+1]]
    atoms_mon2 = [ i[0] for i in data[imon1 + 2:imon1 + imon2 + 2]]
    
    if atoms_mon1 == atoms_mon2:
        res2 = res1

    count = 1
    coords = []
    while count < len(data):
        coords.append([])
        # Parse mon1 coordinates
        coords[-1].append([i[1:] for i in data[count:count+imon1]])
        # Parse mon2 coordinates
        coords[-1].append([i[1:] for i in data[count+imon1+1:count+imon1+imon2+1]])
        count += nsapt_lines

    coords = np.array(coords,dtype=float)
    # Convert from bohr to ang
    coords *= bohr2ang

    return [atoms_mon1, atoms_mon2], coords
###########################################################################


###########################################################################
def read_pdb(file):
    # Read in PDB file
    with open(file,'r') as f:
        lines = f.readlines()
        data = [line.split() for line in lines]

    ## # Get unit cell dimensions
    ## for line in data:
    ##     if line[0] == 'CRYST1':
    ##         unit_cell = [float(i) for i in line[1:7]]
    ##         break

    # Parse atom positions
    positions = [[[],[]]]
    info = [[],[]]
    get_info = True
    for line in data:
        if line[0] == 'ATOM' or line[0] == 'HETATM':
            imon = int(line[4])
            positions[-1][imon-1].append([float(i) for i in line[5:8]])
            if get_info:
                element = line[2]
                ## residue = line[3]
                ## nresidue = int(line[4])
                info[imon-1].append(element)
        #elif line[0] == 'MODEL':
        elif line[0] == 'ENDMDL':
            get_info = False
            #positions.append([[],[]])
        else:
            continue

    return info, np.array(positions,dtype=float)
###########################################################################


###########################################################################
def write_sapt(file,names,coords):

    imon1 = len(coords[0,0,:])
    imon2 = len(coords[0,1,:])

    sapt_text = '''\
E1pol               -1.116317
E1exch              0.056701
E1exch(S2)          0.056698
E2ind(unc)          -0.110409
E2ind               -0.085579
E2ind-exch          0.007376
E2disp(unc)         -0.431671
E2disp              -0.285280
E2disp-exch(unc)    0.010401
E2disp-exch         0.009240
E1tot               -1.059616
E2tot               -0.354243
E1tot+E2tot         -1.413859
E2ind[B->A]         0.000000
E2ind[A->B]         0.000000
E2exchind_BA        0.003120
E2exchind_AB        0.004250
dhf                 -0.005981

'''

    with open(file,'w') as f:
        template = '{:5s} {:16.8f} {:16.8f} {:16.8f}\n'
        for line in coords:
            imon1 = len([n for n in names[0] if 'dru' not in n.lower()])
            imon2 = len([n for n in names[1] if 'dru' not in n.lower()])
            f.write('{}\n'.format(imon1))
            for name, xyz in zip(names[0],line[0]):
                #Don't include drude particles
                if 'dru' in name.lower():
                    continue
                xyz *= ang2bohr
                f.write(template.format(name,*xyz))
            f.write('{}\n'.format(imon2))
            for name, xyz in zip(names[1],line[1]):
                #Don't include drude particles
                if 'dru' in name.lower():
                    continue
                xyz *= ang2bohr
                f.write(template.format(name,*xyz))
            f.write(sapt_text)

    return
###########################################################################


###########################################################################
def write_pdb(file, coords):
    sys.exit('PDB output not yet implemented!')
###########################################################################


if ifile.split('.')[-1] == 'sapt':
    names, coords = read_sapt(ifile)
elif ifile.split('.')[-1] == 'pdb':
    names, coords = read_pdb(ifile)
else:
    sys.exit('Unrecognized input file type!')

if ofile.split('.')[-1] == 'sapt':
    write_sapt(ofile,names,coords)
elif ofile.split('.')[-1] == 'pdb':
    write_pdb(ofile,names,coords)
else:
    sys.exit('Unrecognized output file type!')




###########################################################################
###########################################################################
