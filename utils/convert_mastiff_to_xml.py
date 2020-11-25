#!/usr/bin/env python
"""

Last Updated:
"""

# Standard modules
import numpy as np
import sys
import os
from distutils.version import LooseVersion
import json
# mvanvleet specific modules
#from chemistry import io
import pointer
import format_mom

###########################################################################
####################### Global Variables ##################################
error_message='''
---------------------------------------------------------------------------
Improperly formatted arguments. Proper usage is as follows:

$ {} <constraints_ifile> <mom_ifile> <axes_ifle>

(<...> indicates required argument, [...] indicates optional argument)
---------------------------------------------------------------------------
    '''

#thole = 1.0
# List of all possible spherical harmonic prameters included
_all_sph = ['y10', 'y20', 'y22c']
#_sph_conv = [np.sqrt(4*np.pi/3), np.sqrt(4*np.pi/5), np.sqrt(4*np.pi/5)]
_sph_conv = [1,1,1]

# Convert parameters to SimTK unit system (daltons, nm, ps)
au_to_dalton = 0.000548579909
au_to_nm = 0.0529177
au_to_ps = 2.418884254e-5

_conv_energy = au_to_dalton * au_to_nm**2 / au_to_ps**2
_conv_a = np.sqrt(_conv_energy)
_conv_b = 1/au_to_nm
_conv_cn = [ np.sqrt(_conv_energy * au_to_nm **n) for n in range(6,14,2) ]
_conv_alpha = au_to_nm**3


aniso_pre_text = '''
<CustomAnisotropicNonbondedForce bondCutoff="4"
     energy="(A*K2*exBr - Adi*(f6*C6/(r^6) + f8*C8/(r^8) + f10*C10/(r^10) + f12*C12/(r^12)));
    A=Aex-Ael-Ain-Adh;
    ;
    Aex=(Aexch1*Aexch2*Aexch1_sph*Aexch2_sph);
    Aexch1_sph= 1 + aexch_y101*y10_1 + aexch_y201*y20_1 + aexch_y22c1*y22c_1;
    Aexch2_sph= 1 + aexch_y102*y10_2 + aexch_y202*y20_2 + aexch_y22c2*y22c_2;
    ;
    Ael=(Aelec1*Aelec2*Aelec1_sph*Aelec2_sph);
    Aelec1_sph= 1 + aelec_y101*y10_1 + aelec_y201*y20_1 + aelec_y22c1*y22c_1;
    Aelec2_sph= 1 + aelec_y102*y10_2 + aelec_y202*y20_2 + aelec_y22c2*y22c_2;
    ;
    Ain=(Aind1*Aind2*Aind1_sph*Aind2_sph);
    Aind1_sph= 1 + aind_y101*y10_1 + aind_y201*y20_1 + aind_y22c1*y22c_1;
    Aind2_sph= 1 + aind_y102*y10_2 + aind_y202*y20_2 + aind_y22c2*y22c_2;
    ;
    Adh=(Adhf1*Adhf2*Adhf1_sph*Adhf2_sph);
    Adhf1_sph= 1 + adhf_y101*y10_1 + adhf_y201*y20_1 + adhf_y22c1*y22c_1;
    Adhf2_sph= 1 + adhf_y102*y10_2 + adhf_y202*y20_2 + adhf_y22c2*y22c_2;
    ;
    Adi=(Adisp1*Adisp2*Adisp1_sph*Adisp2_sph);
    Adisp1_sph= 1 + adisp_y101*y10_1 + adisp_y201*y20_1 + adisp_y22c1*y22c_1;
    Adisp2_sph= 1 + adisp_y102*y10_2 + adisp_y202*y20_2 + adisp_y22c2*y22c_2;
    ;
    K2=(Br^2)/3 + Br + 1;
    f12 = f10 - exX*( (1/39916800)*(X^11)*(1 + X/12) );
    f10 = f8 - exX*( (1/362880)*(X^9)*(1 + X/10 ) );
    f8 = f6 - exX*( (1/5040)*(X^7)*(1 + X/8 ) );
    f6 = 1 - exX*(1 + X * (1 + (1/2)*X*(1 + (1/3)*X*(1 + (1/4)*X*(1 + (1/5)*X*(1 + (1/6)*X ) ) )  ) ) );
    exX = exp(-X);
    X = Br - r * ( 2*(B^2)*r + 3*B )/(Br^2 + 3*Br + 3) ;
    exBr = exp(-Br);
    Br = B*r;
    B=sqrt(Bexp1*Bexp2);
    C6=(C61*C62); C8=(C81*C82); C10=(C101*C102); C12=(C121*C122)">
  <PerParticleParameter name="Aexch"/>
  <PerParticleParameter name="aexch_y10"/>
  <PerParticleParameter name="aexch_y20"/>
  <PerParticleParameter name="aexch_y22c"/>
  <PerParticleParameter name="Aelec"/>
  <PerParticleParameter name="aelec_y10"/>
  <PerParticleParameter name="aelec_y20"/>
  <PerParticleParameter name="aelec_y22c"/>
  <PerParticleParameter name="Aind"/>
  <PerParticleParameter name="aind_y10"/>
  <PerParticleParameter name="aind_y20"/>
  <PerParticleParameter name="aind_y22c"/>
  <PerParticleParameter name="Adhf"/>
  <PerParticleParameter name="adhf_y10"/>
  <PerParticleParameter name="adhf_y20"/>
  <PerParticleParameter name="adhf_y22c"/>
  <PerParticleParameter name="Adisp"/>
  <PerParticleParameter name="adisp_y10"/>
  <PerParticleParameter name="adisp_y20"/>
  <PerParticleParameter name="adisp_y22c"/>
  <PerParticleParameter name="Bexp"/>
  <PerParticleParameter name="C6"/>
  <PerParticleParameter name="C8"/>
  <PerParticleParameter name="C10"/>
  <PerParticleParameter name="C12"/>
'''

aniso_post_text = '''
</CustomAnisotropicNonbondedForce>
'''

multipole_pre_text = '''
<AmoebaMultipoleForce  direct11Scale="1.0"  direct12Scale="1.0" direct13Scale="1.0"  direct14Scale="1.0"  mpole12Scale="0.5"
    mpole13Scale="0.5"  mpole14Scale="0.8"  mpole15Scale="0.8" mutual11Scale="0.0"  mutual12Scale="0.0"  mutual13Scale="1.0"
    mutual14Scale="1.0"  polar12Scale="1.0"  polar13Scale="0.0" polar14Intra="0.5"  polar14Scale="1.0"  polar15Scale="1.0"  >
'''

multipole_post_text = '''
</AmoebaMultipoleForce>
'''


###########################################################################
###########################################################################


###########################################################################
def convert_sph(a_params,param_labels,ncomponents):
    '''
    '''

    sph_params = [[ 0 for j in _all_sph] for i in range((ncomponents))]
    for i,label in enumerate(param_labels):
        try:
            j = _all_sph.index(label)
        except ValueError:
            print(label, ' is not a valid spherical harmonic!')
            sys.exit()
        for k in range(ncomponents):
            # Sort and convert from normalized spherical harmonics to
            # unormalized spherical harmonics
            sph_params[k][j] = a_params[k][i]*_sph_conv[j]
            #sph_params[k][j] = a_params[k][i]

    return sph_params
    

###########################################################################


###########################################################################
######################## Command Line Arguments ###########################
try:
    coeffs_ifile = sys.argv[1]
    #atomtypes_ifile = sys.argv[2]
    mom_ifile = sys.argv[2]
    axes_ifile = sys.argv[3]
except IndexError:
    print(error_message.format(sys.argv[0]))
    sys.exit()


###########################################################################
###########################################################################


###########################################################################
########################## Main Code ######################################

assert LooseVersion(pointer.__version__) >= LooseVersion('2.0.0'), "Requires POInter v2.0 or higher"

# Read in parameters from .constraints file
atom_params = {}
with open(coeffs_ifile,'r') as f:
    atom_params.update(json.load(f))

atomtypes = list(atom_params.keys())
atomtype_numbers = [i+1 for i in range(len(atomtypes))]

# Perform unit conversions from a.u. to OpenMM units (nm, dalton, ps)
for atom in atomtypes:
    atom_params[atom]['A'] = [ i*_conv_a for i in atom_params[atom]['A'] ]
    atom_params[atom]['B'] *= _conv_b
    atom_params[atom]['C'] = [ i*j for i,j in zip(atom_params[atom]['C'],_conv_cn)]
    atom_params[atom]['alpha'] = (atom_params[atom]['drude_charge'])**2/atom_params[atom]['springcon']*_conv_alpha

    # Correct Adisp param, which should not have been scaled
    iadisp = atom_params[atom]['params_order'].index('Dispersion')
    atom_params[atom]['A'][iadisp] /= _conv_a

# Sort aniso params to match spherical harmonic ordering in .xml template
for atom in atomtypes:
    ncomponents = len(atom_params[atom]['params_order'])
    atom_params[atom]['aniso'] = convert_sph(
            atom_params[atom]['aniso'],atom_params[atom]['sph_harm'],ncomponents)

# Determine atom z (and x, if applicable) local coordinates
# TODO: Make this function more robust to different local coordinate
# definitions
origin_error = ''' 
For OpenMM compatability, the origin of all local-axis vectors
for an atom must be the atom itself. Your current local-axis definition of atomtype
{} does not match this criteria. You will need to modify your .axes file (and
possibly re-run POInter).
'''
for atom in atomtypes:
    atomz = (atom_params[atom]['comments'][1]).split()
    try:
        iz = atomz.index('to') + 2
        origin_atom = atomz[atomz.index('from') + 2]
        atom_params[atom]['z-axis'] = atomtype_numbers[atomtypes.index(atomz[iz])]
    except ValueError: # no z-axis
        atom_params[atom]['z-axis'] = 0
    else:
        assert origin_atom == atom, origin_error.format(atom)


    atomx = (atom_params[atom]['comments'][2]).split()
    try:
        ix = atomx.index('to') + 2
        origin_atom = atomz[atomx.index('from') + 2]
        atom_params[atom]['x-axis'] = atomtype_numbers[atomtypes.index(atomx[ix])]
    except ValueError: # no x-axis
        atom_params[atom]['x-axis'] = 0
    else:
        assert origin_atom == atom, origin_error.format(atom)


atom_template = '<Atom type="{}" AtomZ="{:d}" AtomX="{:d}"' 
exch_sph = [ a.replace('y','aexch_y') for a in _all_sph]
exch_template = ('\nAexch="{{:8.6e}}" ' + ('{:}="{{:8.6e}}" '*len(exch_sph))).format(*exch_sph)
elec_sph = [ a.replace('y','aelec_y') for a in _all_sph]
elec_template = ('\nAelec="{{:8.6e}}" ' + ('{:}="{{:8.6e}}" '*len(elec_sph))).format(*elec_sph)
ind_sph = [ a.replace('y','aind_y') for a in _all_sph]
ind_template = ('\nAind="{{:8.6e}}" ' + ('{:}="{{:8.6e}}" '*len(ind_sph))).format(*ind_sph)
dhf_sph = [ a.replace('y','adhf_y') for a in _all_sph]
dhf_template = ('\nAdhf="{{:8.6e}}" ' + ('{:}="{{:8.6e}}" '*len(dhf_sph))).format(*dhf_sph)
disp_sph = [ a.replace('y','adisp_y') for a in _all_sph]
disp_template = ('\nAdisp="{{:8.6e}}" ' + ('{:}="{{:8.6e}}" '*len(disp_sph))).format(*disp_sph)
end_template = '\nBexp="{:8.6e}" C6="{:8.6e}" C8="{:8.6e}" C10="{:8.6e}" C12="{:8.6e}"/>'

template = ''.join([atom_template, exch_template,elec_template,
                    ind_template,dhf_template,disp_template,end_template])

print('<!--Atomtype map:')
for i,k in zip(atomtype_numbers,list(atom_params.keys())):
    print(k, ' = atomtype ',i)
print('-->')

# Print CustomAniso portion of .xml file
print(aniso_pre_text)
kz = []
kx = []
for i,(k, v) in enumerate(atom_params.items()):
    z = v['z-axis']
    x = v['x-axis']
    kz.append(z)
    kx.append(x)
    aexch, aelst, aind, adhf, adisp = v['A']
    aexch_sph, aelst_sph, aind_sph, adhf_sph, adisp_sph = v['aniso']
    b = v['B']
    cn = v['C']

    params = [atomtype_numbers[i]] + [z] + [x] +\
             [aexch] + aexch_sph +\
             [aelst] + aelst_sph +\
             [aind] + aind_sph +\
             [adhf] + adhf_sph +\
             [adisp] + adisp_sph +\
             [b] + cn

    print(template.format(*params))
print(aniso_post_text)


# Print AmoebaMultipole Portion of .xml file
print()
print(multipole_pre_text)
local_moments, _ = format_mom.average_mom(mom_ifile,axes_ifile)
print(format_mom.convert_mom_to_xml(local_moments,kz,kx,atomtype_numbers))
for i,(k, v) in enumerate(atom_params.items()):
    template = '<Polarize type="{}" polarizability="{:8.5e}" thole="{:6.4f}" pgrp1="{}" />'
    pgrps = list(set(atomtype_numbers) - set([atomtype_numbers[i]]))
    args = [atomtype_numbers[i]] + [v['alpha']] + [v['thole']] + pgrps
    print(template.format(*args))
print(multipole_post_text)








###########################################################################
###########################################################################
