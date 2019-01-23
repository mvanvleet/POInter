#!/usr/bin/env python
"""

Last Updated:
"""

# Standard modules
import argparse
import numpy as np
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as patches
import itertools
from matplotlib import gridspec
# mvanvleet specific modules
#from chemistry import io

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
parser = argparse.ArgumentParser()
displayhelp="In addition to saving the file, immediately display the plot using plt.show()"
cutoffhelp="""Choose which region of the potential to display in the error
analysis. Total energies above the cutoff value will not be displayed. Default
0 kJ/mol"""
asymptotichelp="""Only display the asymptotic_scale fraction of points with the
lowest SAPT exchange energies (which serve as a proxy for the most asymptotic
points)"""
asymptoticscalehelp="Dictates the fraction of points that will be displayed.  See asymptotic above."

#parser.add_argument("energy_file", type=str, help=energyhelp)
parser.add_argument("prefix", help="prefix", default='fit_exp_')
parser.add_argument("suffix", help="suffixes", default='_unconstrained.dat')
## parser.add_argument("-p","--prefix", help="prefix", default='constrain_exp_')
## parser.add_argument("-s","--suffix", help="suffixes", default='_unconstrained.dat')
parser.add_argument("-d","--display", help=displayhelp,\
         action="store_true", default=False)
parser.add_argument("-c","--cutoff", help=cutoffhelp,\
         type=float, default=0.0)
## parser.add_argument("--fade", help=fadehelp,\
##          action="store_true", default=False)
parser.add_argument("--asymptotic", help=asymptotichelp,\
         action="store_true", default=False)
parser.add_argument("--asymptotic_scale", help=asymptoticscalehelp,\
         type=float,default=0.05)

args = parser.parse_args()

## try:
##     component_prefix = sys.argv[1]
##     component_suffix = sys.argv[2]
## except IndexError:
##     component_prefix = 'fit_exp_'
##     component_suffix = '_unconstrained.dat'

component_prefix = args.prefix
component_suffix = args.suffix

# Filenames to read in 
exchange_file = component_prefix +  'exchange' + component_suffix
electrostatics_file = component_prefix +  'electrostatics' + component_suffix
induction_file = component_prefix +  'induction' + component_suffix
dhf_file = component_prefix +  'dhf' + component_suffix
dispersion_file = component_prefix +  'dispersion' + component_suffix
total_energy_file = component_prefix +  'total_energy' + component_suffix

## exchange_file = 'slater_exchange_unconstrained.dat'
## electrostatics_file = 'slater_electrostatics_unconstrained.dat'
## induction_file = 'slater_induction_unconstrained.dat'
## dhf_file = 'slater_dhf_unconstrained.dat'
## dispersion_file = 'slater_dispersion_unconstrained.dat'
## total_energy_file = 'slater_total_energy_unconstrained.dat'

###########################################################################
###########################################################################


###########################################################################
########################## Main Code ######################################

# Read data from each energy component

exchange = pd.read_csv(
                exchange_file,delim_whitespace=True,names=['qm','ff'],skiprows=1)
electrostatics = pd.read_csv(
                electrostatics_file,delim_whitespace=True,names=['qm','ff'],skiprows=1)
induction = pd.read_csv(
                induction_file,delim_whitespace=True,names=['qm','ff'],skiprows=1)
dhf = pd.read_csv(
                dhf_file,delim_whitespace=True,names=['qm','ff'],skiprows=1)
dispersion = pd.read_csv(
                dispersion_file,delim_whitespace=True,names=['qm','ff'],skiprows=1)
total_energy = pd.read_csv(
                total_energy_file,delim_whitespace=True,names=['qm','ff'],skiprows=1)

# Convert units from Hartrees to kjmol
au2kjmol = 2625.5
exchange *= au2kjmol
electrostatics *= au2kjmol
induction *= au2kjmol
dhf *= au2kjmol
dispersion *= au2kjmol
total_energy *= au2kjmol

eng = exchange['qm']
min_energy = np.min(eng)
max_energy = np.max(eng)

if args.asymptotic:
    scale = args.asymptotic_scale
else:
    scale = 1.0
xmax = scale*(max_energy - min_energy) + min_energy
xmin = min_energy

include_points = np.all([xmin*np.ones_like(eng) < eng,
                         xmax*np.ones_like(eng) > eng],axis=0)

exchange = exchange[include_points]
electrostatics = electrostatics[include_points]
induction = induction[include_points]
dhf = dhf[include_points]
dispersion = dispersion[include_points]
total_energy = total_energy[include_points]


# Set some global color preferences for how the graph's colors should look
sns.set_context('paper')
sns.axes_style("darkgrid")
sns.set_color_codes()
## vmax=max(abs(total_energy['qm']))
## vmin = -vmax
#vmax=max(total_energy['qm'])
vmax=args.cutoff
vmin=min(total_energy['qm'])
cmap = sns.cubehelix_palette(8, start=.5, rot=-.75,as_cmap=True, reverse=True)
order = np.argsort(total_energy['qm'])[::-1]
colors = total_energy['qm'][order]

# Overal graph layout and title
ncols=4
nrows=2
#fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(20,10))
fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(nrows, ncols, width_ratios=[1,1,0.5,2]) 
fig.suptitle('FF Fitting Quality Benchmarks',y=0.95,fontweight='bold', fontsize=14)
fig.text(0.30,0.075, 'Absolute Error in the Total Energy (kjmol, FF - SAPT)',ha='center', va='center')
fig.text(0.095,0.5, 'Absolute Error (FF - SAPT) in Component Energy (kjmol)',ha='center', va='center',rotation='vertical')

# Scale error plot axes
xy_max = 0
for component in electrostatics,exchange,dispersion,induction,dhf,total_energy:
    # Only scale axes based on net attractive energies
    #i = np.abs(total_energy['qm'][order]).argmin()
    ## y_qm = component['qm'][order][i:]
    ## y_ff = component['ff'][order][i:]
    x = total_energy['qm'][order]
    y_qm = component['qm'][order]
    y_ff = component['ff'][order]
    y = y_qm - y_ff
    ymax = np.amax(np.where(x < args.cutoff, np.abs(y), 0))
    #ymax = np.amax(np.where(x < 0, y, 0))
    xy_max = max(ymax,xy_max)
    xy_min = -xy_max

# Plot each energy component
count=0
titles=['Electrostatics','Exchange','','Induction + $\delta$HF','Dispersion',
        'Total Energy','Total Energy (attractive configurations)']
for component in electrostatics,exchange,None,(induction+dhf),dispersion,None:
    count += 1
    if count > 3:
        # Ignore last column for now
        #ax = plt.subplot(nrows*100 + ncols*10 + count + 1)
        ax = plt.subplot(gs[count])
        #ax = plt.subplot(gs[count],sharey=ax,sharex=ax)
    ## elif count > 1:
    ##     #ax = plt.subplot(nrows*100 + ncols*10 + count)
    ##     ax = plt.subplot(gs[count-1],sharey=ax,sharex=ax)
    else:
        ax = plt.subplot(gs[count-1])
    if component is None:
        # Plot colorbar instead of energy component
        ax.axis('off')
        continue

    x_qm = total_energy['qm'][order]
    x_ff = total_energy['ff'][order]
    y_qm = component['qm'][order]
    y_ff = component['ff'][order]

    x = - x_qm + x_ff
    y = - y_qm + y_ff

    # Scatterplot settings
    sc = plt.scatter(x,y,
            c=colors, vmin=vmin, vmax=vmax, cmap=cmap, s=25, lw =.0)
            #c=colors, vmin=vmin, vmax=vmax, cmap=cmap, s=25, lw =.75)

    # Axes scaling and title settings
    scale = 0.02
    ## xy_max = max(np.amax(np.abs(x)),np.amax(np.abs(y)))
    ## xy_min = -xy_max
    lims = [ xy_min - scale*abs(xy_max - xy_min), 
             xy_max + scale*abs(xy_max - xy_min) ]
    if titles[count-1] == titles[-1]:
        # Only plot attractive energies in this last plot
        lims[1] = args.cutoff
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(titles[count-1])

    # Plot y=x line
    plt.plot(lims, lims, 'k-', alpha=0.75)

    # Plot grid lines
    plt.axhline(0, color='k',linestyle='--',alpha=0.75)
    plt.axvline(0, color='k',linestyle='--',alpha=0.75)

cbaxes = fig.add_axes([0.48, 0.34, 0.01, 0.35]) 
cb = plt.colorbar(sc, cax = cbaxes, extend='max') 
cb.set_label('SAPT Total Energy (kjmol)')

# Finally, plot graph summarizing all errors contributing to errors in the
# total energy
ax = plt.subplot(gs[ncols-1])
x_qm = total_energy['qm'][order]
x_ff = total_energy['ff'][order]
count = 0
(xmin, xmax) = [np.amin(x_qm),args.cutoff]
(ymin, ymax) = [0,0]
marker = itertools.cycle(('v', '8', 'd', 'p', 's','o','+')) 
labels=itertools.cycle(
        ('Electrostatics','Exchange','Induction','$\delta$HF','Dispersion','Total Energy'))
for component in electrostatics,exchange,induction,dhf,dispersion,total_energy:
    y_qm = component['qm'][order]
    y_ff = component['ff'][order]
    x = x_qm
    y = - y_qm + y_ff
    ymin_i = np.amin(np.where(x < args.cutoff, y, 0))
    ymax_i = np.amax(np.where(x < args.cutoff, y, 0))
    (ymin,ymax) = (min(ymin,ymin_i),max(ymax,ymax_i))
    label=labels.next()
    if label == 'Total Energy':
        sc = plt.scatter(x,y,
                facecolors='none', edgecolors='k', s=25, lw = 0.75,
                #c=colors, vmin=vmin, vmax=vmax, cmap=cmap, s=25, lw =.75,
                label=label,zorder=10)
    else:
        plt.plot(x,y,marker = marker.next(), markersize=5, linestyle='', label=label)
ax.set_xlabel('SAPT Energy (kjmol)')
ax.set_ylabel('Absolute Error (FF Energy - QM energy) (kjmol)')
ax.set_title('Absolute Error in FF Fitting')
plt.axhline(0, color='k',linestyle='--',alpha=0.75,zorder=20)
handles, label = ax.get_legend_handles_labels()
ax.legend(handles, label)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

xlims = [ xmin - scale*abs(xmax - xmin), 
         xmax ]
ylims = [ ymin - scale*abs(ymax - ymin), 
         ymax + scale*abs(ymax - ymin) ]
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Total Attractive energies (for reference)
ax = plt.subplot(gs[ncols*nrows -1])
x = total_energy['qm'][order]
y = total_energy['ff'][order]
sc = plt.scatter(x,y,
        c=colors, vmin=vmin, vmax=vmax, cmap=cmap, s=25, lw =.0)
ymin = np.amin(np.where(x < args.cutoff, y, 0))
ymax = np.amax(np.where(x < args.cutoff, y, 0))
xy_min = min(ymin,np.amin(x))
xy_max = max(ymax,args.cutoff)
lims = [ xy_min - scale*abs(xy_max - xy_min), 
         args.cutoff + scale*abs(xy_max - xy_min) ]
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_title('Overall Fit Quality')
ax.set_xlabel('SAPT Total Energy (kjmol)')
ax.set_ylabel('FF Total Energy (kjmol)')
# Plot y=x line
plt.plot(lims, lims, 'k-', alpha=0.75)
# Shade in region to indicate +/- 10% error in ff
rel = 0.1
abs1 = 1 # 1 kJ/mol
#abs2 = 4.184 # 1 kcal/mol
abs2 = 2 # 2 kJ/mol
x1 = np.arange(lims[0],lims[1], 0 + 0.01)
#plt.fill_between(x1,x1-x1*rel,x1+x1*rel,zorder=0,alpha=0.25)
plt.fill_between(x1,x1-abs1,x1+abs1,zorder=0,alpha=0.2)
plt.fill_between(x1,x1-abs2,x1+abs2,zorder=0,alpha=0.1)


fig.savefig(component_prefix + 'sapt_errors.png',dpi=100,bbox_inches='tight')
if args.display:
    plt.show()

###########################################################################
###########################################################################
