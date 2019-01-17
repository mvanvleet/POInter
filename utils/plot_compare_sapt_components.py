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
from matplotlib._png import read_png
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1 import ImageGrid
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
fadehelp="Weight the opacity of each point depending on the SAPT Total Energy."
asymptotichelp="""Only display the scale_asymptotic fraction of points with the
lowest SAPT exchange energies (which serve as a proxy for the most asymptotic
points)"""
asymptoticscalehelp="Dictates the fraction of points that will be displayed.  See asymptotic above."

#parser.add_argument("energy_file", type=str, help=energyhelp)
parser.add_argument("-p","--prefixes", help="prefixes", nargs="*", default=['fit_exp_'])
parser.add_argument("-s","--suffixes", help="suffixes", nargs="*", default='DEFAULT_SUFFIX')

parser.add_argument("--display", help=displayhelp,\
         action="store_true", default=False)

parser.add_argument("-a","--asymptotic", help=asymptotichelp,\
         action="store_true", default=False)
parser.add_argument("-x","--scale_asymptotic", help=asymptoticscalehelp,\
         type=float,default=0.05)

parser.add_argument("--fade", help=fadehelp,\
         action="store_true", default=False)

args = parser.parse_args()

if args.suffixes == 'DEFAULT_SUFFIX':
    args.suffixes = ['_unconstrained.dat']*len(args.prefixes)

component_prefixes = args.prefixes
component_suffixes = args.suffixes

## component_prefixes = ['isa_constrained_exp_','saptff_constrained_exp_']
## component_suffixes = ['_unconstrained.dat','_unconstrained.dat']

## if component_suffixes == []:
##     component_prefixes = ['fit_exp_']
##     component_suffixes = ['_unconstrained.dat']


###########################################################################
###########################################################################


###########################################################################
########################## Main Code ######################################

# Read data from each energy component
# Set some global color preferences for how the graph's colors should look
## sns.set_context('paper')
## sns.axes_style("darkgrid")
## sns.set_style("ticks")
## sns.set_color_codes()
## #palette = itertools.cycle(sns.color_palette())
## #palette = sns.color_palette()
sns.set(context='talk',font_scale=1.1)
palette = sns.color_palette("Set2")
#sns.set_style('darkgrid',
sns.set_style("ticks", 
                  {
                  'ytick.direction': 'in', 'xtick.direction': 'in',
                  'font.sans-serif': ['CMU Sans Serif','Helectiva','Arial', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
                  }
                    )

# Overal graph layout and title
ncols=4
nrows=2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(18,10))
fig.text(0.5,0.055, 'SAPT Energy (kJ/mol)',ha='center',
        va='center',
        fontweight='semibold',
        fontsize=24)
fig.text(0.080,0.5, 'FF Energy (kJ/mol)',
        ha='center', va='center',
        rotation='vertical',
        fontweight='semibold',
        fontsize=24)

first=True
labels = []
label_texts = itertools.cycle(['Slater-ISA FF','Born-Mayer-IP FF'])
zorder = itertools.cycle([8,9,10])
m_values = []
b_values = []
for i, (component_prefix,component_suffix) in enumerate(zip(component_prefixes,component_suffixes)):
    last = (component_prefix == component_prefixes[-1])
    last = (last and component_suffix == component_suffixes[-1])
    z=zorder.next()

    color = palette[i]

    # Filenames to read in 
    exchange_file = component_prefix +  'exchange' + component_suffix
    electrostatics_file = component_prefix +  'electrostatics' + component_suffix
    induction_file = component_prefix +  'induction' + component_suffix
    dhf_file = component_prefix +  'dhf' + component_suffix
    dispersion_file = component_prefix +  'dispersion' + component_suffix
    total_energy_file = component_prefix +  'total_energy' + component_suffix

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

    # Convert units from Hartrees to kJ/mol
    au2kJmol = 2625.5
    exchange *= au2kJmol
    electrostatics *= au2kJmol
    induction *= au2kJmol
    dhf *= au2kJmol
    dispersion *= au2kJmol
    total_energy *= au2kJmol


    # Plot each energy component
    count=0
    titles=['Electrostatics','Exchange','Dispersion','','Induction','$\delta$HF',
            'Total Energy','Total Energy (Attractive Region)']


    #order = np.argsort(total_energy['qm'])[::-1]
    eng = exchange['qm']
    min_energy = np.min(eng)
    max_energy = np.max(eng)

    if args.asymptotic:
        scale = args.scale_asymptotic
    else:
        scale = 1.0
    xmax = scale*(max_energy - min_energy) + min_energy
    xmin = min_energy

    include_points = np.all([xmin*np.ones_like(eng) <= eng,
                             xmax*np.ones_like(eng) >= eng],axis=0)

    for component in electrostatics,exchange,dispersion,None,induction,dhf,total_energy,total_energy:
        count += 1
        ax = plt.subplot(nrows*100 + ncols*10 + count)

        if titles[count-1] == titles[ncols-1]:
            # Plot colorbar instead of energy component
            ax.axis('off')

            label_text = component_prefix.rstrip('_') + ' ' +\
                    component_suffix.lstrip('_').replace('.dat','')
            labels.append(patches.Patch(color=color, label=label_text))
            ax.legend(handles=labels, 
                fontsize=22,
                bbox_to_anchor=(0.93,0.79),
                bbox_transform=ax.figure.transFigure)

            m_values.append('')
            b_values.append('')

            continue


        x = component['qm'][include_points]
        y = component['ff'][include_points]

        ## x = component['qm'][order]
        ## y = component['ff'][order]

        if args.fade and not 'Total Energy' in titles[count-1]:
            etot = total_energy['qm']
            alpha = np.where( etot < 0, 0.6, 0.1 )
            alpha_color = np.asarray([color + (a,) for a in alpha])

            sc = ax.scatter(x,y, color=alpha_color, 
                                s=25, lw =.75, 
                                #facecolors='none',
                                edgecolors='none',
                                zorder=z)

        else:
            sc = ax.scatter(x,y, color=color, 
                                s=25, lw =.75, 
                                #facecolors='none',
                                edgecolors='none',
                                alpha=0.5,zorder=z)

        # Axes scaling and title settings
        scale = 0.02
        xy_min = min(np.amin(x),np.amin(y))
        xy_max = max(np.amax(x),np.amax(y))
        if titles[count-1] == titles[-1]:
            xy_max = 0.0
        m, b = np.polyfit(x, y, 1)
        m_values.append(m)
        b_values.append(b)
            ## ax.plot(np.array(lims),m*np.array(lims)+b,':',color=color)
        lims = [ xy_min - scale*abs(xy_max - xy_min), 
                 xy_max + scale*abs(xy_max - xy_min) ]
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        if 'Total Energy' in titles[count-1]:
            ax.set_title(titles[count-1],fontweight='bold')
        else:
            ax.set_title(titles[count-1])


        # Shade in region to indicate +/- 10% error in ff
        if last:
            # Plot y=x line
            plt.plot(lims, lims, 'k-', alpha=0.75, zorder=20)
            rel = 0.1
            lims = ax.get_ylim()
            x1 = np.arange(lims[0],lims[1],0.01)
            if titles[count-1] != titles[-1]:
                plt.fill_between(x1,x1-x1*rel,x1+x1*rel,
                        edgecolor='None',
                        zorder=0,alpha=0.1,color='k')
            else:
                # Plot +/- 1 kJ/mol errors
                kjmol = 1.0
                plt.fill_between(x1,x1-kjmol,x1+kjmol,
                        edgecolor='None',
                        zorder=0,alpha=0.1,color='k')
        ## kcal = 0.627
        ## plt.fill_between(x1,x1-kcal,x1+kcal,zorder=0,alpha=0.25)


        ## # Add RMS Error legends to each plot
        ## rms_error = np.sqrt(np.sum((x-y)**2)/len(x))
        ## x1 = total_energy['qm']
        ## attractive_rms_error = np.sqrt(np.sum(
        ##                             np.select([x1 < 0], [(x-y)**2])/len(x1[x1<0])))
        ## textstr = 'RMS Error: {:6.3f} kJ/mol\nRMS Error (excl. repulsive points): {:6.3f} kJ/mol'\
        ##             .format(rms_error,attractive_rms_error)
        ## props = dict(facecolor=sns.axes_style()['axes.facecolor'], lw=0, alpha=1)
        ## ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=8,
        ##                 verticalalignment='bottom', horizontalalignment='right',bbox=props)

        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(7))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(7))

    first = False

# Plot line of best fit in both the total energy and zoomed total energy views
subplots = xrange(1,nrows*ncols+1)
ncomponents = 8
for count in subplots:
    ax = plt.subplot(nrows*100 + ncols*10 + count)
    lims = ax.get_xlim()
    labels = []
    for i, (m,b) in enumerate(zip(
            m_values[count-1::ncomponents],
            b_values[count-1::ncomponents])):

        if m == '':
            break
        if b > 0:
            textstr = '$y = {:6.3f}x + {:6.3f}$'.format(m,b)
        else:
            textstr = '$y = {:6.3f}x - {:6.3f}$'.format(m,abs(b))

        labels.append(textstr)
        ax.plot(np.array(lims),m*np.array(lims)+b,':',color=palette[i],label=textstr)
        

        # Add line of best fit legends to each plot
        props = dict(facecolor=sns.axes_style()['axes.facecolor'], lw=0, alpha=1)
        ax.text(0.98, 0.10-0.08*i, textstr, transform=ax.transAxes, 
                fontsize=15,
                color = palette[i],
                verticalalignment='bottom', horizontalalignment='right',bbox=props)
    #ax.legend(loc=4)

if args.asymptotic:
    outtitle='asymptotic_sapt_comparison.png'
else:
    outtitle='sapt_comparison.png'
fig.savefig(outtitle,bbox_inches='tight',dpi=100)
if args.display:
    plt.show()

###########################################################################
###########################################################################
