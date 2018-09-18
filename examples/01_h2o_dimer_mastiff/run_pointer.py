#!/usr/bin/env python
from pointer import FitFFParameters as Fit

settings = {}
settings['mon1'] = 'h2o'
settings['mon2'] = 'h2o'
settings['ofile_prefix'] = 'output/fit_exp_'
settings['ofile_suffix'] = '_unconstrained'

settings_files = ['mastiff']

Fit(settings_files,**settings)

