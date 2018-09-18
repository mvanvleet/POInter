#!/usr/bin/env python
from pointer import FitFFParameters as Fit

settings = {}
settings['constraint_files'] = ['input/h2o.constraints']
settings['constrained_atomtypes'] = ['O']

settings_files = ['mastiff','config']


Fit(settings_files,**settings)

