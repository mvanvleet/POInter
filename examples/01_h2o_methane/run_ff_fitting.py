#!/usr/bin/env python
from pointer.fit_ff_parameters import FitFFParameters

sapt_file='h2o_methane.sapt'
param_file='h2o_methane.param'
coeffs_file='coeffs.out'

FitFFParameters(sapt_file, param_file, coeffs_file,
        slater_correction=True, fit_bii=True,
        aij_combination_rule='geometric',
        bij_combination_rule='geometric_mean',
        cij_combination_rule='geometric')
