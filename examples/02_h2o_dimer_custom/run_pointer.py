#!/usr/bin/env python
from pointer import FitFFParameters as Fit

settings = {}

settings_files = ['mastiff','config']

Fit(settings_files,**settings)

