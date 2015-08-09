## @package unittests.testRadTransient
#  Contains unit test to run a radiation transient to steady-state using BDF2

import sys
sys.path.append('../src')

import matplotlib as plt
from pylab import *
import numpy as np
from copy import deepcopy
import unittest
import operator

from mesh import Mesh
from crossXInterface import ConstantCrossSection
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotScalarFlux, computeScalarFlux
from utilityFunctions import computeDiscreteL1Norm
from radiation import Radiation
from transient import runLinearTransient
from transientSource import computeRadiationExtraneousSource

## Derived unittest class to run a transient radiation problem
#
class TestRadTransient(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_RadTransientBDF2(self):

       # create uniform mesh
       mesh = Mesh(50, 5.0)
   
       # compute uniform cross sections
       sig_s = 1.0
       sig_a = 2.0
       cross_sects = [(ConstantCrossSection(sig_s, sig_s+sig_a),
                       ConstantCrossSection(sig_s, sig_s+sig_a))
                       for i in xrange(mesh.n_elems)]
   
       # transient options
       dt = 0.1              # time step size
       t_start  = 0.0        # start time
       t_end = 10.0          # end time
       time_stepper = 'BDF2' # time-stepper
   
       # boundary fluxes
       psi_left  = 2.5
       psi_right = 2.2
   
       # create source function handles
       psim_src = lambda x,t: 2.4
       psip_src = lambda x,t: 2.4
   
       # IC
       n_dofs = mesh.n_elems * 4
       rad_IC = Radiation(np.zeros(n_dofs))

       # if run standalone, then be verbose
       if __name__ == '__main__':
          verbose = True
       else:
          verbose = False

       # run transient
       rad_new = runLinearTransient(
          mesh         = mesh,
          time_stepper = time_stepper,
          dt_option    = 'constant',
          dt_constant  = dt,
          t_start      = t_start,
          t_end        = t_end,
          psi_left     = psi_left,
          psi_right    = psi_right,
          cross_sects  = cross_sects,
          rad_IC       = rad_IC,
          psim_src     = psim_src,
          psip_src     = psip_src,
          verbose      = verbose)

       # compute the steady-state solution
       Q = computeRadiationExtraneousSource(psim_src, psip_src, mesh, t_start)
       rad_ss = radiationSolveSS(mesh, cross_sects, Q,
          bc_psi_right = psi_right, bc_psi_left = psi_left)

       # compute difference of transient and steady-state scalar flux
       phi_diff = [tuple(map(operator.sub, rad_new.phi[i], rad_ss.phi[i]))
          for i in xrange(len(rad_new.phi))]

       # compute discrete L1 norm of difference
       L1_norm_diff = computeDiscreteL1Norm(phi_diff)
   
       # assert that solution has converged to steady-state
       n_decimal_places = 12
       self.assertAlmostEqual(L1_norm_diff,0.0,n_decimal_places)
       
       # plot solutions if run standalone
       if __name__ == "__main__":
          plotScalarFlux(mesh, rad_new.psim, rad_new.psip,
             scalar_flux_exact=rad_ss.phi, exact_data_continuous=False)

    
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

