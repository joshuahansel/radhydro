## @package testRadTransient
#  Contains unit test to run a radiation transient to steady-state using BDF2
#

import sys
sys.path.append('../src')

import matplotlib as plt
from pylab import *
import numpy as np
from copy import deepcopy
import unittest
import operator # for adding tuples to each other elementwise

from mesh import Mesh
from crossXInterface import ConstantCrossSection
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotAngularFlux, plotScalarFlux, computeScalarFlux
from utilityFunctions import computeDiscreteL1Norm
from radiationTimeStepper import RadiationTimeStepper
from radUtilities import extractAngularFluxes
from radiation import Radiation

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
       t  = 0.0              # begin time
       t_end = 10.0          # end time
       time_stepper = 'BDF2' # time-stepper
   
       # boundary fluxes
       psi_left  = 2.5
       psi_right = 2.2
   
       # create the steady-state source
       n_dofs = mesh.n_elems * 4
       Q = 2.4 * np.ones(n_dofs)
   
       # compute the steady-state solution
       rad_ss = radiationSolveSS(mesh, cross_sects, Q,
          bc_psi_right = psi_right, bc_psi_left = psi_left)
   
       # run transient solution from arbitrary IC, such as zero
       rad_old   = Radiation(np.zeros(n_dofs))
       rad_older = deepcopy(rad_old)
   
       # create time-stepper
       radiation_time_stepper = RadiationTimeStepper(mesh, time_stepper)

       # transient loop
       transient_incomplete = True # boolean flag signalling end of transient
       while transient_incomplete:
   
           # adjust time step size if necessary
           if t + dt >= t_end:
              dt = t_end - t
              t = t_end
              transient_incomplete = False # signal end of transient
           else:
              t += dt

           # take radiation step
           rad = radiation_time_stepper.takeStep(
              dt            = dt,
              bc_flux_left  = psi_left,
              bc_flux_right = psi_right,
              cx_older      = cross_sects,
              cx_old        = cross_sects,
              cx_new        = cross_sects,
              rad_older     = rad_older,
              rad_old       = rad_old,
              Q_older       = Q,
              Q_old         = Q,
              Q_new         = Q)

           # compute difference of transient and steady-state scalar flux
           phi_diff = [tuple(map(operator.sub, rad.phi[i], rad_ss.phi[i]))
              for i in xrange(len(rad.phi))]

           # compute discrete L1 norm of difference
           L1_norm_diff = computeDiscreteL1Norm(phi_diff)

           # print each time step if run standalone
           if __name__ == '__main__':
              print("t = %0.3f -> %0.3f: L1 norm of diff with steady-state: %7.3e"
                 % (t-dt,t,L1_norm_diff))
   
           # save oldest solutions
           rad_older = deepcopy(rad_old)
   
           # save old solutions
           rad_old = deepcopy(rad)
   
       # plot solutions if run standalone
       if __name__ == "__main__":
          plotScalarFlux(mesh, rad.psim, rad.psip, scalar_flux_exact=rad_ss.phi,
             exact_data_continuous=False)

       # assert that solution has converged
       n_decimal_places = 12
       self.assertAlmostEqual(L1_norm_diff,0.0,n_decimal_places)
       
    
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

