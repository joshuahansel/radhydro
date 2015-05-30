## @package testRadTransient
#  Contains unit test to run a radiation transient to steady-state
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
from crossXInterface import CrossXInterface
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotAngularFlux, plotScalarFlux, computeScalarFlux
from utilityFunctions import computeDiscreteL1Norm
from radiationTimeStepper import RadiationTimeStepper

## Derived unittest class to run a transient radiation problem
#
class TestRadTransient(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_RadTransient(self):

       # create uniform mesh
       mesh = Mesh(50, 5.0)
   
       # compute uniform cross sections
       sig_s = 1.0
       sig_a = 2.0
       cross_sects = [(CrossXInterface(sig_a, sig_s), CrossXInterface(sig_a, sig_s))
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
       psim_ss, psip_ss, E, F = radiationSolveSS(mesh, cross_sects, Q,
          bc_psi_right = psi_right, bc_psi_left = psi_left)
   
       # compute steady state scalar flux
       phi_ss = computeScalarFlux(psip_ss, psim_ss)
   
       # run transient solution from arbitrary IC, such as zero
       psip_old   = [(0.0,0.0) for i in range(mesh.n_elems)]
       psim_old   = deepcopy(psip_old)
       psip_older = deepcopy(psip_old)
       psim_older = deepcopy(psip_old)
       E_old      = deepcopy(psip_old)
       E_older    = deepcopy(psip_old)
   
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
           psim, psip, E, F = radiation_time_stepper.takeStep(
              dt            = dt,
              bc_flux_left  = psi_left,
              bc_flux_right = psi_right,
              cx_older      = cross_sects,
              cx_old        = cross_sects,
              cx_new        = cross_sects,
              psim_older    = psim_older,
              psip_older    = psip_older,
              psim_old      = psim_old,
              psip_old      = psip_old,
              E_older       = E_older,
              E_old         = E_old,
              Q_older       = Q,
              Q_old         = Q,
              Q_new         = Q)

           # compute scalar flux
           phi = computeScalarFlux(psip, psim)

           # compute difference of transient and steady-state scalar flux
           phi_diff = [tuple(map(operator.sub, phi[i], phi_ss[i]))
              for i in xrange(len(phi))]

           # compute discrete L1 norm of difference
           L1_norm_diff = computeDiscreteL1Norm(phi_diff)

           # print each time step if run standalone
           if __name__ == '__main__':
              print("t = %0.3f -> %0.3f: L1 norm of diff with steady-state: %7.3e"
                 % (t-dt,t,L1_norm_diff))
   
           # save oldest solutions
           psip_older = deepcopy(psip_old)
           psim_older = deepcopy(psim_old)
           E_older    = deepcopy(E_old)
   
           # save old solutions
           psip_old = deepcopy(psip)
           psim_old = deepcopy(psim)
           E_old    = deepcopy(E)
   
       # plot solutions if run standalone
       if __name__ == "__main__":
          plotScalarFlux(mesh, psim, psip, scalar_flux_exact=phi_ss,
             exact_data_continuous=False)

       # assert that solution has converged
       n_decimal_places = 12
       self.assertAlmostEqual(L1_norm_diff,0.0,n_decimal_places)
       
    
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

