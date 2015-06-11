## @package testRadSpatialConvergence
#  Tests spatial convergence rate of a transient MMS radiation problem.
#

import sys
sys.path.append('../src')

import numpy as np
from copy import deepcopy
import unittest
import operator # for adding tuples to each other elementwise
from math import pi, sin, cos
import sympy as sym

from mesh import Mesh
from crossXInterface import ConstantCrossSection
from radiationSolveSS import radiationSolveSS
from utilityFunctions import computeDiscreteL1Norm, getIndex
from radiationTimeStepper import takeRadiationStep
from radUtilities import mu, computeScalarFlux, extractAngularFluxes
import globalConstants as GC
from integrationUtilities import computeL1ErrorLD
from utilityFunctions import computeConvergenceRates, printConvergenceTable
from radiation import Radiation
from createMMSSourceFunctions import createMMSSourceFunctionsRadOnly
from transient import runLinearTransient

## Derived unittest class to run a transient radiation MMS problem
#
class TestRadSpatialConvergence(unittest.TestCase):
   def setUp(self):
      pass

   def tearDown(self):
      pass

   def test_RadSpatialConvergenceBE(self):
      time_stepper = 'BE'
      self.runConvergenceTest(time_stepper)

   def test_RadSpatialConvergenceCN(self):
      time_stepper = 'CN'
      self.runConvergenceTest(time_stepper)

   def test_RadSpatialConvergenceBDF2(self):
      time_stepper = 'BDF2'
      self.runConvergenceTest(time_stepper)

   def runConvergenceTest(self,time_stepper):

      # transient options
      dt_start = 0.01 # time step size
      t_start  = 0.0  # start time
      t_end    = 0.1  # end time
   
      # constant cross section values
      sig_s = 1.0
      sig_a = 2.0

      # boundary fluxes
      psi_left  = 0.0
      psi_right = 0.0
   
      # compute exact scalar flux solution
      def exactScalarFlux(x):
         return t_end*sin(pi*x) + 2.0*t_end*sin(pi*(1.0-x))

      # create symbolic expressions for MMS solution
      x, t, alpha = sym.symbols('x t alpha')
      psim = 2*t*sym.sin(sym.pi*(1-x))
      psip = t*sym.sin(sym.pi*x)

      # number of elements
      n_elems = 10

      # number of refinement cycles
      n_cycles = 5

      # initialize lists for mesh size and L1 error for each cycle
      max_dx   = list()
      L1_error = list()

      # print header
      if __name__ == '__main__':
         print('\n%s:' % time_stepper)

      # loop over refinement cycles
      for cycle in xrange(n_cycles):

         if __name__ == '__main__':
            print("\nCycle %d of %d: n_elems = %d" % (cycle+1,n_cycles,n_elems))

         # create uniform mesh
         mesh = Mesh(n_elems, 1.0)
         # append max dx for this cycle to list
         max_dx.append(mesh.max_dx)
     
         # compute uniform cross sections
         cross_sects = [(ConstantCrossSection(sig_s, sig_s+sig_a),
                         ConstantCrossSection(sig_s, sig_s+sig_a))
                         for i in xrange(mesh.n_elems)]
  
         # create source function handles
         psim_src, psip_src = createMMSSourceFunctionsRadOnly(
            psim = psim,
            psip = psip,
            sigma_s_value = sig_s,
            sigma_a_value = sig_a)
    
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
            dt_constant  = dt_start,
            t_start      = t_start,
            t_end        = t_end,
            psi_left     = psi_left,
            psi_right    = psi_right,
            cross_sects  = cross_sects,
            rad_IC       = rad_IC,
            psim_src     = psim_src,
            psip_src     = psip_src,
            verbose      = verbose)

         # compute L1 error
         L1_error.append(\
            computeL1ErrorLD(mesh, rad_new.phi, exactScalarFlux))

         # double number of elements for next cycle
         n_elems *= 2

      # compute convergence rates
      rates = computeConvergenceRates(max_dx,L1_error)

      # print convergence table if not being run in suite
      if __name__ == '__main__':
         printConvergenceTable(max_dx,L1_error,rates=rates,
            dx_desc='dx',err_desc='L1')

      # check that final rate is approximately 2nd order
      self.assert_(rates[n_cycles-2] > 1.95)
       
    
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

