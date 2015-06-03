## @package testSSConvergence
#  Runs a diffusion problem and compares to exact solution.

# add source directory to module search path
import sys
sys.path.append('../src')

from math import sqrt, sinh, cosh
from scipy.integrate import quad # adaptive quadrature function
import unittest

from mesh import Mesh
from crossXInterface import ConstantCrossSection
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotScalarFlux, makeContinuousXPoints
from radUtilities import computeScalarFlux, extractAngularFluxes
from integrationUtilities import computeL1ErrorLD
from utilityFunctions import computeConvergenceRates, printConvergenceTable

## Derived unittest class to run a diffusion problem and compare to exact solution.
class TestSSConvergence(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_SSConvergence(self):

      # physics data
      sig_a = 0.25         # absorption cross section
      sig_s = 0.75         # scattering cross section
      sig_t = sig_s+sig_a  # total cross section
      D = 1.0/(3*sig_t)    # diffusion coefficient
      L = sqrt(D/sig_a)    # diffusion length
      xL = 0.0             # left boundary of domain
      xR = 3.0             # right boundary of domain
      inc_j_minus = 0      # incoming minus direction half-range current
      inc_j_plus  = 0      # incoming plus  direction half-range current
      Q = 1.0              # isotropic source
      A = 2.4084787907     # constant used in exact solution function
      B = -2.7957606046    # constant used in exact solution function
   
      # function for exact scalar flux solution
      def exactScalarFlux(x):
         return A*sinh(x/L) + B*cosh(x/L) + Q*L*L/D

      # number of elements
      n_elems = 10

      # number of refinement cycles
      n_cycles = 5

      # initialize lists for mesh size and L1 error for each cycle
      max_dx   = list()
      L1_error = list()

      # loop over refinement cycles
      for cycle in xrange(n_cycles):

         if __name__ == '__main__':
            print("Cycle %d of %d: n_elems = %d" % (cycle+1,n_cycles,n_elems))

         # mesh
         mesh = Mesh(n_elems,xR)
         # append max dx for this cycle to list
         max_dx.append(mesh.max_dx)
      
         # cross sections
         cross_sects = [(ConstantCrossSection(sig_s,sig_t),
                         ConstantCrossSection(sig_s,sig_t))
                         for i in xrange(n_elems)]

         # sources
         Q_iso  = [(0.5*Q) for i in xrange(mesh.n_elems*4)]
      
         # compute LD solution
         rad = radiationSolveSS(mesh,
                                cross_sects,
                                Q_iso,
                                bound_curr_lt=inc_j_plus,
                                bound_curr_rt=inc_j_minus)
      
         # compute L1 error
         L1_error.append(\
            computeL1ErrorLD(mesh, rad.phi, exactScalarFlux))

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

