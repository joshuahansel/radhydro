## @package testPureScatteringProblem
#  Runs a pure scattering problem and compares to exact solution.

# add source directory to module search path
import sys
sys.path.append('../src')

from math import sqrt
from scipy.integrate import quad # adaptive quadrature function
import unittest

from mesh import Mesh
from crossXInterface import CrossXInterface
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotScalarFlux, makeContinuousXPoints
from radUtilities import computeScalarFlux, extractAngularFluxes
from integrationUtilities import computeL1ErrorLD

## Derived unittest class to run a pure scattering problem and compare to
#  exact solution.
class TestPureScatteringProblem(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_PureScatteringProblem(self):

      # physics data
      sig_a = 0.0         # absorption cross section
      sig_s = 1.0         # scattering cross section
      sig_t = sig_s+sig_a # total cross section
      xL = 0.0            # left boundary of domain
      width = 100.0       # domain width
      xR = xL + width     # right boundary of domain
      inc_j_minus = 1     # incoming minus direction half-range current
      inc_j_plus  = 3     # incoming plus  direction half-range current
      Q = 1.0             # isotropic source
      D = 1.0/(3*sig_t)   # diffusion coefficient
   
      # number of elements
      n_elems = 50
      # mesh
      mesh = Mesh(n_elems,width,xL)
   
      # cross sections
      cross_sects = [(CrossXInterface(sig_s,sig_t),CrossXInterface(sig_s,sig_t))
         for i in xrange(n_elems)]
      # sources
      Q_src  = [0.5*Q for i in xrange(mesh.n_elems*4)]
   
      # compute LD solution
      psi = radiationSolveSS(mesh,
                             cross_sects,
                             Q_src,
                             bound_curr_lt=inc_j_plus,
                             bound_curr_rt=inc_j_minus)
   
      # get continuous x-points
      xlist = makeContinuousXPoints(mesh)
   
      # function for the exact scalar flux solution
      def exactScalarFlux(x):
         return Q/(2*D)*(2*D*xR + xR*x - x*x)
   
      # compute exact scalar flux solution
      scalar_flux_exact = [exactScalarFlux(x) for x in xlist]
   
      # extract angular fluxes from solution vector
      psim, psip = extractAngularFluxes(psi,mesh)

      # plot solutions
      plotScalarFlux(mesh, psim, psip, save=True,
         filename='testPureScattering.pdf', scalar_flux_exact=scalar_flux_exact)

      # compute numerical scalar flux
      numerical_scalar_flux = computeScalarFlux(psim, psip)
   
      # compute L1 error
      L1_error = computeL1ErrorLD(mesh,numerical_scalar_flux,exactScalarFlux)
   
      # compute L1 norm of exact solution to be used as normalization constant
      L1_norm_exact = quad(exactScalarFlux, xL, xR)[0]

      # compute relative L1 error
      L1_relative_error = L1_error / L1_norm_exact

      # check that L1 error is small
      n_decimal_places = 2
      self.assertAlmostEqual(L1_relative_error,0.0,n_decimal_places)


# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

