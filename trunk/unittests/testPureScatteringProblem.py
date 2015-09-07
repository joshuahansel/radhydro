## @package unittests.testPureScatteringProblem
#  Runs a pure scattering problem and compares to exact solution.

# add source directory to module search path
import sys
sys.path.append('../src')

from math import sqrt
from scipy.integrate import quad # adaptive quadrature function
import unittest

from mesh import Mesh
from crossXInterface import ConstantCrossSection
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotScalarFlux, makeContinuousXPoints
from radUtilities import computeScalarFlux, extractAngularFluxes
from integrationUtilities import computeL1ErrorLD
from radBC import RadBC

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
   
      # radiation BC
      rad_BC = RadBC(mesh, "dirichlet", psi_left=2*inc_j_plus, psi_right=2*inc_j_minus)

      # cross sections
      cross_sects = [(ConstantCrossSection(sig_s,sig_t),
                      ConstantCrossSection(sig_s,sig_t))
                      for i in xrange(n_elems)]
      # sources
      Q_src  = [0.5*Q for i in xrange(mesh.n_elems*4)]
   
      # compute LD solution
      rad = radiationSolveSS(mesh,
                             cross_sects,
                             Q_src,
                             rad_BC=rad_BC)
   
      # get continuous x-points
      xlist = makeContinuousXPoints(mesh)
   
      # function for the exact scalar flux solution
      def exactScalarFlux(x):
         return Q/(2*D)*(2*D*xR + xR*x - x*x)
   
      # compute exact scalar flux solution
      scalar_flux_exact = [exactScalarFlux(x) for x in xlist]
   
      # plot solutions
      if __name__ == '__main__':
         plotScalarFlux(mesh, rad.psim, rad.psip, scalar_flux_exact=scalar_flux_exact)

      # compute L1 error
      L1_error = computeL1ErrorLD(mesh, rad.phi, exactScalarFlux)
   
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

