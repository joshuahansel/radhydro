## @package unittests.testPureAbsorberProblem
#  Runs a pure absorber problem and compares to exact solution.

# add source directory to module search path
import sys
sys.path.append('../src')

from math import sqrt, exp
from scipy.integrate import quad # adaptive quadrature function
import unittest

from mesh import Mesh
from crossXInterface import ConstantCrossSection
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotAngularFlux, makeContinuousXPoints
from radUtilities import mu, computeScalarFlux, extractAngularFluxes
from integrationUtilities import computeL1ErrorLD
from radBC import RadBC

## Derived unittest class to run a pure absorber problem and compare to
#  exact solution.
class TestPureAbsorberProblem(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_PureAbsorberProblem(self):

      # number of elements
      n_elems = 50
      # mesh
      L = 10.0       # domain length
      mesh = Mesh(n_elems,L)
   
      # physics data
      sig_t = 0.1    # total cross section
      sig_s = 0.0    # scattering cross section
      inc_minus = 10 # isotropic incoming angular flux for minus direction
      inc_plus  = 20 # isotropic incoming angular flux for plus  direction
      rad_BC    = RadBC(mesh, "dirichlet", psi_left=inc_plus, psi_right=inc_minus)
   
      # cross sections
      cross_sects = [(ConstantCrossSection(sig_s,sig_t),
                      ConstantCrossSection(sig_s,sig_t))
                      for i in xrange(n_elems)]
      # sources
      Q  = [0.0 for i in xrange(mesh.n_elems*4)]
   
      # compute LD solution
      rad = radiationSolveSS(mesh,
                             cross_sects,
                             Q,
                             rad_BC=rad_BC)
   
      # get continuous x-points
      xlist = makeContinuousXPoints(mesh)
   
      # exact solution functions
      def exactPsiMinus(x):
         return inc_minus*exp(-sig_t/mu["-"]*(x-L))
      def exactPsiPlus(x):
         return inc_plus *exp(-sig_t/mu["+"]*x)
      def exactScalarFlux(x):
         return exactPsiMinus(x) + exactPsiPlus(x)

      # compute exact solutions
      psim_exact = [exactPsiMinus(x) for x in xlist]
      psip_exact = [exactPsiPlus(x)  for x in xlist]
      exact_scalar_flux = [psi_m+psi_p for psi_m, psi_p
         in zip(psim_exact, psip_exact)]

      # plot solutions
      if __name__ == '__main__':
         plotAngularFlux(mesh, rad.psim, rad.psip,
            psi_minus_exact=psim_exact, psi_plus_exact=psip_exact)

      # compute L1 error
      L1_error = computeL1ErrorLD(mesh,rad.phi,exactScalarFlux)
   
      # compute L1 norm of exact solution to be used as normalization constant
      L1_norm_exact = quad(exactScalarFlux, 0.0, L)[0]

      # compute relative L1 error
      L1_relative_error = L1_error / L1_norm_exact

      # check that L1 error is small
      n_decimal_places = 3
      self.assertAlmostEqual(L1_relative_error,0.0,n_decimal_places)


# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

