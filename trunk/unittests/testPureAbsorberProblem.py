## @package testPureAbsorberProblem
#  Runs a pure absorber problem and compares to exact solution.

# add source directory to module search path
import sys
sys.path.append('../src')

from math import sqrt, exp
from scipy.integrate import quad # adaptive quadrature function
import unittest

from mesh import Mesh
from crossXInterface import CrossXInterface
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotAngularFlux, makeContinuousXPoints
from radUtilities import mu, computeScalarFlux, extractAngularFluxes
from integrationUtilities import computeL1ErrorLD

## Derived unittest class to run a pure absorber problem and compare to
#  exact solution.
class TestPureAbsorberProblem(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_PureAbsorberProblem(self):

      # physics data
      sig_t = 0.1    # total cross section
      L = 10.0       # domain length
      inc_minus = 10 # isotropic incoming angular flux for minus direction
      inc_plus = 20  # isotropic incoming angular flux for plus  direction
   
      j_minus = 0.5*inc_minus # incoming current in minus direction
      j_plus  = 0.5*inc_plus  # incoming current in plus  direction
   
      # number of elements
      n_elems = 50
      # mesh
      mesh = Mesh(n_elems,L)
   
      # cross sections
      cross_sects = [(CrossXInterface(sig_t,0),CrossXInterface(sig_t,0))
         for i in xrange(n_elems)]
      # sources
      Q  = [0.0 for i in xrange(mesh.n_elems*4)]
   
      # compute LD solution
      psi = radiationSolveSS(mesh,
                             cross_sects,
                             Q,
                             bound_curr_lt=j_plus,
                             bound_curr_rt=j_minus)
   
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

      # extract angular fluxes from solution vector
      psim, psip = extractAngularFluxes(psi,mesh)
   
      # plot solutions
      plotAngularFlux(mesh,psim,psip,
         save=True,filename='testPureAbsorber.pdf',
         psi_minus_exact=psim_exact, psi_plus_exact=psip_exact)

      # compute numerical and exact scalar flux
      numerical_scalar_flux = computeScalarFlux(psim,psip)
   
      # compute L1 error
      L1_error = computeL1ErrorLD(mesh,numerical_scalar_flux,exactScalarFlux)
   
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

