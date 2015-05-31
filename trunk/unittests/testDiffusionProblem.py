## @package testDiffusionProblem
#  Runs a diffusion problem and compares to exact solution.

# add source directory to module search path
import sys
sys.path.append('../src')

from math import sqrt, sinh, cosh
from scipy.integrate import quad # adaptive quadrature function
import unittest

from mesh import Mesh
from crossXInterface import CrossXInterface
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotScalarFlux, makeContinuousXPoints
from radUtilities import computeScalarFlux, extractAngularFluxes
from integrationUtilities import computeL1ErrorLD

## Derived unittest class to run a diffusion problem and compare to exact solution.
class TestDiffusionProblem(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_DiffusionProblem(self):

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
   
      # number of elements
      n_elems = 50
      # mesh
      mesh = Mesh(n_elems,xR)
   
      # cross sections
      cross_sects = [(CrossXInterface(sig_a,sig_s),CrossXInterface(sig_a,sig_s))
         for i in xrange(n_elems)]
      # sources
      Q_iso  = [(0.5*Q) for i in xrange(mesh.n_elems*4)]
   
      # compute LD solution
      psi = radiationSolveSS(mesh,
                             cross_sects,
                             Q_iso,
                             bound_curr_lt=inc_j_plus,
                             bound_curr_rt=inc_j_minus)
   
      # get continuous x-points
      xlist = makeContinuousXPoints(mesh)
   
      # function for exact scalar flux solution
      def exactScalarFlux(x):
         A = 2.4084787907
         B = -2.7957606046
         return A*sinh(x/L)+B*cosh(x/L)+Q*L*L/D
   
      # compute exact scalar flux solution at each x-point
      scalar_flux_exact = [exactScalarFlux(x) for x in xlist]

      # extract angular fluxes
      psim, psip = extractAngularFluxes(psi,mesh)
   
      # plot solutions
      plotScalarFlux(mesh,psim,psip,save=True,filename='testDiffusion.pdf',
         scalar_flux_exact=scalar_flux_exact)
   
      # compute numerical scalar flux
      numerical_scalar_flux = computeScalarFlux(psim, psip)
   
      # compute L1 error
      L1_error = computeL1ErrorLD(mesh,numerical_scalar_flux,exactScalarFlux)
   
      # compute L1 norm of exact solution to be used as normalization constant
      L1_norm_exact = quad(exactScalarFlux, xL, xR)[0]

      # compute relative L1 error
      L1_relative_error = L1_error / L1_norm_exact

      # check that L1 error is small
      n_decimal_places = 3
      self.assertAlmostEqual(L1_relative_error,0.0,n_decimal_places)


# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

