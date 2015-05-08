## @package testPureScatteringProblem
#  Runs a pure scattering problem and compares to exact solution.

# add source directory to module search path
import sys
sys.path.append('../src')

from math import sqrt
from mesh import Mesh
from crossXInterface import CrossXInterface
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotScalarFlux, makeContinuousXPoints
import matplotlib.pyplot as plt

## Function to run a pure scattering problem and compare to exact solution.
def main():

   # set directions
   mu = {"-" : -1/sqrt(3), "+" : 1/sqrt(3)}

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
   cross_sects = [(CrossXInterface(sig_a,sig_s),CrossXInterface(sig_a,sig_s))
      for i in xrange(n_elems)]
   # sources
   Q_plus  = [(Q,Q) for i in xrange(mesh.n_elems)]
   Q_minus = [(Q,Q) for i in xrange(mesh.n_elems)]

   # compute LD solution
   psi_minus, psi_plus, E, F = radiationSolveSS(mesh,
                                                cross_sects,
                                                Q_minus,
                                                Q_plus,
                                                bound_curr_lt=inc_j_plus,
                                                bound_curr_rt=inc_j_minus)

   # get continuous x-points
   xlist = makeContinuousXPoints(mesh)

   # compute exact scalar flux solution
   scalar_flux_exact = [Q/(2*D)*(2*D*xR + xR*x - x*x) for x in xlist]

   # plot solutions
   plotScalarFlux(mesh,psi_minus,psi_plus,False,scalar_flux_exact)


# run main function
if __name__ == '__main__':
   main()
