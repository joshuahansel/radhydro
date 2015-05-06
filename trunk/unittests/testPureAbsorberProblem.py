## @package testPureAbsorberProblem
#  Runs a pure absorber problem and compares to exact solution.

# add source directory to module search path
import sys
sys.path.append('../src')

from math import sqrt, exp
from mesh import Mesh
from crossXInterface import CrossXInterface
from radiationSolver import radiationSolver
from plotUtilities import plotAngularFlux, makeContinuousXPoints
import matplotlib.pyplot as plt

## Function to run a pure absorber problem and compare to exact solution.
def main():

   # set directions
   mu = {"-" : -1/sqrt(3), "+" : 1/sqrt(3)}

   # physics data
   sig_t = 1.0   # total cross section
   L = 1.0       # domain length
   inc_minus = 3 # incoming angular flux for minus direction
   inc_plus = 2  # incoming angular flux for plus direction

   # number of elements
   n_elems = 50
   # mesh
   mesh = Mesh(n_elems,L)

   # cross sections
   cross_sects = [(CrossXInterface(sig_t,0),CrossXInterface(sig_t,0))
      for i in xrange(n_elems)]
   # sources
   Q_plus  = [(0,0) for i in xrange(mesh.n_elems)]
   Q_minus = [(0,0) for i in xrange(mesh.n_elems)]

   # compute LD solution
   psi_minus, psi_plus, E, F = radiationSolver(mesh,
                                               cross_sects,
                                               Q_minus,
                                               Q_plus,
                                               bound_curr_lt=inc_plus,
                                               bound_curr_rt=inc_minus)

   # get continuous x-points
   xlist = makeContinuousXPoints(mesh)

   # compute exact solutions
   psi_minus_exact = [inc_minus*exp(-sig_t/mu["-"]*(x-L))
      for x in xlist]
   psi_plus_exact  = [inc_plus *exp(-sig_t/mu["+"]*x)
      for x in xlist]

   # plot solutions
   plotAngularFlux(mesh,psi_minus,psi_plus,False,psi_minus_exact,psi_plus_exact)


# run main function
if __name__ == '__main__':
   main()