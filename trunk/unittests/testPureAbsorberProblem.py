# add parent directory to module search path
import sys
sys.path.append('..')

from src.Mesh import Mesh
from src.CrossXInterface import *
from src.radiationSolver import *
from src.plotS2Solution import *
import matplotlib.pyplot as plt

def main():

   # set directions
   mu = {"-" : -1/math.sqrt(3), "+" : 1/math.sqrt(3)}

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
   psi_minus_exact = [inc_minus*math.exp(-sig_t/mu["-"]*(x-L))
      for x in xlist]
   psi_plus_exact  = [inc_plus *math.exp(-sig_t/mu["+"]*x)
      for x in xlist]

   # plot solutions
   plotS2Solution(mesh, psi_minus, psi_plus, False, psi_minus_exact, psi_plus_exact)


# run main function
if __name__ == '__main__':
   main()
