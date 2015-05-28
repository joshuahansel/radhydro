# add source directory to module search path
import sys
sys.path.append('src')

from mesh import Mesh
from crossXInterface import CrossXInterface
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotAngularFlux, plotScalarFlux

## Main function.
#
#  This function does main stuff.
def main():

    # create uniform mesh
    mesh = Mesh(100, 5.)

    # compute cross sections
    cross_sects = [(CrossXInterface(1,3), CrossXInterface(1,3))
       for i in xrange(mesh.n_elems)]

    # sources
    Q = [0.5 for i in xrange(mesh.n_elems*4)]

    # call radiation solver
    psi_minus, psi_plus, E, F = radiationSolveSS(mesh, cross_sects, Q)

    # plot solution
    plotScalarFlux(mesh, psi_minus, psi_plus)


# run main function
if __name__ == "__main__":
    main()
