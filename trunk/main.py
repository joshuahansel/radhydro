from src.Mesh import Mesh
from src.CrossXInterface import *
from src.radiationSolver import *
from src.plotS2Solution import plotS2Solution

## Main function.
#
#  This function does main stuff.
def main():

    # create uniform mesh
    mesh = Mesh(10, 5.)

    # compute cross sections
    cross_sects = [(CrossXInterface(1,3), CrossXInterface(1,3)) for i in xrange(mesh.n_elems)]

    # call radiation solver
    Q_plus  = [(0,0) for i in xrange(mesh.n_elems)]
    Q_minus = [(0,0) for i in xrange(mesh.n_elems)]
    psi_minus, psi_plus, E, F = radiationSolver(mesh, cross_sects, Q_minus, Q_plus)

    # plot S-2 solution
    plotS2Solution(mesh, psi_minus, psi_plus)


# run main function
if __name__ == "__main__":
    main()
