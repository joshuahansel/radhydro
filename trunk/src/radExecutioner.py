## @package src.hydroExecutioner
#  Runs a hydrodynamics problem.
import sys
sys.path.append('src')

from mesh import Mesh
from crossXInterface import CrossXInterface
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotAngularFlux, plotScalarFlux
from sourceHandlers import * 
import numpy as np

## Main executioner for Radiation solve. Currently in a testing state.
def solveRadProblem():

    #-------------------------------------------------------------------------------
    # create uniform mesh
    mesh = Mesh(100, 5.)
    sig_s = 0.6
    sig_a = 2.0

    # compute cross sections
    cross_sects = [(CrossXInterface(sig_a, sig_s), CrossXInterface(sig_a, sig_s)) for i in xrange(mesh.n_elems)]
    dt = 0.01
    t  = 0.0

    #consistent BC's, eventually lets just switch to psi's and forget the currents
    Q_iso = 5.
    from math import pi
    psi_left = Q_iso/(2.*sig_a)
    psi_right = Q_iso/(2.*sig_a)
    bound_curr_lt = psi_left*0.5
    bound_curr_rt = psi_right*0.5

    # call radiation solver to get the steady state solution
    Q = []
    for i in xrange(mesh.n_elems):
    
        Q_new = [0.0 for i in range(4)]
        Q_new[getLocalIndex("L","+")] = Q_iso/2.
        Q_new[getLocalIndex("R","+")] = Q_iso/2.
        Q_new[getLocalIndex("L","-")] = Q_iso
        Q_new[getLocalIndex("R","-")] = Q_iso
        Q += Q_new


    Q = np.array(Q)
    psi_minusSS, psi_plusSS, E, F = radiationSolveSS(mesh, cross_sects, Q,
            bc_psi_right = psi_right, bc_psi_left = psi_left)

    plotScalarFlux(mesh, psi_minusSS, psi_plusSS)

    print psi_minusSS, psi_plusSS
    diag_terms = {"CN":0.5, "BDF2":2./3., "BE":1.}

    #now run transient solution and see if it gives back the same answer
    psi_plus_old   = np.array(psi_plusSS)*0.5
    psi_minus_old  = np.array(psi_minusSS)*0.58
    while t <= 1.:

        print "t is ", t
        t += dt
        react_term = 1./(GC.SPD_OF_LGT*dt)

        # Create the sources for time stepper
        ts = "BE" #timestepper
        source_handles = [OldIntensitySrc(mesh, dt, t, ts), 
                          StreamingSrc(mesh, dt, t, ts),
                          ReactionSrc(mesh, dt, t, ts)]

        #Check all derived classes are implemented correctly
        assert all([isinstance(i, SourceHandler) for i in source_handles])

        #base type
        Q_tot = np.array(Q)
        for src in source_handles:

            # build src for this term
            Q_new = src.buildSource(psi_plus_old = psi_plus_old, psi_minus_old = 
                    psi_minus_old, bc_flux_left = psi_left, bc_flux_right = psi_right,
                    cx_old = cross_sects)

            # Add elementwise the src to the total
            Q_tot += Q_new


        #solve the system
        psi_minus, psi_plus, E, F = radiationSolveSS(mesh, cross_sects, Q_tot,
                bc_psi_left = psi_left, bc_psi_right = psi_right, diag_add_term =
                react_term)


        psi_plus_old = psi_plus
        psi_minus_old = psi_minus

        print sum([(psi_plus[i][0]- psi_plusSS[i][0])/psi_plus[i][0] for i in range(len(psi_plus_old))])

    # plot solutiona
    plotScalarFlux(mesh, psi_minus, psi_plus)

    print psi_plus
    print psi_plusSS
    print psi_minus
    print psi_minusSS



if __name__ == "__main__":
    solveRadProblem()
