## @package src.hydroExecutioner
#  Runs a hydrodynamics problem.
import sys
sys.path.append('src')

from mesh import Mesh
from crossXInterface import CrossXInterface
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotAngularFlux, plotScalarFlux, computeScalarFlux
from sourceHandlers import * 
import matplotlib as plt
from pylab import *
import numpy as np
from copy import deepcopy

## Main executioner for Radiation solve. Currently in a testing state.
def solveRadProblem():

    #-------------------------------------------------------------------------------
    # create uniform mesh
    mesh = Mesh(100, 5.)
    sig_s = 10.0
    sig_a = 2.0

    # compute cross sections
    cross_sects = [(CrossXInterface(sig_a, sig_s), CrossXInterface(sig_a, sig_s)) for i in xrange(mesh.n_elems)]
    dt = 0.1
    t  = 0.0
    t_end = 10.

    #consistent BC's, eventually lets just switch to psi's and forget the currents
    Q_iso = 5.
    from math import pi
    psi_left = Q_iso/(2.*sig_a)*0.5
    psi_right = Q_iso/(2.*sig_a)*0.9

    # call radiation solver to get the steady state solution
    Q = []
    for i in xrange(mesh.n_elems):
    
        Q_new = [0.0 for i in range(4)]
        Q_new[getLocalIndex("L","+")] = Q_iso/2.
        Q_new[getLocalIndex("R","+")] = Q_iso/2.
        Q_new[getLocalIndex("L","-")] = Q_iso/2.
        Q_new[getLocalIndex("R","-")] = Q_iso/2.
        Q += Q_new


    Q = np.array(Q)
    psi_minusSS, psi_plusSS, E, F = radiationSolveSS(mesh, cross_sects, Q,
            bc_psi_right = psi_right, bc_psi_left = psi_left)

    #print "Psi_minusSS", psi_minusSS
    #print "Psi_plusSS", psi_plusSS
    diag_terms = {"CN":0.5, "BDF2":2./3., "BE":1.}
    plotScalarFlux(mesh, psi_minusSS, psi_plusSS)

    #now run transient solution from arbitrary starting and see if it gives back the same answera    = n
    psi_p_i= psi_plusSS[mesh.n_elems/2]*0.5
    psi_m_i= psi_minusSS[mesh.n_elems/2]*0.25
    psi_plus_old   = [(psi_p_i,psi_p_i) for i in
        range(mesh.n_elems) ]
    psi_minus_old  =  [(psi_m_i,psi_m_i) for i in
        range(mesh.n_elems) ] 
    E_old = [(GC.SPD_OF_LGT*(psi_plus_old[i][0] + psi_minus_old[i][0]),
              GC.SPD_OF_LGT*(psi_plus_old[i][1] + psi_minus_old[i][1]))  for i in
            range(len(psi_plus_old))]
    while t <= t_end:

        t += dt
        react_term = 1./(GC.SPD_OF_LGT*dt)

        # Create the sources for time stepper
        ts = "BE" #timestepper
        source_handles = [OldIntensitySrc(mesh, dt, ts), 
                          StreamingSrc(mesh, dt, ts),
                          ReactionSrc(mesh, dt, ts),
                          ScatteringSrc(mesh, dt, ts)]

        #Check all derived classes are implemented correctly
        assert all([isinstance(i, SourceHandler) for i in source_handles])

        #base type
        Q_tot = np.array(Q)
        for src in source_handles:

            # build src for this term
            Q_new = src.buildSource(psi_plus_old = psi_plus_old, psi_minus_old = 
                    psi_minus_old, bc_flux_left = psi_left, bc_flux_right = psi_right,
                    cx_old = cross_sects,)
         #   print "src", src
         #   print "Q_new: "
         #   print Q_new

            # Add elementwise the src to the total
            Q_tot += Q_new


        #solve the system
        psi_minus, psi_plus, E, F = radiationSolveSS(mesh, cross_sects, Q_tot,
                bc_psi_left = psi_left, bc_psi_right = psi_right, diag_add_term = react_term)


        psi_plus_old = deepcopy(psi_plus)
        psi_minus_old = deepcopy(psi_minus)

        print sum([(psi_plus[i][0]- psi_plusSS[i][0])/psi_plus[i][0] for i in range(len(psi_plus_old))])

    # plot solutiona
 #   plotScalarFlux(mesh, psi_minus, psi_plus, save=True)

    #compare steady state and transient phi's
    phiSS = computeScalarFlux(psi_plusSS, psi_minusSS)
    phiTR = computeScalarFlux(psi_plus, psi_minus)
    phiSSi = [0.5*(i[0] + i[1]) for i in phiSS]
    phiTRi = [0.5*(i[0] + i[1]) for i in phiTR]
 #   print "sol", (sol*1./(GC.SPD_OF_LGT*dt))

  #  print "Scalar flux SS: "  ,  phiSS
  #  print "Scalar flux TR: " , phiTR
  #  print "Psi SS + : ", psi_plusSS
  #  print "Psi TR + : ", psi_plus
  #  print "Psi SS - : ", psi_minusSS
  #  print "Psi TR - : ", psi_minus
  #  print "Diff in averages: ", [(phiSSi[i] - phiTRi[i])/phiSSi[i] for i in range(len(phiSS))]
  #  print "Diff in edge: ", [phiSS[i][0] - phiTR[i][0] for i in range(len(phiSS))]
  #  print "Diff in edge: ", [phiSS[i][1] - phiTR[i][1] for i in range(len(phiSS))]
    

if __name__ == "__main__":
    solveRadProblem()
