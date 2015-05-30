## @package src.radExecutioner
#  Runs a radiation problem.
import sys
sys.path.append('src')

import matplotlib as plt
from pylab import *
import numpy as np
from copy import deepcopy

from mesh import Mesh
from crossXInterface import CrossXInterface
from radiationSolveSS import radiationSolveSS
from plotUtilities import plotAngularFlux, plotScalarFlux
from transientSource import * 
from radUtilities import computeEnergyDensity, computeScalarFlux

## Main executioner for Radiation solve. Currently in a testing state.
def solveRadProblem():

    # create uniform mesh
    mesh = Mesh(50, 5.)

    # compute uniform cross sections
    sig_s = 1.0
    sig_a = 2.0
    cross_sects = [(CrossXInterface(sig_a, sig_s), CrossXInterface(sig_a, sig_s))
                  for i in xrange(mesh.n_elems)]

    # transient options
    dt = 0.1            # time step size
    t  = 0.0            # begin time
    t_end = 10.0        # end time
    time_stepper = "CN" # time-stepper

    # boundary fluxes
    psi_left  = 2.5
    psi_right = 2.2
   
    # create the steady-state source
    Q = list()
    for i in xrange(mesh.n_elems):
        Q_new = [0.0 for i in range(4)]
        Q_new[getLocalIndex("L","+")] = 2.4
        Q_new[getLocalIndex("R","+")] = 2.4
        Q_new[getLocalIndex("L","-")] = 2.4
        Q_new[getLocalIndex("R","-")] = 2.4
        Q += Q_new
    Q = np.array(Q)

    # compute the steady-state solution
    psim_ss, psip_ss, E, F = radiationSolveSS(mesh, cross_sects, Q,
       bc_psi_right = psi_right, bc_psi_left = psi_left)

    # run transient solution from arbitrary IC, such as zero
    psip_old   = [(0.0,0.0) for i in range(mesh.n_elems)]
    psim_old   = deepcopy(psip_old)
    psip_older = deepcopy(psip_old)
    psim_older = deepcopy(psip_old)
    E_old      = deepcopy(psip_old)
    E_older    = deepcopy(psip_old)

    # create transient source
    transientSource = TransientSource(mesh, time_stepper)

    # transient loop
    transient_incomplete = True # boolean flag signalling end of transient
    while transient_incomplete:

        # adjust time step size if necessary
        if t + dt >= t_end:
           dt = t_end - t
           t = t_end
           transient_incomplete = False # signal end of transient
        else:
           t += dt
        print("t = %0.3f -> %0.3f" % (t-dt,t))

        # build source for this handler
        Q_tr = transientSource.evaluate(
           dt            = dt,
           bc_flux_left  = psi_left,
           bc_flux_right = psi_right,
           cx_older      = cross_sects,
           cx_old        = cross_sects,
           psim_older    = psim_older,
           psip_older    = psip_older,
           psim_old      = psim_old,
           psip_old      = psip_old,
           E_older       = E_older,
           E_old         = E_old,
           Q_older       = Q,
           Q_old         = Q,
           Q_new         = Q)

        # solve the transient system
        alpha = 1./(GC.SPD_OF_LGT*dt)
        beta = {"CN":0.5, "BDF2":2./3., "BE":1.}
        psim, psip, E, F = radiationSolveSS(mesh, cross_sects, Q_tr,
           bc_psi_left = psi_left, bc_psi_right = psi_right,
           diag_add_term = alpha, implicit_scale = beta[time_stepper] )

        # save oldest solutions
        psip_older = deepcopy(psip_old)
        psim_older = deepcopy(psim_old)
        E_older    = deepcopy(E_old)

        # save old solutions
        psip_old = deepcopy(psip)
        psim_old = deepcopy(psim)
        E_old    = deepcopy(E)

    # compute steady state scalar flux
    phi_ss = computeScalarFlux(psip_ss, psim_ss)

    # plot solutions
    plotScalarFlux(mesh, psim, psip, scalar_flux_exact=phi_ss,
       exact_data_continuous=False)
    

if __name__ == "__main__":
    solveRadProblem()

