## @package src.radiationSolveSS
#  Provides functions to solve the steady-state S-2 equations. Also provides
#  means of employing solver to be used in temporal discretizations.

import math
import numpy as np
from numpy import array
from mesh import Mesh
from utilityFunctions import getIndex

## Steady-state solve function for the S-2 equations.
#
#  @param[in] mesh     a mesh object
#  @param[in] cross_x  list of cross sections for each element, stored as tuple
#                      for each cell.
#  @param[in] Q_minus  total isotropic source in minus direction:
#                      \f$Q = Q_0 + 3\mu^-Q_1\f$
#  @param[in] Q_plus   total isotropic source in plus direction:
#                      \f$Q = Q_0 + 3\mu^+Q_1\f$
#  @param[in] diag_add_term       term to add to reaction term for use in
#                                 time-dependent solvers, e.g., alpha*sigma_t*psi_L
#                                 -> (alpha*sigma_t + 1/c*delta_t)psi_L. Must be
#                                 done after scale
#  @param[in] bound_curr_lt       left  boundary current data, \f$j^+\f$
#  @param[in] bound_curr_rt       right boundary current data, \f$j^-\f$
#
#  @return 
#          -# \f$\psi^+\f$, angular flux in plus directions
#          -# \f$\psi^-\f$, angular flux in minus directions
#          -# \f$\mathcal{E}\f$: radiation energy
#          -# \f$\mathcal{F}\f$: radiation flux
#
def radiationSolveSS(mesh, cross_x, Q_minus, Q_plus, diag_add_term=0.0,
    bound_curr_lt=0.0, bound_curr_rt=0.0):

    # set directions
    mu = {"-" : -1/math.sqrt(3), "+" : 1/math.sqrt(3)}

    # 1/(4*pi) constant for isotropic source Q
    c_Q = 1/(4*math.pi)

    # compute boundary fluxes based on incoming currents
    boundary_flux_plus  =  bound_curr_lt / (2*math.pi*mu["+"])
    boundary_flux_minus = -bound_curr_rt / (2*math.pi*mu["-"])

    # initialize numpy arrays for system matrix and rhs
    n = 4*mesh.n_elems
    matrix = np.zeros((n, n))
    rhs    = np.zeros(n)

    # loop over interior cells
    for i in xrange(mesh.n_elems):
       # compute indices
       iprevRplus  = getIndex(i-1,"R","+") # dof i-1,R,+
       iLminus     = getIndex(i,  "L","-") # dof i,  L,-
       iLplus      = getIndex(i,  "L","+") # dof i,  L,+
       iRminus     = getIndex(i,  "R","-") # dof i,  R,-
       iRplus      = getIndex(i,  "R","+") # dof i,  R,+
       inextLminus = getIndex(i+1,"L","-") # dof i+1,L,-

       # get cell size
       h = mesh.getElement(i).dx

       # get cross sections
       cx_sL = cross_x[i][0].sig_s # Left  scattering
       cx_sR = cross_x[i][1].sig_s # Right scattering
       cx_tL = cross_x[i][0].sig_t # Left  total
       cx_tR = cross_x[i][1].sig_t # Right total

       # get sources
       QLminus = Q_minus[i][0] # minus direction, Left
       QLplus  = Q_plus [i][0] # plus  direction, Left
       QRminus = Q_minus[i][1] # minus direction, Right
       QRplus  = Q_plus [i][1] # plus  direction, Right

       # Left control volume, minus direction
       row = np.zeros(n)
       row[iLminus]    = -0.5*mu["-"] + 0.5*(cx_tL+diag_add_term)*h - 0.25*cx_sL*h
       row[iLplus]     = -0.25*cx_sL*h
       row[iRminus]    = 0.5*mu["-"]
       matrix[iLminus] = row
       rhs[iLminus]    = 0.5*c_Q*h*QLplus
      
       # Left control volume, plus direction
       row = np.zeros(n)
       if i == 0:
          rhs[iLplus] = mu["+"]*boundary_flux_plus
       else:
          row[iprevRplus] = -mu["+"]
       row[iLminus]    = -0.25*cx_sL*h
       row[iLplus]     = 0.5*mu["+"] + 0.5*(cx_tL+diag_add_term)*h - 0.25*cx_sL*h
       row[iRplus]     = 0.5*mu["+"]
       matrix[iLplus]  = row
       rhs[iLplus]    += 0.5*c_Q*h*QLminus

       # Right control volume, minus direction
       row = np.zeros(n)
       row[iLminus]     = -0.5*mu["-"]
       row[iRminus]     = -0.5*mu["-"] + 0.5*(cx_tR+diag_add_term)*h - 0.25*cx_sR*h
       row[iRplus]      = -0.25*cx_sR*h
       if i == mesh.n_elems-1:
          rhs[iRminus] = -mu["-"]*boundary_flux_minus
       else:
          row[inextLminus] = mu["-"]
       matrix[iRminus]  = row
       rhs[iRminus]    += 0.5*c_Q*h*QRminus

       # Right control volume, plus direction
       row = np.zeros(n)
       row[iLplus]      = -0.5*mu["+"]
       row[iRminus]     = -0.25*cx_sR*h
       row[iRplus]      = 0.5*mu["+"] + 0.5*(cx_tR+diag_add_term)*h - 0.25*cx_sR*h
       matrix[iRplus]   = row
       rhs[iRplus]      = 0.5*c_Q*h*QRplus

    # solve linear system
    solution = np.linalg.solve(matrix, rhs)

    # extract solution from global vector
    psi_minus = [(solution[4*i],  solution[4*i+2]) for i in xrange(mesh.n_elems)]
    psi_plus  = [(solution[4*i+1],solution[4*i+3]) for i in xrange(mesh.n_elems)]

    # return zeros for time being
    E = [(0, 0) for i in range(mesh.n_elems)]
    F = [(0, 0) for i in range(mesh.n_elems)]

    return psi_minus, psi_plus, E, F

