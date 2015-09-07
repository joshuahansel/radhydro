## @package src.radiationSolveSS
#  Provides functions to solve the steady-state S-2 equations. Also provides
#  means of employing solver to be used in temporal discretizations.
#
#  TODO: Currently this code builds the matrix as a full matrix and then
#  converts it over to a sparse matrix for solving (the sparse solver is much faster).
#  It would probably be significantly more efficient to build the matrix as a CSR matrix directly.  We didn't
#  have time to figure out the python syntax for storing rows.  A banded
#  matrix cannot be used directly due to periodic boundary conditions, but this
#  could be used if you do it properly.  See http://www4.ncsu.edu/~stsynkov/book_sample_material/Sections_5.4-5.5.pdf

import math
import numpy as np
from numpy import array

from mesh import Mesh
from utilityFunctions import getIndex
from radUtilities import mu, computeScalarFlux
import globalConstants as GC
from radiation import Radiation
from scipy.sparse import csr_matrix, linalg

## Steady-state solve function for the S-2 equations.
#
#  @param[in] mesh     a mesh object
#  @param[in] cross_x  list of cross sections for each element, stored as tuple
#                      for each cell.
#  @param[in] Q        \f$\tilde{\mathcal{Q}}\f$ as defined in documentation
#                      for time-dependent solvers. This is passed in as a vector
#                      with the same global numbering as $\Psi$ unknowns.
#  @param[in] diag_add_term       \f$\alpha\f$ as defined in documentation for
#                                 time-dependent solvers
#  @param[in] implicit_scale      \f$\beta\f$ as defined in documentation for
#                                 time-dependent solvers
#  @param[in] bound_curr_lt       left  boundary current data, \f$j^+\f$
#  @param[in] bound_curr_rt       right boundary current data, \f$j^-\f$
#  @param[in] bound_flux_left     left  boundary psi
#  @param[in] bound_flux_right    right boundary psi
#
#  @return 
#          -# \f$\Psi^+\f$, angular flux in plus directions multiplied by \f$2\pi\f$
#          -# \f$\Psi^-\f$, angular flux in minus directions multiplied by \f$2\pi\f$
#          -# \f$\mathcal{E}\f$: radiation energy
#          -# \f$\mathcal{F}\f$: radiation flux
#
def radiationSolveSS(mesh, cross_x, Q, rad_BC, diag_add_term=0.0, implicit_scale=1.0):

    # abbreviation for the scale term
    beta = implicit_scale

    # Handle boundary conditions  
    if rad_BC.bc_type == 'periodic':
        bc_psi_left = bc_psi_right = None
    else:
        bc_psi_left, bc_psi_right = rad_BC.getIncidentFluxes()

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
       QLminus = Q[iLminus] # minus direction, Left
       QLplus  = Q[iLplus]  # plus  direction, Left
       QRminus = Q[iRminus] # minus direction, Right
       QRplus  = Q[iRplus]  # plus  direction, Right

       # Left control volume, minus direction
       row = np.zeros(n)
       row[iLminus]    = -beta*mu["-"]/h + (beta*cx_tL + diag_add_term) - 0.5*beta*cx_sL
       row[iLplus]     = -0.5*beta*cx_sL
       row[iRminus]    = beta*mu["-"]/h
       matrix[iLminus] = row
       rhs[iLminus]    = QLminus
      
       # Left control volume, plus direction
       row = np.zeros(n)

       #Handle BC's
       if i == 0:
          if rad_BC.bc_type == 'periodic':
             rhs[iLplus] = 0.0    #If periodic no term to add 
             iPerPlus = getIndex(mesh.n_elems-1,"R","+") #periodic outflow index
             row[iPerPlus] = -2.0*beta*mu["+"]/h #negative because on LHS of eq
          else:
             rhs[iLplus] = 2.0*beta*mu["+"]/h*bc_psi_left
       else:
          row[iprevRplus] = -2.0*beta*mu["+"]/h

       row[iLminus]    = -0.5*beta*cx_sL
       row[iLplus]     = beta*mu["+"]/h + (beta*cx_tL + diag_add_term) - 0.5*beta*cx_sL
       row[iRplus]     = beta*mu["+"]/h
       matrix[iLplus]  = row
       rhs[iLplus]    += QLplus

       # Right control volume, minus direction
       row = np.zeros(n)
       row[iLminus]     = -beta*mu["-"]/h
       row[iRminus]     = -beta*mu["-"]/h + (beta*cx_tR + diag_add_term) - 0.5*beta*cx_sR
       row[iRplus]      = -0.5*beta*cx_sR

       #Handle BC
       if i == mesh.n_elems-1:
          if rad_BC.bc_type == 'periodic':
             rhs[iRminus] = 0.0
             iPerMinus = getIndex(0,"L","-")
             row[iPerMinus] = 2.0*beta*mu["-"]/h #no negative because on LHS of eq
          else:
             rhs[iRminus] = -2.0*beta*mu["-"]/h*bc_psi_right
       else:
          row[inextLminus] = 2.0*beta*mu["-"]/h

       matrix[iRminus]  = row
       rhs[iRminus]    += QRminus

       # Right control volume, plus direction
       row = np.zeros(n)
       row[iLplus]      = -beta*mu["+"]/h
       row[iRminus]     = -0.5*beta*cx_sR
       row[iRplus]      = beta*mu["+"]/h + (beta*cx_tR + diag_add_term) - 0.5*beta*cx_sR
       matrix[iRplus]   = row
       rhs[iRplus]      = QRplus

    # solve linear system as a sparse matrix
    sparse_matrix = csr_matrix(matrix)
    solution = linalg.spsolve(sparse_matrix, rhs)

 #   solution = np.linalg.solve(matrix,rhs)

    #return solution
    return Radiation(solution)

