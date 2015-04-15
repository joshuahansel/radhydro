import numpy as np
from numpy import array
from Mesh import Mesh    #Import the class

#------------------------------------------------------------------------------------
"""Input is the mesh, list of cross sections for each element, Q0 isotropic source,
Q1 source for plus directions, and Q1_minus direction source.  Keyword arguments
passed in are the boundary currents, which are assumed vacuum by default, and the
scale_factor and diag_add_term. 

CX's and all sources are passed in as a list, where the index corresponds to the
element id. They are all passed in as a tuple with the left and right value, e.g.,
cx[0] = (cx_l,0, cx_r,0).

stream_scale_factor: the streaming scale factor (call it alpha) multiplies the
reaction and the entire streaming term, i.e., sigma_t psi + mu \dpsi/dx ->
alpha*(sigma_t psi + mu \dpsi/dx).  This is for used with various time dependent
solvers.

diag_add_term: added to the reaction term in each equation. For use in time-dependent
solvers, e.g., alpha*sigma_t*psi_L -> (alpha*sigma_t + 1/c*delta_t)psi_L. Must be done after scale
factor is applied.

"""
def radiationSolver(mesh, cross_x, Q0, Q1_plus, Q1_minus, stream_scale_factor=1.0,
        diag_add_term=0.0, bound_curr_lt=0.0, bound_curr_rt=0.0):






    psi_l = 0.0
    psi_r = 0.0


    psi_minus = [(psi_l, psi_r) for i in range(mesh.n_elems)]
    psi_plus = [(psi_l, psi_r) for i in range(mesh.n_elems)]
   # E = 
   # F = 

    return psi_minus, psi_plus, E, F


