import numpy as np
from numpy import       array
from Mesh import        Mesh
from math import        sqrt
from MusclHanc import   hydroPredictor, main, HydroState, \
                        plotHydroSolutions, hydroCorrector

#------------------------------------------------------------------------------------
#Main executioner class.  
#
#
#Q1 source for plus directions, and Q1_minus direction source.  Keyword arguments
#passed in are the boundary currents, which are assumed vacuum by default, and the
#scale_factor and diag_add_term. 
#
#CX's and all sources are passed in as a list, where the index corresponds to the
#element id. They are all passed in as a tuple with the left and right value, e.g.,
#cx[0] = (cx_l,0, cx_r,0).
#
#stream_scale_factor: the streaming scale factor (call it alpha) multiplies the
#reaction and the entire streaming term, i.e., sigma_t psi + mu \dpsi/dx ->
#alpha*(sigma_t psi + mu \dpsi/dx).  This is for used with various time dependent
#solvers.
#
#diag_add_term: added to the reaction term in each equation. For use in time-dependent
#solvers, e.g., alpha*sigma_t*psi_L -> (alpha*sigma_t + 1/c*delta_t)psi_L. Must be done after scale
#factor is applied.
#
#"""
#  
#

## Main executioner for Hydro solve. Currently in a testing state.
#
#  @param[in] mesh     a mesh object
#  @param[in] cross_x  list of cross sections for each element
#  @param[in] Q0       isotropic source
#  @param[in] Q1_plus  source for plus directions
#  @param[in] Q1_minus source for minus directions
#  @param[in] stream_scale_factor scale factor for reaction and streaming terms
#  @param[in] diag_add_term       term to add to reaction term
#  @param[in] bound_curr_lt       left boundary current
#  @param[in] bound_curr_rt       right boundary current
#
#  @return 
#          -# \f$\psi^+\f$, angular flux in plus directions
#          -# \f$\psi^-\f$, angular flux in minus directions
#          -# \f$\mathcal{E}\f$: radiation energy
#          -# \f$\mathcal{F}\f$: radiation flux
#
def solveHydroProblem():

    #-------------------------------------------------------------------------------
    # Construct initial hydro states, probably create an initializer class
    #-------------------------------------------------------------------------------
    width = 1.0
    t_end = 0.05

    t = 0.0
    cfl = 0.5
    n = 500

    #Left and right BC and initial values
    gamma = 1.4 #gas constant
    p_left = 1.0
    u_left  = .750
    rho_left = 1.

    p_right = 0.1
    u_right = 0.0
    rho_right = 0.125

    #Create a mesh, currently hardcoded
    mesh = Mesh(n, width)
    dx = mesh.getElement(0).dx

    if n % 2 != 0: #simple error raise, % has lots of diff meanings in python
        raise ValueError("Must be even number of cells")

    i_left = int(0.3*n)
    i_right = int(0.7*n)

    #Create cell centered variables
    states_a = [HydroState(u=u_left,p=p_left,gamma=gamma,rho=rho_left) for i in range(i_left)]
    states_a = states_a + [HydroState(u=u_right,p=p_right,gamma=gamma,rho=rho_right) for i in
            range(i_right)]

    #-----------------------------------------------------------------
    # Solve Problem
    #----------------------------------------------------------------

    #loop over time steps
    while (t < t_end):

        #Compute a new time step size based on CFL
        c = [sqrt(i.p*i.gamma/i.rho)+abs(i.u) for i in states_a]
        dt_vals = [cfl*(mesh.elements[i].dx)/c[i] for i in range(len(c))]
        dt = min(dt_vals)
        print "new dt:", dt

        t += dt
        #shorten last step to be exact
        if (t > t_end):
            t -= dt
            dt = t_end - t + 0.000000001
            t += dt

        #Solve predictor step
        states_l, states_r = hydroPredictor(mesh, states_a, dt)

        #Solve corrector step
        states_a = hydroCorrector(mesh, states_a, states_l, states_r, dt)


    spat_coors = [i.x_cent for i in mesh.elements]
    plotHydroSolutions(spat_coors,states=states_a)
    for i in states_a:
        print i
    


if __name__ == "__main__":

    solveHydroProblem()
