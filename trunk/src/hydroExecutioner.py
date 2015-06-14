## @package src.hydroExecutioner
#  Runs a hydrodynamics problem.

import numpy as np
from numpy import       array
from mesh import        Mesh
from math import        sqrt
from hydroState import HydroState
from hydroSlopes import HydroSlopes
from musclHancock import hydroPredictor, hydroCorrector
from plotUtilities import plotHydroSolutions

## Main executioner for Hydro solve. Currently in a testing state.
def solveHydroProblem():

    # option to print solution
    print_solution = False

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

    #made up spec heat
    spec_heat = 1.0

    #Create a mesh, currently hardcoded
    mesh = Mesh(n, width)
    dx = mesh.getElement(0).dx

    if n % 2 != 0: #simple error raise, % has lots of diff meanings in python
        raise ValueError("Must be even number of cells")

    i_left = int(0.3*n)
    i_right = int(0.7*n)

    #Create cell centered variables
    states_a = [HydroState(u=u_left,p=p_left,gamma=gamma,rho=rho_left,spec_heat=spec_heat)
       for i in range(i_left)]
    states_a = states_a + [HydroState(u=u_right,p=p_right,gamma=gamma,spec_heat=spec_heat,rho=rho_right) for i in
            range(i_right)]

    #-----------------------------------------------------------------
    # Solve Problem
    #----------------------------------------------------------------

    #loop over time steps
    time_index = 0
    while (t < t_end):

        # increment time index
        time_index += 1

        #Compute a new time step size based on CFL
        c = [sqrt(i.p*i.gamma/i.rho)+abs(i.u) for i in states_a]
        dt_vals = [cfl*(mesh.elements[i].dx)/c[i] for i in range(len(c))]
        dt = min(dt_vals)

        t += dt
        #shorten last step to be exact
        if (t > t_end):
            t -= dt
            dt = t_end - t + 0.000000001
            t += dt

        print("Time step %d: t = %f -> %f" % (time_index,t-dt,t))

        # compute slopes
        slopes = HydroSlopes(states_a)

        #Solve predictor step
        states_a = hydroPredictor(mesh, states_a, slopes, dt)

        #Solve corrector step
        states_a = hydroCorrector(mesh, states_a, dt)


    # plot solution
    plotHydroSolutions(mesh,states=states_a)

    # print solution
    if print_solution:
       for state in states_a:
          print state
    

if __name__ == "__main__":
    solveHydroProblem()

