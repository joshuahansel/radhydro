## @package src.musclHancock
#  Contains functions for implementing MUSCL-Hancock.

import numpy as np
from pylab import *
from math import sqrt, isinf
from copy import deepcopy
from utilityFunctions import *
from hydroState import HydroState
from hydroSlopes import HydroSlopes

## Predictor solver for hydro.
#
#  @param[in] mesh        mesh object
#  @param[in] states_old  old cell-average states, \f$\mathbf{H}^n_i\f$
#  @param[in] slopes      slopes, \f$\Delta_i\f$
#  @param[in] dt          full time step size, \f$\Delta t\f$
# 
#  @return
#     -# predicted cell-average states, \f$\mathbf{H}^{n+\frac{1}{2}_i}\f$
#
def hydroPredictor(mesh, states_old, slopes, dt):

    # mesh size. NOTE: uniform mesh size is assumed here
    dx = mesh.getElement(0).dx

    # number of elements
    n = mesh.n_elems

    ##Create vectors of conserved quantities
    #rho = [s.rho for s in states]
    #mom = [s.rho*s.u for s in states]
    #erg = [s.rho*(0.5*s.u*s.u + s.e) for s in states]
    #rho_l, rho_r, mom_l, mom_r, erg_l, erg_r =\
    #   slopes.computeEdgeConservativeVariablesValues(states)

    ##Compute left and right states
    #states_l = [deepcopy(i) for i in states] #initialize 
    #states_r = [deepcopy(i) for i in states]
    #for i in range(len(rho_l)):
    #    states_l[i].updateState(rho_l[i], mom_l[i], erg_l[i])
    #    states_r[i].updateState(rho_r[i], mom_r[i], erg_r[i])

    ##Initialize predicited conserved quantities
    #rho_l_p = [0.0 for i in range(len(rho_l))]
    #rho_r_p = [0.0 for i in range(len(rho_l))]
    #mom_l_p = [0.0 for i in range(len(rho_l))]
    #mom_r_p = [0.0 for i in range(len(rho_l))]
    #erg_l_p = [0.0 for i in range(len(rho_l))]
    #erg_r_p = [0.0 for i in range(len(rho_l))]

    ##Advance in time each edge variable
    #for i in range(len(rho_l)):

    #    #rho
    #    rho_l_p[i] = advCons(rho_l[i],dx,0.5*dt,rhoFlux(states_l[i]),rhoFlux(states_r[i])) 
    #    rho_r_p[i] = advCons(rho_r[i],dx,0.5*dt,rhoFlux(states_l[i]),rhoFlux(states_r[i])) 

    #    #mom
    #    mom_l_p[i] = advCons(mom_l[i],dx,0.5*dt,momFlux(states_l[i]),momFlux(states_r[i])) 
    #    mom_r_p[i] = advCons(mom_r[i],dx,0.5*dt,momFlux(states_l[i]),momFlux(states_r[i])) 

    #    #erg
    #    erg_l_p[i] = advCons(erg_l[i],dx,0.5*dt,ergFlux(states_l[i]),ergFlux(states_r[i])) 
    #    erg_r_p[i] = advCons(erg_r[i],dx,0.5*dt,ergFlux(states_l[i]),ergFlux(states_r[i])) 

    ##Advance the primitive variables at the edges
    #for i in range(len(rho_l)):
    #    states_l[i].updateState(rho_l_p[i], mom_l_p[i], erg_l_p[i])
    #    states_r[i].updateState(rho_r_p[i], mom_r_p[i], erg_r_p[i])

    ##Return the new averages
    #for i in range(len(states)):
    #    states[i].updateState(0.5*(rho_l_p[i]+rho_r_p[i]),
    #                       0.5*(mom_l_p[i]+mom_r_p[i]),
    #                       0.5*(erg_l_p[i]+erg_r_p[i]))

    #Create vectors of conserved quantities
    rho = np.zeros(n)
    mom = np.zeros(n)
    erg = np.zeros(n)
    for i in xrange(n):
       rho[i], mom[i], erg[i] = states_old[i].getConservativeVariables()

    # compute edge values
    rho_L, rho_R, mom_L, mom_R, erg_L, erg_R =\
       slopes.computeEdgeConservativeVariablesValues(states_old)

    # compute left and right edge states
    states_L = [deepcopy(i) for i in states_old]
    states_R = [deepcopy(i) for i in states_old]
    for i in xrange(n):
        states_L[i].updateState(rho_L[i], mom_L[i], erg_L[i])
        states_R[i].updateState(rho_R[i], mom_R[i], erg_R[i])

    # initialize predicited conserved quantities
    rho_p = np.zeros(n)
    mom_p = np.zeros(n)
    erg_p = np.zeros(n)

    # advance each conservation variable by half a time step
    for i in xrange(n):

        #rho
        rho_p[i] = advCons(rho[i],dx,0.5*dt,rhoFlux(states_L[i]),rhoFlux(states_R[i])) 

        #mom
        mom_p[i] = advCons(mom[i],dx,0.5*dt,momFlux(states_L[i]),momFlux(states_R[i])) 

        #erg
        erg_p[i] = advCons(erg[i],dx,0.5*dt,ergFlux(states_L[i]),ergFlux(states_R[i])) 
        
    # compute the predicted cell-average states
    states_half = deepcopy(states_old)
    for i in xrange(n):
        states_half[i].updateState(rho_p[i], mom_p[i], erg_p[i])

    return states_half

    
## Corrector solver for hydro.
#
# @param[in] mesh         mesh object
# @param[in] states_half  predicted cell-average states,
#    \f$\mathbf{H}^{n+\frac{1}{2}}\f$
# @param[in] states_old   old cell-average states, \f$\mathbf{H}^n\f$
# @param[in] slopes_old   old slopes, \f$\Delta^n\f$
# @param[in] dt           full time step size, \f$\Delta t\f$
# @param[in] bc           hydro BC object
#
# @return
#    -#  new cell-average states, \f$\mathbf{H}^{n+1}_i\f$
#
def hydroCorrectorJosh(mesh, states_old, states_half, slopes_old, dt, bc):

    #Choose riemann solver
    riem_solver = HLLCSolver #HLLSolver, HLLCSolver

    #Solve for fluxes and values at faces
    n = mesh.n_elems

    #Create vectors of predicted variables
    rho_p = [s.rho                     for s in states_half]
    mom_p = [s.rho*s.u                 for s in states_half]
    erg_p = [s.rho*(0.5*s.u*s.u + s.e) for s in states_half]

    # get boundary values and states
    rho_BC_L, rho_BC_R, mom_BC_L, mom_BC_R, erg_BC_L, erg_BC_R =\
       bc.getBoundaryValues()
    state_BC_L, state_BC_R = bc.getBoundaryStates()

    rho_F = np.zeros(n+1)
    mom_F = np.zeros(n+1)
    erg_F = np.zeros(n+1)

    # solve Riemann problem at each interface
    for i in range(0,n+1):

        # get left and right states for Riemann problem at interface
        if i == 0: # left boundary edge
            rho_L = rho_BC_L
            rho_R = rho_p[i]
            mom_L = mom_BC_L
            mom_R = mom_p[i]
            erg_L = erg_BC_L
            erg_R = erg_p[i]
            state_L = state_BC_L
            state_R = states_half[i]
        elif i == n: # right boundary edge
            rho_L = rho_p[i-1]
            rho_R = rho_BC_R
            mom_L = mom_p[i-1]
            mom_R = mom_BC_R
            erg_L = erg_p[i-1]
            erg_R = erg_BC_R
            state_L = states_half[i-1]
            state_R = state_BC_R
        else: # interior edge
            rho_L = rho_p[i-1]
            rho_R = rho_p[i]
            mom_L = mom_p[i-1]
            mom_R = mom_p[i]
            erg_L = erg_p[i-1]
            erg_R = erg_p[i]
            state_L = states_half[i-1]
            state_R = states_half[i]

        # solve Riemann problem at interface
        rho_F[i] = riem_solver(rho_L, rho_R, state_L, state_R, rhoFlux)
        mom_F[i] = riem_solver(mom_L, mom_R, state_L, state_R, momFlux)
        erg_F[i] = riem_solver(erg_L, erg_R, state_L, state_R, ergFlux)

    #Intialize cell average quantity arrays at t_old
    rho = [s.rho for s in states_old]
    mom = [s.rho*s.u for s in states_old]
    erg = [s.rho*(0.5*s.u**2. + s.e) for s in states_old]
    
    #Advance conserved values at centers based on edge fluxes
    for i in range(len(rho)):

        dx = mesh.getElement(i).dx

        #Example of edge fluxes:
        #   i is 0 for 1st element, so edge 0 and edge 1 is i and i+1
        rho[i] = advCons(rho[i],dx,dt,rho_F[i],rho_F[i+1])
        mom[i] = advCons(mom[i],dx,dt,mom_F[i],mom_F[i+1])
        erg[i] = advCons(erg[i],dx,dt,erg_F[i],erg_F[i+1])

    # store the boundary fluxes
    bound_F_left = {}
    bound_F_right = {}
    bound_F_left['rho'] = rho_F[0]
    bound_F_left['mom'] = mom_F[0]
    bound_F_left['erg'] = erg_F[0]
    bound_F_right['rho'] = rho_F[-1]
    bound_F_right['mom'] = mom_F[-1]
    bound_F_right['erg'] = erg_F[-1]

    #Advance primitive variables
    states_new = [deepcopy(i) for i in states_old] 
    for i in range(len(states_new)):
        states_new[i].updateState(rho[i],mom[i],erg[i])

    return states_new, bound_F_left, bound_F_right


def hydroCorrectorSimon(mesh, states_old, states_half, slopes_old, dt, bc):

    # choose riemann solver
    riem_solver = HLLCSolver #HLLSolver, HLLCSolver

    #Create edge values
    rho_half_L, rho_half_R, mom_half_L, mom_half_R, erg_half_L, erg_half_R =\
       slopes_old.computeEdgeConservativeVariablesValues(states_half)

    # Create edge states
    states_L = deepcopy(states_half) 
    states_R = deepcopy(states_half)
    for i in range(len(rho_half_L)):
        states_L[i].updateState(rho_half_L[i], mom_half_L[i], erg_half_L[i])
        states_R[i].updateState(rho_half_R[i], mom_half_R[i], erg_half_R[i])

    # get boundary values and states
    rho_BC_L, rho_BC_R, mom_BC_L, mom_BC_R, erg_BC_L, erg_BC_R =\
       bc.getBoundaryValues()
    state_BC_L, state_BC_R = bc.getBoundaryStates()

    n = mesh.n_elems
    rho_F = np.zeros(n+1)
    mom_F = np.zeros(n+1)
    erg_F = np.zeros(n+1)

    #Check if it is reflective, if so we need to reset the states due to the way the
    #edge values work. This is kind of crappy coding
    if bc.bc_type == "reflective":

        rho_F[0] = rhoFlux(states_L[0])
        mom_F[0] = momFlux(states_L[0])
        erg_F[0] = ergFlux(states_L[0])

        rho_F[-1] = rhoFlux(states_R[-1])
        mom_F[-1] = momFlux(states_R[-1])
        erg_F[-1] = ergFlux(states_R[-1])

    else: # Dirichlet

        rho_F[0] = riem_solver(rho_BC_L, rho_half_L[0], state_BC_L, states_L[0], rhoFlux)
        mom_F[0] = riem_solver(mom_BC_L, mom_half_L[0], state_BC_L, states_L[0], momFlux)
        erg_F[0] = riem_solver(erg_BC_L, erg_half_L[0], state_BC_L, states_L[0], ergFlux)

        rho_F[-1] = riem_solver(rho_BC_R, rho_half_R[-1], state_BC_R, states_R[-1], rhoFlux)
        mom_F[-1] = riem_solver(mom_BC_R, mom_half_R[-1], state_BC_R, states_R[-1], momFlux)
        erg_F[-1] = riem_solver(erg_BC_R, erg_half_R[-1], state_BC_R, states_R[-1], ergFlux)

    #Do the interior cells
    for i in range(0,n-1):

        rho_F[i+1] = riem_solver(rho_half_R[i], rho_half_L[i+1], states_R[i],
                states_L[i+1], rhoFlux)
        mom_F[i+1] = riem_solver(mom_half_R[i], mom_half_L[i+1], states_R[i],
                states_L[i+1], momFlux)
        erg_F[i+1] = riem_solver(erg_half_R[i], erg_half_L[i+1], states_R[i],
                states_L[i+1], ergFlux)

    #Store the boundary condition fluxes
    bound_F_left = {}
    bound_F_right = {}
    bound_F_left['rho'] = rho_F[0]
    bound_F_left['mom'] = mom_F[0]
    bound_F_left['erg'] = erg_F[0]
    bound_F_right['rho'] = rho_F[-1]
    bound_F_right['mom'] = mom_F[-1]
    bound_F_right['erg'] = erg_F[-1]

    #Intialize cell average quantity arrays at t_old
    rho = [s.rho for s in states_old]
    mom = [s.rho*s.u for s in states_old]
    erg = [s.rho*(0.5*s.u**2. + s.e) for s in states_old]
    
    #Advance conserved values at centers based on edge fluxes
    for i in range(len(rho)):

        dx = mesh.getElement(i).dx

        #Example of edge fluxes:
        #   i is 0 for 1st element, so edge 0 and edge 1 is i and i+1
        rho[i] = advCons(rho[i],dx,dt,rho_F[i],rho_F[i+1])
        mom[i] = advCons(mom[i],dx,dt,mom_F[i],mom_F[i+1])
        erg[i] = advCons(erg[i],dx,dt,erg_F[i],erg_F[i+1])

    #Advance primitive variables
    states_new = [deepcopy(i) for i in states_old] 
    for i in range(len(states_new)):
        states_new[i].updateState(rho[i],mom[i],erg[i])

    return states_new, bound_F_left, bound_F_right


#------------------------------------------------------------------------------------
# Define some functions for evaluating fluxes for different state variables
#------------------------------------------------------------------------------------
def rhoFlux(s):
    return s.rho*s.u

def momFlux(s):
    return s.rho*s.u*s.u+s.p

def ergFlux(s):
    return (s.rho*(0.5*s.u*s.u+s.e) + s.p) * s.u


#------------------------------------------------------------------------------------
# Create function for advancing conserved quantities in time
#------------------------------------------------------------------------------------
def advCons(val, dx, dt, f_left, f_right):

    return val + dt/dx*(f_left - f_right)

# ----------------------------------------------------------------------------------
def minMod(a,b):

    if a > 0 and b > 0:
        return min(a,b)
    elif a<0 and b<0:
        return max(a,b)
    else:
        return 0.

#------------------------------------------------------------------------------------
def HLLSolver(U_l, U_r, L, R, flux): #quantity of interest U, state to the left, state to right

    #sound speed:
    a_l = L.getSoundSpeed()
    a_r = R.getSoundSpeed()

    #Compute bounding speeds
    S_l = min(L.u - a_l, R.u - a_r)
    S_r = max(L.u + a_l, R.u + a_r)

    #Compute fluxes at the boundaries
    F_l = flux(L)
    F_r = flux(R)

    #Compute the state at the face
    U_hll = (S_r*U_r - S_l*U_l + F_l - F_r) / (S_r-S_l)
    F_hll = (S_r*F_l - S_l*F_r + S_l*S_r*(U_r - U_l) ) / (S_r - S_l)

    #Return appropraite state
    if S_r < 0:
        return F_r
    elif S_l <= 0.0 and S_r >= 0.0:
        return F_hll
    elif S_l > 0.0 and S_r > 0.0:
        return F_l
    else:
        raise ValueError("HLL solver produced unrealistic fluxes\n")
    
#------------------------------------------------------------------------------------
def HLLCSolver(U_l, U_r, L, R, flux): #quantity of interest U, state to the left, state to right

    #sound speed:
    a_l = L.getSoundSpeed()
    a_r = R.getSoundSpeed()

    #Compute bounding speeds
    S_l = min(L.u - a_l, R.u - a_r)
    S_r = max(L.u + a_l, R.u + a_r)
    S_star = ( (R.p - L.p + L.rho*L.u*(S_l - L.u) - R.rho*R.u*(S_r - R.u)) /
               (L.rho*(S_l - L.u) - R.rho*(S_r - R.u)) )

    #Check for zero velocity differences:
    if L == R:
        S_star = L.u

    #Compute fluxes at the boundaries
    F_l = flux(L)
    F_r = flux(R)

    #Compute star values
    coeff_l = L.rho*(S_l - L.u)/(S_l - S_star)
    coeff_r = R.rho*(S_r - R.u)/(S_r - S_star)

    if flux == rhoFlux:

        U_lstar = coeff_l
        U_rstar = coeff_r

    elif flux == momFlux:

        U_lstar = coeff_l*S_star
        U_rstar = coeff_r*S_star

    elif flux == ergFlux:

        U_lstar = coeff_l*( (0.5*L.u*L.u + L.e) + (S_star-L.u)*(S_star +
            L.p/(L.rho*(S_l - L.u)) ) )
        U_rstar = coeff_r*( (0.5*R.u*R.u + R.e) + (S_star-R.u)*(S_star +
            R.p/(R.rho*(S_r - R.u)) ) )

    else:

        raise ValueError("Ended up in a wierd place in HLLC")
    

    #Compute the state at the face
    F_lstar = F_l + S_l*(U_lstar - U_l)
    F_rstar = F_r + S_r*(U_rstar - U_r)

    #Return appropraite state
    if S_r < 0:

        #print "Return F_r", F_r
        return F_r

    elif S_l <= 0.0 and S_star > 0.0:
        
        #print "Retrun F_lstar", F_lstar
        return F_lstar

    elif S_star <= 0.0 and S_r > 0.0:

        #print "Return F_rstar", F_rstar
        return F_rstar

    elif S_l > 0.0:

        #print "Return F_l", F_l
        return F_l

    else:

        #print S_l, S_star, S_r
        raise ValueError("HLLC solver produced unrealistic fluxes\n")


#------------------------------------------------------------------------------------
# Create function for reconstructing slopes in entire array
#------------------------------------------------------------------------------------
def slopeReconstruction(u):

    limiter = "vanleer"

    #evaluate left and right value in each cell
    omega = 0.
    u_l = [0.0 for i in u]
    u_r = [0.0 for i in u]

    for i in range(len(u)):
        
        if i==0:
            del_i = u[i+1]-u[i]
        elif i==len(u)-1:
            del_i = u[i]-u[i-1]
        else:
            del_rt = u[i+1]-u[i]
            del_lt = u[i] - u[i-1]
            del_i = 0.5*(1.+omega)*del_lt + 0.5*(1.-omega)*del_rt

            #Simple slope limiters
            if limiter == "minmod":

                del_i = minMod(del_rt,del_lt)

            elif limiter == "vanleer":

                beta = 1.

                #Catch if divide by zero
                if abs(u[i+1] - u[i]) < 0.000000001*(u[i]+0.00001):
                    if abs(u[i]-u[i-1]) < 0.000000001*(u[i]+0.00001):
                        r = 1.0
                    else:
                        r = -1.
                else:
                    r = del_lt/del_rt
                        
                if r < 0.0 or isinf(r):
                    del_i = 0.0
                else:

                    zeta =  min(2.*r/(1.+r),2.*beta/(1.-omega+(1.+omega)*r))
                    del_i = zeta*del_i

            elif limiter == "none":
                del_i = del_i

            elif limiter == "step":
                del_i = 0.0

            else:
                raise ValueError("Invalid slope limiter\n")
            
        u_l[i] = u[i] - 0.5*del_i
        u_r[i] = u[i] + 0.5*del_i

    return u_l, u_r



#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

