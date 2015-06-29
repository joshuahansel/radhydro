
import numpy as np
from pylab import *
from math import sqrt, isinf
from copy import deepcopy
from utilityFunctions import *
from hydroState import HydroState
from hydroSlopes import HydroSlopes

#-------------------------------------------------------------------------------------
#
## Predictor solver for hydro.
#
# Boundary Conditions: Essentially boundaries are treated as reflective. The fluxes
# on the boundary are just estimated based on the flux based on the edge state at
# that node. There is no need to pass in the boundary conditions then.
#
# @param[in] mesh           Basic spatial mesh object
# @param[in] states_old_a   Averages at old state. 
# @param[in] dt             time step size for this hydro solve. To predict values at 
#                           0.5 dt, pass in 1.0 dt
# 
# @return
#       -#  predicted states at averages
#       -#  predicted states slopes
#
def hydroPredictor(mesh, states_old_a, slopes, dt):

    dx = mesh.getElement(0).dx #currently a fixed width

    if mesh.n_elems % 2 != 0:
        raise ValueError("Must be even number of cells")

    #Initialize cell centered variables as passed in
    states = states_old_a

    #-----------------------------------------------------------------
    # Solve Problem
    #----------------------------------------------------------------

    #Create vectors of conserved quantities
    rho = [s.rho                     for s in states]
    mom = [s.rho*s.u                 for s in states]
    erg = [s.rho*(0.5*s.u*s.u + s.e) for s in states]

    # extract slopes
    rho_slopes, mom_slopes, erg_slopes = slopes.extractSlopes()

    # compute linear representations
    rho_l, rho_r = createLinearRepresentation(rho,rho_slopes)
    mom_l, mom_r = createLinearRepresentation(mom,mom_slopes)
    erg_l, erg_r = createLinearRepresentation(erg,erg_slopes)

    #Compute left and right states
    states_l = [deepcopy(i) for i in states] #initialize 
    states_r = [deepcopy(i) for i in states]
    for i in range(len(rho_l)):
        states_l[i].updateState(rho_l[i], mom_l[i], erg_l[i])
        states_r[i].updateState(rho_r[i], mom_r[i], erg_r[i])

    #Initialize predicited conserved quantities
    rho_p = [0.0 for i in range(len(rho))]
    mom_p = [0.0 for i in range(len(rho))]
    erg_p = [0.0 for i in range(len(rho))]

    #Advance in time each edge variable
    for i in range(len(rho)):

        #rho
        rho_p[i] = advCons(rho[i],dx,0.5*dt,rhoFlux(states_l[i]),rhoFlux(states_r[i])) 

        #mom
        mom_p[i] = advCons(mom[i],dx,0.5*dt,momFlux(states_l[i]),momFlux(states_r[i])) 

        #erg
        erg_p[i] = advCons(erg[i],dx,0.5*dt,ergFlux(states_l[i]),ergFlux(states_r[i])) 
        
    #Advance the primitive variables
    for i in range(len(rho)):
        states[i].updateState(rho_p[i], mom_p[i], erg_p[i])

    #Return states at left and right values
    return states

    
#-------------------------------------------------------------------------------------
#
## Corrector solver for hydro.
#
# The corrector solve takes in a predicted state at dt/2, and computes new values at
# dt.  The input is averages and slopes, the output is new averages, with the slopes
# un-adjusted.
#
# The slopes are defined based on the following relation in a cell:
# 
# \f$U(x) = U_a + \frac{2U_x}{h_x}(x - x_i) \f$
#
# Thus, \f$U_R = U_a + U_x\f$ and \f$U_L = U_a - U_x\f$, and 
# \f$U_x = \frac{U_R - U_L}{2}\f$
#
# @param[in] mesh           Basic spatial mesh object
# @param[in] states_old_a   Averages at old state. 
# @param[in] states_l       Predicted values at left nodes, at dt/2
# @param[in] states_r       Predicted values at right nodes, at dt/2
# @param[in] delta_t        time step size for this hydro solve. To predict values at 
#                           0.5 dt, pass in 1.0 dt
# 
#
# @return
#       -#  predicted states averages
#
def hydroCorrector(mesh, states_old_a, dt, bc):

        #for state in states_a:
        #   state.printConservativeVariables()
        #break
    #Choose riemann solver
    riem_solver = HLLCSolver #HLLSolver, HLLCSolver

    #Solve for fluxes and values at faces
    n = mesh.n_elems
    rho_F = np.zeros(n+1)
    mom_F = np.zeros(n+1)
    erg_F = np.zeros(n+1)

    #Create vectors of predicted variables
    rho_p = [s.rho                     for s in states_old_a]
    mom_p = [s.rho*s.u                 for s in states_old_a]
    erg_p = [s.rho*(0.5*s.u*s.u + s.e) for s in states_old_a]

    # get boundary values and states
    rho_BC_L, rho_BC_R, mom_BC_L, mom_BC_R, erg_BC_L, erg_BC_R =\
       bc.getBoundaryValues()
    state_BC_L, state_BC_R = bc.getBoundaryStates()

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
            state_R = states_old_a[i]
            #print "%f %f %f %f %f %f" % (rho_L, rho_R, mom_L, mom_R, erg_L, erg_R)
            #print state_L
            #print state_R
            #print riem_solver(rho_L, rho_R, state_L, state_R, rhoFlux)
            #print riem_solver(mom_L, mom_R, state_L, state_R, momFlux)
            #print riem_solver(erg_L, erg_R, state_L, state_R, ergFlux)
        elif i == n: # right boundary edge
            rho_L = rho_p[i-1]
            rho_R = rho_BC_R
            mom_L = mom_p[i-1]
            mom_R = mom_BC_R
            erg_L = erg_p[i-1]
            erg_R = erg_BC_R
            state_L = states_old_a[i-1]
            state_R = state_BC_R
            #print "%f %f %f %f %f %f" % (rho_L, rho_R, mom_L, mom_R, erg_L, erg_R)
            #print state_L
            #print state_R
            #print "R: ", rho_L, rho_R, mom_L, mom_R, erg_L, erg_R
            #print riem_solver(rho_L, rho_R, state_L, state_R, rhoFlux)
            #print riem_solver(mom_L, mom_R, state_L, state_R, momFlux)
            #print riem_solver(erg_L, erg_R, state_L, state_R, ergFlux)
        else: # interior edge
            rho_L = rho_p[i-1]
            rho_R = rho_p[i]
            mom_L = mom_p[i-1]
            mom_R = mom_p[i]
            erg_L = erg_p[i-1]
            erg_R = erg_p[i]
            state_L = states_old_a[i-1]
            state_R = states_old_a[i]
            #print "%f %f %f %f %f %f" % (rho_L, rho_R, mom_L, mom_R, erg_L, erg_R)
            #print state_L
            #print state_R

        # solve Riemann problem at interface
        rho_F[i] = riem_solver(rho_L, rho_R, state_L, state_R, rhoFlux)
        mom_F[i] = riem_solver(mom_L, mom_R, state_L, state_R, momFlux)
        erg_F[i] = riem_solver(erg_L, erg_R, state_L, state_R, ergFlux)

    #for i in xrange(0,n+1):
    #   print "%f %f %f" % (rho_F[i],mom_F[i],erg_F[i])
    #rho_F[0]  = rhoFlux(states_old_a[0])
    #rho_F[-1] = rhoFlux(states_old_a[-1])
    #mom_F[0]  = momFlux(states_old_a[0])
    #mom_F[-1] = momFlux(states_old_a[-1])
    #erg_F[0]  = ergFlux(states_old_a[0])
    #erg_F[-1] = ergFlux(states_old_a[-1])

    #Intialize cell average quantity arrays at t_old
    rho = [s.rho for s in states_old_a]
    mom = [s.rho*s.u for s in states_old_a]
    erg = [s.rho*(0.5*s.u**2. + s.e) for s in states_old_a]

    #Advance conserved values at centers based on edge fluxes
    for i in range(len(rho)):

        dx = mesh.getElement(i).dx

        #Example of edge fluxes:
        #   i is 0 for 1st element, so edge 0 and edge 1 is i and i+1
        rho[i] = advCons(rho[i],dx,dt,rho_F[i],rho_F[i+1])
        mom[i] = advCons(mom[i],dx,dt,mom_F[i],mom_F[i+1])
        erg[i] = advCons(erg[i],dx,dt,erg_F[i],erg_F[i+1])

    #Advance primitive variables
    states_a = [deepcopy(i) for i in states_old_a] 
    for i in range(len(states_a)):
        states_a[i].updateState(rho[i],mom[i],erg[i])

    return states_a

#------------------------------------------------------------------------------------
# Define some functions for evaluating fluxes for different state variables
#------------------------------------------------------------------------------------
def rhoFlux(s):
    return s.rho*s.u

def momFlux(s):
    return s.rho*s.u*s.u+s.p

def ergFlux(s):
    return (s.rho*(0.5*s.u*s.u+s.e) + s.p) * s.u


## Creates linear representation for solution using slopes
#
def createLinearRepresentation(u,slopes):

   u_l = [u[i] - 0.5*slopes[i] for i in xrange(len(u))]
   u_r = [u[i] + 0.5*slopes[i] for i in xrange(len(u))]
   return u_l, u_r

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
        return F_r

    elif S_l <= 0.0 and S_star > 0.0:
        return F_lstar

    elif S_star <= 0.0 and S_r > 0.0:
        return F_rstar

    elif S_l > 0.0:
        return F_l

    else:
        print S_l, S_star, S_r
        raise ValueError("HLLC solver produced unrealistic fluxes\n")


#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

