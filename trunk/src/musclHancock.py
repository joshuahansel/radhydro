
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
#
def hydroPredictor(mesh, states_old_a, slopes, dt):

    dx = mesh.getElement(0).dx #currently a fixed width

    if mesh.n_elems % 2 != 0: #simple error raise, % has lots of diff meanings in python
        raise ValueError("Must be even number of cells")

    #Can create lists of things in multiple ways
    spat_coors = []
    for i in mesh.elements:
        spat_coors += [i.xl]
    spat_coors += [mesh.elements[-1].xr]
    x = np.array(spat_coors)

    #Initialize cell centered variables as passed in
    states = deepcopy(states_old_a)

    #-----------------------------------------------------------------
    # Solve Problem
    #----------------------------------------------------------------

    #Create vectors of conserved quantities
    rho = [s.rho for s in states]
    mom = [s.rho*s.u for s in states]
    erg = [s.rho*(0.5*s.u*s.u + s.e) for s in states]
    rho_l, rho_r, mom_l, mom_r, erg_l, erg_r =\
                   slopes.createLinearRepresentation(states)

    #Compute left and right states
    states_l = [deepcopy(i) for i in states] #initialize 
    states_r = [deepcopy(i) for i in states]

    for i in range(len(rho_l)):
        states_l[i].updateState(rho_l[i], mom_l[i], erg_l[i])
        states_r[i].updateState(rho_r[i], mom_r[i], erg_r[i])

    #Initialize predicited conserved quantities
    rho_l_p = [0.0 for i in range(len(rho_l))]
    rho_r_p = [0.0 for i in range(len(rho_l))]
    mom_l_p = [0.0 for i in range(len(rho_l))]
    mom_r_p = [0.0 for i in range(len(rho_l))]
    erg_l_p = [0.0 for i in range(len(rho_l))]
    erg_r_p = [0.0 for i in range(len(rho_l))]

    #Advance in time each edge variable
    for i in range(len(rho_l)):

        #rho
        rho_l_p[i] = advCons(rho_l[i],dx,0.5*dt,rhoFlux(states_l[i]),rhoFlux(states_r[i])) 
        rho_r_p[i] = advCons(rho_r[i],dx,0.5*dt,rhoFlux(states_l[i]),rhoFlux(states_r[i])) 

        #mom
        mom_l_p[i] = advCons(mom_l[i],dx,0.5*dt,momFlux(states_l[i]),momFlux(states_r[i])) 
        mom_r_p[i] = advCons(mom_r[i],dx,0.5*dt,momFlux(states_l[i]),momFlux(states_r[i])) 

        #erg
        erg_l_p[i] = advCons(erg_l[i],dx,0.5*dt,ergFlux(states_l[i]),ergFlux(states_r[i])) 
        erg_r_p[i] = advCons(erg_r[i],dx,0.5*dt,ergFlux(states_l[i]),ergFlux(states_r[i])) 

    #Advance the primitive variables at the edges
    for i in range(len(rho_l)):
        states_l[i].updateState(rho_l_p[i], mom_l_p[i], erg_l_p[i])
        states_r[i].updateState(rho_r_p[i], mom_r_p[i], erg_r_p[i])


    #Return the new averages
    for i in range(len(states)):
        states[i].updateState(0.5*(rho_l_p[i]+rho_r_p[i]),
                           0.5*(mom_l_p[i]+mom_r_p[i]),
                           0.5*(erg_l_p[i]+erg_r_p[i]))


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
# @param[in] states_half_a  States evaluate at dt/2
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
def hydroCorrector(mesh, states_old_a, states_half, slopes_old, dt, bc):

    debug_mode = False

    #Choose riemann solver
    riem_solver = HLLCSolver #HLLSolver, HLLCSolver

    #Solve for fluxes and values at faces
    n = mesh.n_elems
    rho_F = np.zeros(n+1)
    mom_F = np.zeros(n+1)
    erg_F = np.zeros(n+1)

    #Create edge values
    rho_l_p, rho_r_p, mom_l_p, mom_r_p, erg_l_p, erg_r_p =\
                   slopes_old.createLinearRepresentation(states_half)

    # Create edge states
    states_l = deepcopy(states_half) 
    states_r = deepcopy(states_half)

    for i in range(len(rho_l_p)):

        states_l[i].updateState(rho_l_p[i], mom_l_p[i], erg_l_p[i])
        states_r[i].updateState(rho_r_p[i], mom_r_p[i], erg_r_p[i])



    # get boundary values and states
    rho_BC_L, rho_BC_R, mom_BC_L, mom_BC_R, erg_BC_L, erg_BC_R =\
       bc.getBoundaryValues()
    state_BC_L, state_BC_R = bc.getBoundaryStates()

    #Solve Rieman problem at each face, for each quantity
    #For boundaries it is easily defined
    
    #Check if it is reflective, if so we need to reset the states due to the way the
    #edge values work. This is kind of crappy coding
    if bc.bc_type == "reflective":

        rho_F[0] = rhoFlux(states_l[0])
        mom_F[0] = momFlux(states_l[0])
        erg_F[0] = ergFlux(states_l[0])

        rho_F[-1] = rhoFlux(states_r[-1])
        mom_F[-1] = momFlux(states_r[-1])
        erg_F[-1] = ergFlux(states_r[-1])

    else:

        rho_F[0] = riem_solver(rho_BC_L, rho_l_p[0], state_BC_L, states_l[0], rhoFlux)
        mom_F[0] = riem_solver(mom_BC_L, mom_l_p[0], state_BC_L, states_l[0], momFlux)
        erg_F[0] = riem_solver(erg_BC_L, erg_l_p[0], state_BC_L, states_l[0], ergFlux)

        if debug_mode:
           print "HI LEFT "  
           print "rho_F    ", rho_BC_L, rho_l_p[0], rho_F[0], rhoFlux(states_l[0]), rhoFlux(state_BC_L)
           print "mom_F    ", mom_BC_L, mom_l_p[0], mom_F[0], momFlux(states_l[0]), momFlux(state_BC_L)
           print "erg_F    ", erg_BC_L, erg_l_p[0], erg_F[0], ergFlux(states_l[0]), ergFlux(state_BC_L)

        rho_F[-1] = riem_solver(rho_BC_R, rho_r_p[-1], state_BC_R, states_r[-1], rhoFlux)
        mom_F[-1] = riem_solver(mom_BC_R, mom_r_p[-1], state_BC_R, states_r[-1], momFlux)
        erg_F[-1] = riem_solver(erg_BC_R, erg_r_p[-1], state_BC_R, states_r[-1], ergFlux)

        if debug_mode:
           print "HI RIGHT "  
           print "rho_F    ", "BC_value: ", rho_BC_R, rho_r_p[-1], "chosen F", rho_F[-1], rhoFlux(states_r[-1]), rhoFlux(state_BC_R)
           print "mom_F    ", "BC_value: ", mom_BC_R, mom_r_p[-1], "chosen F", mom_F[-1], momFlux(states_r[-1]), momFlux(state_BC_R)
           print "erg_F    ", "BC_value: ", erg_BC_R, erg_r_p[-1], "chosen F", erg_F[-1], ergFlux(states_r[-1]), ergFlux(state_BC_R)

    #Do the interior cells
    for i in range(0,n-1):

        rho_F[i+1] = riem_solver(rho_r_p[i], rho_l_p[i+1], states_r[i],
                states_l[i+1], rhoFlux)
        mom_F[i+1] = riem_solver(mom_r_p[i], mom_l_p[i+1], states_r[i],
                states_l[i+1], momFlux)
        erg_F[i+1] = riem_solver(erg_r_p[i], erg_l_p[i+1], states_r[i],
                states_l[i+1], ergFlux)


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

