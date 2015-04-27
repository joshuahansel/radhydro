import numpy as np #numpy module has all the goodies
import matplotlib as plt #alias module as 'plt'
from pylab import *
from math import sqrt, isinf
from copy import deepcopy
from utilityFunctions import *


#--------------------------------------------------------------------------------------
## Main is my original hydro code, we shouldnt need this anymore but it is here
# temporarily in case we need it to compare
def main():
    
    #Python requires indentation for nested functions etc.  It is best to use a text
    #editor that allows for tabs to be expanded as spaces
    width = 1.0
    t_end = 0.05

    t = 0.0
    cfl = 0.5
    n = 500
    dx = width/n
    gamma = 1.4 #gas constant

    if n % 2 != 0: #simple error raise, % has lots of diff meanings in python
        raise ValueError("Must be even number of cells")

    #Choose riemann solver
    riem_solver = HLLCSolver #HLLSolver, HLLCSolver

    #Left and right BC and initial values
    p_left = 1.0
    u_left  = .750
    rho_left = 1.

    p_right = 0.1
    u_right = 0.0
    rho_right = 0.125

    #pick initial time step
    c_init = max(sqrt(gamma*p_left/rho_left)+u_left,sqrt(gamma*p_right/rho_right)+u_right)
    dt_init = cfl*dx/c_init
    print "new dt: ", dt_init

    #Can create lists of things in multiple ways
    spat_coors = []
    xi = 0.0
    for i in range(n+1): #range(3) = [0,1,2], "for" iterates over "each" i "in" list
        spat_coors.append(xi)
        xi += dx
    x = np.array(spat_coors) #Similar to a matlab vector
    x_cent = [0.5*(x[i]+x[i+1]) for i in range(len(x)-1)] #for plotting cell centers

 
    i_left = int(0.3*n)
    i_right = int(0.7*n)

    #Create cell centered variables
    states = [HydroState(u=u_left,p=p_left,gamma=gamma,rho=rho_left) for i in range(i_left)]
    states = states + [HydroState(u=u_right,p=p_right,gamma=gamma,rho=rho_right) for i in
            range(i_right)]

    #-----------------------------------------------------------------
    # Solve Problem
    #----------------------------------------------------------------
    
    #initialize dt
    dt = dt_init

    #loop over time steps
    while (t < t_end):
        t += dt

        #shorten last step to be exact
        if (t > t_end):
            t -= dt
            dt = t_end - t + 0.000000001
            t += dt

        #Create vectors of conserved quantities
        rho = [s.rho for s in states]
        mom = [s.rho*s.u for s in states]
        erg = [s.rho*(0.5*s.u*s.u + s.e) for s in states]

        #Predict values by advecting each thing
        rho_l, rho_r = slopeReconstruction(rho)
        mom_l, mom_r = slopeReconstruction(mom)
        erg_l, erg_r = slopeReconstruction(erg)


        #Compute left and right states
        states_l = [deepcopy(i) for i in states] #initialize 
        states_r = [deepcopy(i) for i in states]

        for i in range(len(rho_l)):
            states_l[i].updateState(rho_l[i], mom_l[i], erg_l[i])
            states_r[i].updateState(rho_r[i], mom_r[i], erg_r[i])


        #plotHydroSolutions(x,states=states_l)
        #plotHydroSolutions(x,states=states_r)
    
        #Check for positivity in all vars
 #       if not all( [(i > 0.0) for i in rho_l+rho_r+erg_l+erg_r] ):
  #          raise ValueError("Have a negative conserved quantity")


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

    #    plotHydroSolutions(x,states=states_l)
    #    plotHydroSolutions(x,states=states_r)

        #Solve for fluxes and values at faces
        rho_F = [0.0 for i in range(n+1)]
        mom_F = [0.0 for i in range(n+1)]
        erg_F = [0.0 for i in range(n+1)]

        #Solve Rieman problem at each face, for each quantity
        #For boundaries it is easily defined
        rho_F[0] = rhoFlux(states_l[0])
        mom_F[0] = momFlux(states_l[0])
        erg_F[0] = ergFlux(states_l[0])

        rho_F[-1] = rhoFlux(states_r[-1])
        mom_F[-1] = momFlux(states_r[-1])
        erg_F[-1] = ergFlux(states_r[-1])

        for i in range(0,n-1):

            rho_F[i+1] = riem_solver(rho_r_p[i], rho_l_p[i+1], states_r[i],
                    states_l[i+1], rhoFlux)
            mom_F[i+1] = riem_solver(mom_r_p[i], mom_l_p[i+1], states_r[i],
                    states_l[i+1], momFlux)
            erg_F[i+1] = riem_solver(erg_r_p[i], erg_l_p[i+1], states_r[i],
                    states_l[i+1], ergFlux)


   #     plt.figure(3)
  #      plt.plot(x,rho_F)
 #       plt.plot(x,mom_F)
#        plt.plot(x,erg_F)

        #Advance conserved values based on edge fluxes
        for i in range(len(rho)):

            rho[i] = advCons(rho[i],dx,dt,rho_F[i],rho_F[i+1])
            mom[i] = advCons(mom[i],dx,dt,mom_F[i],mom_F[i+1])
            erg[i] = advCons(erg[i],dx,dt,erg_F[i],erg_F[i+1])


        #figure(3)
      #  plt.plot(x_cent,rho)
      #  plt.plot(x_cent ,mom)
      #  plt.plot(x_cent,erg)
      #  plt.show(block=False)

        #Advance primitive variables
        for i in range(len(states)):
            states[i].updateState(rho[i],mom[i],erg[i])
        
        #plotHydroSolutions(x,states=states)
        
        #Compute a new time step
        c = [sqrt(i.p*i.gamma/i.rho)+abs(i.u) for i in states]
        dt_vals = [cfl*(x[i+1]-x[i])/c[i] for i in range(len(c))]
        dt = min(dt_vals)
        print "new dt:", dt

    #plot solution, reads latex commands
    #plotHydroSolutions(x,states=states) 
    """ keyword arguments are cool.  After the
        standard "pass by reference" variables, the keyword varialbes can be specified in
        any order.  It is a little confusing since the keywords can have the same name as
        the vars, i.e., u (keyword) = u(variable)"""
    
    for i in states:
        print i


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
#
# @return
#       -#  predicted states at averages
#       -#  predicted states slopes
#
def hydroPredictor(mesh, states_old_a, dt):

    dx = mesh.getElement(0).dx #currently a fixed width

    if mesh.n_elems % 2 != 0: #simple error raise, % has lots of diff meanings in python
        raise ValueError("Must be even number of cells")

    #Can create lists of things in multiple ways
    spat_coors = []
    for i in mesh.elements:
        spat_coors += [i.xl]
    spat_coors += [mesh.elements[-1].xr]
    x = np.array(spat_coors) #Similar to a matlab vector
    x_cent = [0.5*(x[i]+x[i+1]) for i in range(len(x)-1)] #for plotting cell centers

    #Intialize cell centered variables as passed in
    states = states_old_a

    #-----------------------------------------------------------------
    # Solve Problem
    #----------------------------------------------------------------

    #Create vectors of conserved quantities
    rho = [s.rho for s in states]
    mom = [s.rho*s.u for s in states]
    erg = [s.rho*(0.5*s.u*s.u + s.e) for s in states]

    #Predict values by advecting each thing
    rho_l, rho_r = slopeReconstruction(rho)
    mom_l, mom_r = slopeReconstruction(mom)
    erg_l, erg_r = slopeReconstruction(erg)

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


    #Return states at left and right values
    return states_l, states_r

    
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
def hydroCorrector(mesh, states_old_a, states_l, states_r, dt):

    #Choose riemann solver
    riem_solver = HLLCSolver #HLLSolver, HLLCSolver

    #Initialize the averages by copying left and right states
    states_a = [deepcopy(i) for i in states_l] 

    #Solve for fluxes and values at faces
    n = mesh.n_elems
    rho_F = np.zeros(n+1)
    mom_F = np.zeros(n+1)
    erg_F = np.zeros(n+1)

    #Create vector of predicted variables at edges
    rho_l_p = [s.rho for s in states_l]
    rho_r_p = [s.rho for s in states_r]
    mom_l_p = [s.rho*s.u for s in states_l]
    mom_r_p = [s.rho*s.u for s in states_r]
    erg_l_p = [s.rho*(0.5*s.u*s.u + s.e) for s in states_l]
    erg_r_p = [s.rho*(0.5*s.u*s.u + s.e) for s in states_r]

    #Solve Rieman problem at each face, for each quantity
    #For boundaries it is easily defined
    rho_F[0] = rhoFlux(states_l[0])
    mom_F[0] = momFlux(states_l[0])
    erg_F[0] = ergFlux(states_l[0])

    rho_F[-1] = rhoFlux(states_r[-1])
    mom_F[-1] = momFlux(states_r[-1])
    erg_F[-1] = ergFlux(states_r[-1])

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
    for i in range(len(states_a)):
        states_a[i].updateState(rho[i],mom[i],erg[i])

    return states_a


#--------------------------------------------------------------------------------------
class HydroState:
    'This class handles all states at a point'

    #Constructor, all function definitions must be passed 'self', but when called
    #outside of the class you do not need to pass it in, it is implied.
    #define that it is a member function

    #constructor
    def __init__(self, u=None ,rho=None,p=None,gamma=None):
        self.u = u
        self.rho = rho
        self.e = getIntErg(gamma,rho,p)
        self.p = p
        self.gamma = gamma

    #solve for new values based on a consState variables
    def updateState(self, rho, mom, erg):

        self.rho = rho
        self.u = mom/rho
        self.e = erg/rho - 0.5*self.u*self.u
        self.p = getPressure(self.gamma, self.rho, self.e) 

    def getSoundSpeed(self):

        return sqrt(self.gamma*self.p/self.rho)
    
    #Defin fancy function to compare values
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    #Define a function for printing out the class
    def __str__(self):

        return "u: %.4f rho: %.4f e: %.4f p: %.4f" % (self.u, self.rho, self.e, self.p)


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

#------------------------------------------------------------------------------------
# Create function for advancing conserved quantities in time
#------------------------------------------------------------------------------------
def advCons(val, dx, dt, f_left, f_right):

    return val + dt/dx*(f_left - f_right)
    


#Can also have non class functions like usual
#------------------------------------------------------------------------------------
def getVolume(x1,x2):

    return (x2-x1)*1.

#------------------------------------------------------------------------------------
def getPressure(gamma,rho,e):

    return (gamma-1.)*rho*e

#------------------------------------------------------------------------------------
def getIntErg(gamma, rho, p):

    return p/((gamma-1.)*rho)

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
def plotHydroSolutions(x,states=None): #Default func values is trivial

    plt.figure(figsize=(11,8.5))

    #get the exact values
#    f = open('exact_results.txt', 'r')
#    x_e = []
#    u_e = []
#    p_e = []
#    rho_e = []
#    e_e = []
#    for line in f:
#        if len(line.split())==1:
#            t = line.split()
#        else:
#            data = line.split()
#            x_e.append(float(data[0]))
#            u_e.append(float(data[1]))
#            p_e.append(float(data[2]))
#            rho_e.append(float(data[4]))
#            e_e.append(float(data[3]))


    if states==None:
        raise ValueError("Need to pass in states")
    else:
        u = []
        p = []
        rho = []
        e = []
        for i in states:
            u.append(i.u)
            p.append(i.p)
            rho.append(i.rho)
            e.append(i.e)

    #get edge values
    x_cent = x
    if len(x) == len(states)+1:
        x_cent = [0.5*(x[i]+x[i+1]) for i in xrange(len(x))]
    
    if u != None:
        plotSingle(x_cent,u,"$u$")
    
    if rho != None:
        plotSingle(x_cent,rho,r"$\rho$") 

    if p != None:
        plotSingle(x_cent,p,r"$p$")

    if e != None:
        plotSingle(x_cent,e,r"$e$")

    plt.show(block=False) #show all plots generated to this point
    raw_input("Press anything to continue...")
    plotSingle.fig_num=0

#-------------------------------------------------------------------------------------
def plotSingle(x,y,ylabl):

    #static variable counter
    plotSingle.fig_num += 1
    plt.subplot(2,2,plotSingle.fig_num)
    plt.xlabel('$x$ (cm)')
    plt.ylabel(ylabl)
    plt.plot(x,y,"b+-",label="My Stuff")
    plt.savefig("var_"+str(plot2D.fig_num)+".pdf")
    
plotSingle.fig_num=0


#-------------------------------------------------------------------------------------
def plot2D(x,y,x_ex,y_ex,ylabl):

    #static variable counter
    plot2D.fig_num += 1

    plt.subplot(2,2,plot2D.fig_num)
    plt.xlabel('$x$ (cm)')
    plt.ylabel(ylabl)
    plt.plot(x,y,"b+-",label="Lagrangian")
    plt.plot(x_ex,y_ex,"r--",label="Exact")
    plt.savefig("var_"+str(plot2D.fig_num)+".pdf")
    
plot2D.fig_num=0


#--------------------------------------------------------------------------------------
"""Python does not require a main, but by doing it this way, all functions will be
processed before it calls main, so no functions are undefined"""
if __name__ == "__main__":
    main()



