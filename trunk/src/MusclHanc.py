import numpy as np #numpy module has all the goodies
import matplotlib as plt #alias module as 'plt'
from pylab import *
from math import sqrt, isinf
from copy import deepcopy

#--------------------------------------------------------------------------------------
def main():
    
    #Python requires indentation for nested functions etc.  It is best to use a text
    #editor that allows for tabs to be expanded as spaces
    width = 1.0
    t_end = 0.2

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
    print "Initial dt: ", dt_init

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

        #plotSolutions(x,states=states_l)
        #plotSolutions(x,states=states_r)
    
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

    #    plotSolutions(x,states=states_l)
    #    plotSolutions(x,states=states_r)

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
        
        #plotSolutions(x,states=states)


        #Compute a new time step
        c = [sqrt(i.p*i.gamma/i.rho)+abs(i.u) for i in states]
        dt_vals = [cfl*(x[i+1]-x[i])/c[i] for i in range(len(c))]
        dt = min(dt_vals)
        print "new dt:", dt
        
    #plot solution, reads latex commands
    plotSolutions(x,states=states) 
    """ keyword arguments are cool.  After the
        standard "pass by reference" variables, the keyword varialbes can be specified in
        any order.  It is a little confusing since the keywords can have the same name as
        the vars, i.e., u (keyword) = u(variable)"""
            
        

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
        raise ValueError("HLL solver produced unrealistic fluxes\n")

#-------------------------------------------------------------------------------------
def getArtVisc(x,u,rho,p,gamma):

    q = [0.0 for i in p] #initialize to zero
    for i in range(len(p)):
        if u[i+1] >= u[i]: #this has machine error, but doesnt matter cause those values are zero
            continue
        
        #Do wierd shit at boundaries
        if i==0:
        
            #calculate the terribleness
            del_x_i = x[i+1]-x[i]
            del_x_rt = x[i+2]-x[i+1]

            rho_lt = rho[i]
            rho_rt = (rho[i]*del_x_i    + rho[i+1]*del_x_rt)/(del_x_i +del_x_rt)
            
            c_lt = sqrt(gamma*p[i]/rho[i])
            c_rt = ( sqrt(gamma*p[i+1]/rho[i+1])*del_x_rt 
                 +   sqrt(gamma*p[i]/rho[i])*del_x_i ) / (del_x_rt + del_x_i)

            del_u_i = u[i+1] - u[i]
            del_u_rt = u[i+2] - u[i+1]

            R_lt = 1.
            R_rt = del_u_rt*del_x_i/(del_x_rt*del_u_i)
        
        elif i==len(p)-1:

            #calculate the terribleness
            del_x_i = x[i+1]-x[i]
            del_x_lt = x[i]-x[i-1]

            rho_lt = (rho[i-1]*del_x_lt +    rho[i]*del_x_i)/(del_x_lt+del_x_i )
            rho_rt = rho[i]
            
            c_lt = ( sqrt(gamma*p[i-1]/rho[i-1])*del_x_lt 
                 +   sqrt(gamma*p[i]/rho[i])*del_x_i ) / (del_x_lt + del_x_i)

            c_rt = sqrt(gamma*p[i]/rho[i])

            del_u_i = u[i+1] - u[i]
            del_u_lt = u[i] - u[i-1]

            R_lt = del_u_lt*del_x_i/(del_x_lt*del_u_i)
            R_rt = 1.

        else:

            #calculate the terribleness
            del_x_i = x[i+1]-x[i]
            del_x_lt = x[i]-x[i-1]
            del_x_rt = x[i+2]-x[i+1]

            rho_lt = (rho[i-1]*del_x_lt +    rho[i]*del_x_i)/(del_x_lt+del_x_i )
            rho_rt = (rho[i]*del_x_i    + rho[i+1]*del_x_rt)/(del_x_i +del_x_rt)
            
            c_lt = ( sqrt(gamma*p[i-1]/rho[i-1])*del_x_lt 
                 +   sqrt(gamma*p[i]/rho[i])*del_x_i ) / (del_x_lt + del_x_i)

            c_rt = ( sqrt(gamma*p[i+1]/rho[i+1])*del_x_rt 
                 +   sqrt(gamma*p[i]/rho[i])*del_x_i ) / (del_x_rt + del_x_i)
                           
            del_u_i = u[i+1] - u[i]
            del_u_rt = u[i+2] - u[i+1]
            del_u_lt = u[i] - u[i-1]

            R_lt = del_u_lt*del_x_i/(del_x_lt*del_u_i)
            R_rt = del_u_rt*del_x_i/(del_x_rt*del_u_i)

    
        #same for all cells:
        rho_bar = 2.*rho_lt*rho_rt/(rho_lt + rho_rt)
        c_q = 0.25*(gamma+1.)
        c_bar = min(c_lt,c_rt)
        min1 = min(0.5*(R_lt+R_rt),2.*R_rt)
        min2 = min(1.,2.*R_lt)
        GAM = max( 0., min(min1,min2) )

        q[i] = (1.-GAM)*rho_bar*abs(del_u_i)*( c_q*abs(del_u_i)
                + sqrt(c_q*c_q*del_u_i*del_u_i+c_bar) )
                        

    return q



#-------------------------------------------------------------------------------------
def plotSolutions(x,states=None): #Default func values is trivial

    plt.figure(figsize=(11,8.5))

    #get the exact values
    f = open('exact_results.txt', 'r')
    x_e = []
    u_e = []
    p_e = []
    rho_e = []
    e_e = []
    for line in f:
        if len(line.split())==1:
            t = line.split()
        else:
            data = line.split()
            x_e.append(float(data[0]))
            u_e.append(float(data[1]))
            p_e.append(float(data[2]))
            rho_e.append(float(data[4]))
            e_e.append(float(data[3]))


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
    x_cent = [0.5*(x[i]+x[i+1]) for i in range(len(x)-1)]
    
    if u != None:
        plot2D(x_cent,u,x_e,u_e,"$u$")
    
    if rho != None:
        plot2D(x_cent,rho,x_e,rho_e,r"$\rho$") 

    if p != None:
        plot2D(x_cent,p,x_e,p_e,r"$p$")

    if e != None:
        plot2D(x_cent,e,x_e,e_e,r"$e$")

    plt.show(block=False) #show all plots generated to this point
    raw_input("Press anything to continue...")
    plot2D.fig_num=0

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



