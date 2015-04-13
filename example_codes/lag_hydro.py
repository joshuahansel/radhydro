import numpy as np #numpy module has all the goodies
import matplotlib as plt #alias module as 'plt'
from pylab import *
from math import sqrt

#--------------------------------------------------------------------------------------
def main():
    
    #Python requires indentation for nested functions etc.  It is best to use a text
    #editor that allows for tabs to be expanded as spaces
    width = 1.0
    t_end = 0.2
        
    t = 0.0
    cfl = 0.5
    n = 400
    dx = width/n
    gamma = 1.4 #gas constant

    if n % 2 != 0: #simple error raise, % has lots of diff meanings in python
        raise ValueError("Must be even number of cells")

    #Left and right BC and initial values
    p_left = 1.0
    p_right = 0.1
    u_left  = 0.7500
    u_right = 0.0
    rho_left = 1.
    rho_right = .125

    #pick initial time step
    c_init = sqrt(gamma*p_left/rho_left)
    dt_init = cfl*dx/c_init
    print "Initial dt: ", dt_init

    #Can create lists of things in multiple ways
    spat_coors = []
    xi = 0.0
    for i in range(n+1): #range(3) = [0,1,2], "for" iterates over "each" i "in" list
        spat_coors.append(xi)
        xi += dx
    x = np.array(spat_coors) #Similar to a matlab vector
    x_old = np.array(x) #BE CAREFUL! If you just did x_old = x, x_old and x 
        #'name' the same variable, and changing one would change the other, 
        #until you redefined one. Doesnt work like a pointer though
    
    #print out the spat_coors
#    for i in spat_coors:
#        print i
    i_left = int(0.3*n)
    i_right = int(0.7*n)
    u = [u_left for i in range(i_left)]
    u = u + [u_right for i in range(i_right+1)] #concatenate list u with new list specified
    u[i_left] = u_left
    rho = [rho_left for i in range(i_left)]
    rho = rho + [rho_right for i in range(i_right)] 
    p = [p_left for i in range(i_left)]
    p = p + [p_right for i in range(i_right)] 

    #set mass and energy to be consistent with x's, densities, and pressures
    m = [ rho[i]*(x[i+1]-x[i]) for i in range(n)]   #defined at cell centers, should not change
    e = [ p[i]/((gamma-1.)*rho[i]) for i in range(n)] #defined at centers

    #Same for vars at old time step
    rho_old = list(rho) #be sure to call 'list', or a copy of the name is made, not a copy of the data
    p_old = list(p)
    u_old = list(u) #only defined at faces
    e_old = list(e) #defined at centers

    #store predicted velocities
    u_pred = [0.0 for i in range(n+1)]

    #Need to store sums of initial conditions for balance computation
    ke_init = sum([0.5*u[i]*u[i]*0.5*(m[i]+m[i-1]) for i in range(1,len(u)-1)])
    e_init  = sum([m[i]*e[i] for i in range(len(e))])
    bc_fluxes = 0.0 #this will be summed each step

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

        """Predictor step followed by corrector step
            -multiline comments- """
        for loop in range(2):

            #use old values for pressure
            if loop == 0:
                p = np.array(p_old) #redundant but just for clarity
        
            #Add artifical visocity to non-boundary pressures if necessary
            Q = getArtVisc(x_old,u_old,rho_old,p_old,gamma)
            p = np.array([p[i]+Q[i] for i in range(len(p))])

            #Solve BC first for u_l and u_r
            m_left = 0.5*m[0]
            u[0] = u_old[0]+dt/m_left*(p[0] - p_left)
            u[0] = u_left

            m_right = 0.5*m[-1] #access last member in m
            u[-1] = u_old[-1]+dt/m_right*(p_right - p[-1])
            u[-1] = u_right


            #update all velocities and positions first
            for i in xrange(1,len(x)-1): #loop over spatial cells
                m_half = 0.5*(m[i]+m[i-1])
                u[i] = -1.*dt/m_half*(p[i] - p[i-1]) + u_old[i] # velocities

            #get new positions, after predicted velocities
            for i in xrange(len(x)):
                u_pred[i] = 0.5*(u[i]+u_old[i])
                x[i] = x_old[i] + dt*u_pred[i] #position

            #get new densities and energies
            for i in xrange(len(rho)):
                vol = getVolume(x[i],x[i+1])
                rho[i] = m[i]/vol      # density
                e[i] = -1.*(p[i]*dt/m[i])*(u_pred[i+1] - u_pred[i]) + e_old[i]
                
            #get new pressures
            for i in xrange(len(p)):
                p[i] = getPressure(gamma,rho[i],e[i])

            #Store predicted pressures in p for the corrector step to limit code change
            if loop == 0:
                p = np.array([0.5*(p[i] + p_old[i]) for i in range(len(p))]) 


        #Compute balance
        ke_t = sum([0.5*u[i]*u[i]*0.5*(m[i]+m[i-1]) for i in range(1,len(u)-1)])
        e_t =  sum([m[i]*e[i] for i in range(len(m))])
        bc_fluxes += dt*(p_left*u[0] - p_right*u[-1])

            
        #Store new solution as old solution for next time step
        u_old = np.array(u) #notice here I am copying u's content, not the name
        p_old = np.array(p)
        e_old = np.array(e)
        rho_old = np.array(rho)
        x_old = np.array(x)

        #Compute CFL number
        c = [sqrt(p[i]*gamma/rho[i]) for i in range(len(p))]
        dt_vals = [cfl*(x[i+1]-x[i])/c[i] for i in range(len(p))]
        dt = min(dt_vals)
         

    print "Balance after final step: ", (e_t+ke_t - (ke_init + e_init) - (bc_fluxes)
                )/(ke_init+e_init)
    
    #plot solution, reads latex commands
    plotSolutions(x,u=u,rho=rho,p=p,e=e) 
    """ keyword arguments are cool.  After the
        standard "pass by reference" variables, the keyword varialbes can be specified in
        any order.  It is a little confusing since the keywords can have the same name as
        the vars, i.e., u (keyword) = u(variable)"""
            
        

#--------------------------------------------------------------------------------------
class DOFHandler:
    'This class handles arbitrary DOF for simple 1D mesh'

    #Constructor, all function definitions must be passed 'self', but when called
    #outside of the class you do not need to pass it in, it is implied.
    #define that it is a member function
    def __init__(self): 
        self.lt = 0.0 #Class DOFHandler has member 'lt' (left point in cell)
        self.rt = 0.0

    #Overloaded constructor to take in left and right value
    def __init__(self, lt_val, rt_val):
        self.lt = lt_val
        self.rt = rt_val

    #Define a function for printing out the class
    def __str__(self):
        return "lt: %.4f rt: %.4f" % (float(self.lt), float(self.rt))

    #Get difference
    def getDelta(self):
        return self.rt - self.lt


#Can also have non class functions like usual
#-------------------------------------------------------------------------------------
def getVolume(x1,x2):

    return (x2-x1)*1.

#-------------------------------------------------------------------------------------
def getPressure(gamma,rho,e):

    return (gamma-1.)*rho*e

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
                        
#        if i != 55:
#
#            print rho_bar
#            print c_q
#            print c_bar
#            print "min1", min1
#            print "min2", min2
#            print 0.5*(R_lt+R_rt), 2.*R_rt, 2*R_lt, 1
#            print "Actual min:", GAM
#            print del_u_i
#            print GAM
#
#            print q[i]

 #       if i == 0 or i==len(p)-1:

#            q[i] = 0.00001


    return q



#-------------------------------------------------------------------------------------
def plotSolutions(x,u=None,rho=None,p=None,e=None): #Default func values is trivial


    plt.figure(figsize=(11,8.5))

    #get edge values
    x_edge = [0.5*(x[i]+x[i+1]) for i in range(len(x)-1)]

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
    
    if u != None:
        plot2D(x,u,x_e,u_e,"$u$")
    
    if rho != None:
        plot2D(x_edge,rho,x_e,rho_e,r"$\rho$") 

    if p != None:
        plot2D(x_edge,p,x_e,p_e,r"$p$")

    if e != None:
        plot2D(x_edge,e,x_e,e_e,r"$e$")

    plt.show(block=False) #show all plots generated to this point
    raw_input("Press anything to continue...")

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



