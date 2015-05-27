## @package src.sourceHandlers
# This file contains all of the source handler functions 
#
# Each source is responsible for implementing several functions to build the approps
# source for each of the time stepping methods.  The derived classes will, in
# general, only need to define each of several ``virtual functions''.  Each derived
# class inherits a "buildSource" function. This is the primary function, responsible
# for building the entire right hand side for that term in the equation. 
#
# In general, each derived class is responsible for overriding the different
# functions for the different time stepping methods. Each source term, called it f,
# will be its own derived class.  Most classes will use the default BuildSource
# function, which calls several functions to help build the source on the RHS. In
# particular, they assume the following forms of each of the time stepping methods.
#
# BE: 
# \f[
#   \frac{Y^{n+1} - Y^{n}}{c\Delta t} = A^{n+1}
# \f]
#
# CN: 
# \f[
#   \frac{Y^{n+1} - Y^{n}}{c\Delta t} = \frac{1}{2}\left(A^{n+1} + A^n\right)
# \f]
#
# BDF2: 
# \f[
#   \frac{Y^{n+1} - Y^{n}}{c\Delta t} = \frac{2}{3}A^{n+1} + \frac{1}{6} A^{n} +
#   \frac{1}{6} A^{n-1}
# \f]
# for the general problem 
# \f[
#   \frac{\partial Y}{c\partial t} = A[ Y(t)]
# \f]
# where \f$A\f$ is a operator that is a function of \f$Y,t\f$. NOTE: we keep the
# \f$1/c\Delta t\f$ term on the left hand terms. Each derived class is thus
# responsible for writing functions that evaluate \f$AY^n,\;AY^{n-1},\f$ and
# \f$AY^{n+1}\f$, which are functions evalOld, evalOlder, and evalImplicit,
# respectively.
#
# In implementation, \f$Y^n/c\Delta t\f$
# is also on the right hand side of the equation as a source term. This is an example
# (probably the only one) of a term that its derived class will override buildSource,
# rather than implement the eval*** functions because the term is the same for all
# stepping algorithms


import re
import numpy as np
from mesh import Mesh
import globalConstants as GC
from utilityFunctions import getIndex, getLocalIndex
from math import sqrt

#====================================================================================
## Base class for source handlers. More info given in package documentation. Here, 
#  the evaluate functions are only implemented to raise errors
#  in case developer incorrectly creates a derived class
class SourceHandler:

    #---------------------------------------------------------------------------
    ## Basic constructor
    def __init__(self, mesh, dt, time_stepper):

        self.mesh = mesh #copy mesh
        self.dt   = dt   #current time step size
        self.time_stepper = time_stepper


        #Determine which time stepping function to use in derived class
        ts = time_stepper
        self.func = None
        if re.search("CN", ts):
            self.func = self.evalCN
        elif re.search("BDF2", ts):
            self.func = self.evalBDF2
        else:
            self.func = self.evalBE

    #----------------------------------------------------------------------------
    ## Function to evaluate source at all cells in the mesh. This is the main 
    #  function to be called on all sources.  
    def buildSource(self, **kwargs):

        #Loop over all cells and build source 
        Q = np.array([])
        for i in range(self.mesh.n_elems):
            
            Q_elem = self.func(i, **kwargs) #Append source of element i
            Q = np.append(Q, Q_elem)          

        return Q

    #----------------------------------------------------------------------------
    ## Function to evaluate source in CN time stepping, for element el
    def evalCN(self, el, **kwargs):

        return 0.5*(self.evalOld(el, **kwargs) + self.evalImplicit(el, **kwargs))

    #----------------------------------------------------------------------------
    ## Function to evaluate source backward Euler time stepping, for element el
    def evalBE(self, el, **kwargs):

        return self.evalImplicit(el, **kwargs)

    #----------------------------------------------------------------------------
    ## Function to evaluate source in BDF2 time stepping, for element el
    def evalBDF2(self, el, **kwargs):

        return 1./6.*( self.evalOld(el, **kwargs) + self.evalOlder(el, **kwargs) ) \
                 + 2./3.*(self.evalImplicit(el, **kwargs))

    #----------------------------------------------------------------------------
    ## Function to evaluate the implicit term, if it occurs on right hand side of
    #  equation. For example, the streaming term has no implicit term on the RHS 
    def evalImplicit(self, el, **kwargs):

        raise NotImplementedError("You must define the evalImplicit, or entire build"             
            "source function in the derived class")

    #----------------------------------------------------------------------------
    ## Function to evaluate the old term at \f$t_n\f$. For example, in the CN system,
    #  this is \f$A(Y^n)\f$.  All terms should have this function 
    def evalOld(self, el, **kwargs):

        raise NotImplementedError("You must define the evalOld, or entire build"
            "source function in the derived class")

    #----------------------------------------------------------------------------
    ## Function to evaluate the oldest term, only used by BDF2, i.e.,  at \f$t_n-1\f$. 
    #  All terms should have this function implemented.
    def evalOlder(self, el, **kwargs):

        raise NotImplementedError("You must define the evalOlder, or entire build"
           "source function in the derived class")

#===========================================================================================
# All derived classes
#===========================================================================================

## Derived class for the old intensity
#
#  This is the term resulting from discretization of time derivative, i.e., T_n
class OldIntensitySrc(SourceHandler):
    
    ## Constructor just needs to call base class constructor in this case
    def __init__(self, *args):

        SourceHandler.__init__(self, *args)
        self.c_dt = GC.SPD_OF_LGT * self.dt
    
    #-------------------------------------------------------------------------------
    ## Override the build source function.  Time derivative source term is the same in all 
    # cases (always just psi/(c dt) ), so just make one function used by any type of
    # time derivative.
    def buildSource(self, psi_minus_old=None, psi_plus_old=None, **kwargs):

        #Loop over all cells and build source 
        Q = np.array([])
        for i in range(self.mesh.n_elems):
            
            Q_elem =  self.evaluate(i, psi_minus_old=psi_minus_old,
                    psi_plus_old=psi_plus_old)    #Evaluate source of element i
            Q = np.append(Q, Q_elem)                  #Append source from element i

        return Q

    #----------------------------------------------------------------------------
    ## Time derivative source term is the same in all cases (always just psi/(c dt) ), so just make one function
    #  called by all forms of build source
    def evaluate(self, i, psi_minus_old=None, psi_plus_old=None):
        
        #Get all the indices, this is basically redundant
        iLminus     = getLocalIndex("L","-") # dof i,  L,-
        iLplus      = getLocalIndex("L","+") # dof i,  L,+
        iRminus     = getLocalIndex("R","-") # dof i,  R,-
        iRplus      = getLocalIndex("R","+") # dof i,  R,l+

        Q  = np.zeros(4)
        Q[iLminus] = psi_minus_old[i][0]    
        Q[iRminus] = psi_minus_old[i][1] 
        Q[iLplus] = psi_plus_old[i][0]   
        Q[iRplus] = psi_plus_old[i][1]
        return Q*(1./self.c_dt)


#====================================================================================
## Derived class for Streaming term:  \mu d\psi/dx 
class StreamingSrc(SourceHandler):

    ## Constructor just needs to call base class constructor in this case
    #-------------------------------------------------------------------------------
    def __init__(self, *args):

        SourceHandler.__init__(self, *args)

    #--------------------------------------------------------------------------------
    ## For Backward euler, there is no streaming term since implicit streaming on LHS
    #  of equation is already included in the system
    def evalImplicit(self, el, **kwargs):

        return [0.0 for i in xrange(4)]

    #--------------------------------------------------------------------------------
    ## At time \f$t_n\f$, there is a streaming source based on upwinding in
    #  spatial derivative
    def evalOld(self, i, psi_minus_old=None, psi_plus_old=None, **kwargs):

        Q = np.zeros(4)

        #Get all the indices for this cell
        iLminus     = getLocalIndex("L","-") # dof i,  L,-
        iLplus      = getLocalIndex("L","+") # dof i,  L,+
        iRminus     = getLocalIndex("R","-") # dof i,  R,-
        iRplus      = getLocalIndex("R","+") # dof i,  R,l+

        #Get the edge fluxes
        #for positive mu, upwinding on left term. For neg mu, upwinding on right
        if i == 0: #left boundary special case
            psi_L_face = kwargs['bc_flux_left']
        else:
            psi_L_face = psi_plus_old[i-1][1] #psi_{i-1,r}

        if i == self.mesh.n_elems - 1: #right boundary special case
            psi_R_face = kwargs['bc_flux_right']
        else:
            psi_R_face  = psi_minus_old[i+1][0] #psi_{i+1,l}

        psi_L_m = psi_minus_old[i][0] #m for minus
        psi_L_p = psi_plus_old[i][0]  #p for plus
        psi_R_m = psi_minus_old[i][1]                
        psi_R_p = psi_plus_old[i][1]
        mu = {"-" : -1./sqrt(3), "+" : 1./sqrt(3.)}

        #compute cell averages
        psi_i_m = 0.5*(psi_L_m + psi_R_m)
        psi_i_p = 0.5*(psi_L_p + psi_R_p)

        #Evaluate \f$ \my d\psi/dx \f$ at t_n. The minus is because on RHS of eq.
        print psi_i_m, psi_L_m

        Q[iLminus] = -1.*mu["-"]*(psi_i_m - psi_L_m)
        Q[iLplus]  = -1.*mu["+"]*(psi_i_p - psi_L_face)
        Q[iRminus] = -1.*mu["-"]*(psi_R_face - psi_i_m)
        Q[iRplus]  = -1.*mu["+"]*(psi_R_p - psi_i_p)

        return Q

    #--------------------------------------------------------------------------------
    ## The older term is the same as old, but with the oldest fluxes. So call old
    #  version
    def evalOlder(self, el, psi_minus_older=None, psi_plus_older=None, **kwargs):

        return evalOld(self,el,psi_minus_old=psi_minus_older,
                psi_plus_old=psi_plus_older)


#====================================================================================
## Derived class for reaction term:  \sigma_t* psi
class ReactionSrc(SourceHandler):

    ## Constructor just needs to call base class constructor in this case
    #-------------------------------------------------------------------------------
    def __init__(self, *args):

        SourceHandler.__init__(self, *args)

    #--------------------------------------------------------------------------------
    ## For Backward euler, there is no reaction term since implicit reaction term on LHS
    #  of equation is already included in the system
    def evalImplicit(self, el, **kwargs):

        return [0.0 for i in xrange(4)]

    #--------------------------------------------------------------------------------
    ## At time \f$t_n\f$, there is a reaction source, straight forward
    def evalOld(self, i, psi_minus_old=None, psi_plus_old=None, 
            cx_old = None, **kwargs):

        #Get all the indices, this is basically redundant
        iLminus     = getLocalIndex("L","-") # dof i,  L,-
        iLplus      = getLocalIndex("L","+") # dof i,  L,+
        iRminus     = getLocalIndex("R","-") # dof i,  R,-
        iRplus      = getLocalIndex("R","+") # dof i,  R,l+

        #cross sections
        sig_t_l = cx_old[i][0].sig_t
        sig_t_r = cx_old[i][1].sig_t

        Q  = np.zeros(4)
        #negatives because on RHS of equation
        Q[iLminus] = -1.*psi_minus_old[i][0] * sig_t_l
        Q[iRminus] = -1.*psi_minus_old[i][1] * sig_t_r
        Q[iLplus] = -1.*psi_plus_old[i][0]   * sig_t_l
        Q[iRplus] = -1.*psi_plus_old[i][1]   * sig_t_r

        return Q

    #--------------------------------------------------------------------------------
    ## The older term is the same as old, but with the oldest fluxes. So call old
    #  version
    def evalOlder(self, el, psi_minus_older=None, psi_plus_older=None, 
                   cx_older = None, **kwargs):

        return evalOld(self,el,psi_minus_old=psi_minus_older,
                psi_plus_old=psi_plus_older,cx_old=cx_older,**kwargs)

#====================================================================================
## Derived class for scattering source term:  \sigma_s phi/2
class ScatteringSrc(SourceHandler):

    ## Constructor just needs to call base class constructor in this case
    #-------------------------------------------------------------------------------
    def __init__(self, *args):

        SourceHandler.__init__(self, *args)
        self.c = GC.SPD_OF_LGT

    #--------------------------------------------------------------------------------
    ## For Backward euler, there is no streaming term since implicit reaction term on LHS
    #  of equation is already included in the system
    def evalImplicit(self, el, **kwargs):

        return [0.0 for i in xrange(4)]

    #--------------------------------------------------------------------------------
    ## At time \f$t_n\f$, there is a isotropic scattering source
    def evalOld(self, i, E_old=None, cx_old = None, **kwargs):

        #Get scattering cross section
        sig_s_l = cx_old[i][0].sig_s
        sig_s_r = cx_old[i][1].sig_s

        #Get all the indices, this is basically redundant
        phi_L = E_old[i][0]*c
        phi_R = E_old[i][1]*c

        iLminus     = getLocalIndex("L","-") # dof i,  L,-
        iLplus      = getLocalIndex("L","+") # dof i,  L,+
        iRminus     = getLocalIndex("R","-") # dof i,  R,-
        iRplus      = getLocalIndex("R","+") # dof i,  R,l+

        Q  = np.zeros(4)
        Q[iLminus] = 0.5*phi_L*sig_s_l
        Q[iRminus] = 0.5*phi_R*sig_s_r
        Q[iLplus] = 0.5*phi_L*sig_s_r
        Q[iRplus] = 0.5*phi_R*sig_s_r

        return Q

    #--------------------------------------------------------------------------------
    ## The older term is the same as old, but with the oldest E. So call old
    #  version
    def evalOlder(self, el, E_older = None, cx_older = None, **kwargs):

        return evalOld(self,el,E_old = E_older,cx_old=cx_older,**kwargs)






