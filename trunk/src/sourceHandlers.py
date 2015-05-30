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
from radUtilities import mu

#====================================================================================
## Base class for source handlers. More info given in package documentation. Here, 
#  the evaluate functions are only implemented to raise errors
#  in case developer incorrectly creates a derived class
class SourceHandler:

    #---------------------------------------------------------------------------
    ## Constructor
    #
    #  @param[in] mesh          mesh object
    #  @param[in] dt            time step size
    #  @param[in] time_stepper  string identifier for the chosen time-stepper,
    #                           e.g., 'CN'
    #
    def __init__(self, mesh, dt, time_stepper):

        self.mesh = mesh
        self.dt   = dt
        self.time_stepper = time_stepper

        # Determine which time stepping function to use in derived class
        ts = time_stepper
        self.func = None
        if re.search("BE", ts):
            self.func = self.evalBE
        elif re.search("CN", ts):
            self.func = self.evalCN
        elif re.search("BDF2", ts):
            self.func = self.evalBDF2
        else:
           raise NotImplementedError("Specified an invalid time-stepper")

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
    #
    #  @param[in] i  element id
    #
    def evalCN(self, i, **kwargs):

        return 0.5*(self.evalOld(i, **kwargs) + self.evalImplicit(i, **kwargs))

    #----------------------------------------------------------------------------
    ## Function to evaluate source backward Euler time stepping, for element el
    #
    #  @param[in] i  element id
    #
    def evalBE(self, i, **kwargs):

        return self.evalImplicit(i, **kwargs)

    #----------------------------------------------------------------------------
    ## Function to evaluate source in BDF2 time stepping, for element el
    #
    #  @param[in] i  element id
    #
    def evalBDF2(self, i, **kwargs):

        return 1./6.*( self.evalOld(i, **kwargs) + self.evalOlder(i, **kwargs) ) \
                 + 2./3.*(self.evalImplicit(i, **kwargs))

    #----------------------------------------------------------------------------
    ## Function to evaluate the implicit term, if it occurs on right hand side of
    #  equation. For example, the streaming term has no implicit term on the RHS 
    #
    #  @param[in] i  element id
    #
    def evalImplicit(self, i, **kwargs):

        raise NotImplementedError("You must define the evalImplicit, or entire build"             
            "source function in the derived class")

    #----------------------------------------------------------------------------
    ## Function to evaluate the old term at \f$t_n\f$. For example, in the CN system,
    #  this is \f$A(Y^n)\f$.  All terms should have this function 
    #
    #  @param[in] i  element id
    #
    def evalOld(self, i, **kwargs):

        raise NotImplementedError("You must define the evalOld, or entire build"
            "source function in the derived class")

    #----------------------------------------------------------------------------
    ## Function to evaluate the oldest term, only used by BDF2, i.e.,  at \f$t_n-1\f$. 
    #  All terms should have this function implemented.
    #
    #  @param[in] i  element id
    #
    def evalOlder(self, i, **kwargs):

        raise NotImplementedError("You must define the evalOlder, or entire build"
           "source function in the derived class")

#===========================================================================================
# All derived classes
#===========================================================================================

## Derived class for computing old intensity term, \f$\frac{\Psi^{\pm,n}}{c\Delta t}\f$
class OldIntensitySrc(SourceHandler):
    
    ## Constructor
    def __init__(self, *args):

        # call base class constructor
        SourceHandler.__init__(self, *args)

        self.c_dt = GC.SPD_OF_LGT * self.dt
    
    #-------------------------------------------------------------------------------
    ## Override the build source function, since old intensity term is the same
    #  for all time steppers
    #
    #  @param[in] psim_old  Old angular flux in minus direction, \f$\Psi^{-,n}\f$
    #  @param[in] psip_old  Old angular flux in plus  direction, \f$\Psi^{+,n}\f$
    #
    def buildSource(self, psim_old=None, psip_old=None, **kwargs):

        # Loop over all cells and build source 
        Q = np.array([])
        for i in range(self.mesh.n_elems):

            # Evaluate source of element i
            Q_elem =  self.evaluate(i, psim_old=psim_old, psip_old=psip_old)

            # Append source from element i
            Q = np.append(Q, Q_elem)

        return Q

    #----------------------------------------------------------------------------
    ## Computes old intensity term, \f$\frac{\Psi^{\pm,n}}{c\Delta t}\f$
    #
    #  @param[in] i         element id
    #  @param[in] psim_old  Old angular flux in minus direction, \f$\Psi^{-,n}\f$
    #  @param[in] psip_old  Old angular flux in plus  direction, \f$\Psi^{+,n}\f$
    #
    def evaluate(self, i, psim_old=None, psip_old=None):
        
        # get local indices
        iLm = getLocalIndex("L","-") # dof L,-
        iLp = getLocalIndex("L","+") # dof L,+
        iRm = getLocalIndex("R","-") # dof R,-
        iRp = getLocalIndex("R","+") # dof R,+

        # compute old intensity term
        Q  = np.zeros(4)
        Q[iLm] = psim_old[i][0] / self.c_dt
        Q[iRm] = psim_old[i][1] / self.c_dt
        Q[iLp] = psip_old[i][0] / self.c_dt
        Q[iRp] = psip_old[i][1] / self.c_dt

        return Q


#====================================================================================
## Derived class for computing streaming term,
#  \f$\mu^\pm \frac{\partial\Psi}{\partial x}\f$
class StreamingSrc(SourceHandler):

    #-------------------------------------------------------------------------------
    ## Constructor
    def __init__(self, *args):

        SourceHandler.__init__(self, *args)

    #--------------------------------------------------------------------------------
    ## implicit term is on LHS, so return zeros
    #
    #  @param[in] i         element id
    #
    def evalImplicit(self, i, **kwargs):

        return np.zeros(4)

    #--------------------------------------------------------------------------------
    ## Computes old streaming term, \f$\mu^\pm\frac{\partial\Psi^n}{\partial x}\f$
    #
    #  @param[in] i         element id
    #  @param[in] psim_old  Old angular flux in minus direction, \f$\Psi^{-,n}\f$
    #  @param[in] psip_old  Old angular flux in plus  direction, \f$\Psi^{+,n}\f$
    #
    def evalOld(self, i, psim_old=None, psip_old=None, **kwargs):

        # get local indices
        iLm = getLocalIndex("L","-") # dof L,-
        iLp = getLocalIndex("L","+") # dof L,+
        iRm = getLocalIndex("R","-") # dof R,-
        iRp = getLocalIndex("R","+") # dof R,+

        # psip_{i-1/2}
        if i == 0: # left boundary
            psip_Lface = kwargs['bc_flux_left']
        else:
            psip_Lface = psip_old[i-1][1]

        # psim_{i+1/2}
        if i == self.mesh.n_elems - 1: # right boundary
            psim_Rface = kwargs['bc_flux_right']
        else:
            psim_Rface  = psim_old[i+1][0]

        psim_L = psim_old[i][0] # psim_{i,L}
        psip_L = psip_old[i][0] # psip_{i,L}
        psim_R = psim_old[i][1] # psim_{i,R}
        psip_R = psip_old[i][1] # psip_{i,R}
        psim_Lface = psim_L     # psim_{i-1/2}
        psip_Rface = psip_R     # psip_{i+1/2}

        # compute cell center values
        psim_i = 0.5*(psim_L + psim_R)
        psip_i = 0.5*(psip_L + psip_R)

        # mesh size divided by 2
        h_over_2 = self.mesh.getElement(i).dx/2.0

        # compute streaming source
        Q = np.zeros(4)
        Q[iLm] = -1.*mu["-"]*(psim_i     - psim_Lface)/h_over_2
        Q[iLp] = -1.*mu["+"]*(psip_i     - psip_Lface)/h_over_2
        Q[iRm] = -1.*mu["-"]*(psim_Rface - psim_i)    /h_over_2
        Q[iRp] = -1.*mu["+"]*(psip_Rface - psip_i)    /h_over_2

        return Q

    #--------------------------------------------------------------------------------
    ## Computes older streaming term,
    #  \f$\mu^\pm\frac{\partial\Psi^{n-1}}{\partial x}\f$
    #
    #  @param[in] i           element id
    #  @param[in] psim_older  older angular flux in minus direction,
    #                         \f$\Psi^{-,n-1}\f$
    #  @param[in] psip_older  older angular flux in plus  direction,
    #                         \f$\Psi^{+,n-1}\f$
    #
    def evalOlder(self, i, psim_older=None, psip_older=None, **kwargs):

        # Use old function but with older arguments.
        # carefully pass in **kwargs to avoid duplicating
        return self.evalOld(i, psim_old=psim_older, psip_old=psip_older,
                  bc_flux_left=kwargs['bc_flux_left'],
                  bc_flux_right=kwargs['bc_flux_right'])


#====================================================================================
## Derived class for computing reaction term, \f$\sigma_t\Psi^\pm\f$
class ReactionSrc(SourceHandler):

    #-------------------------------------------------------------------------------
    ## Constructor
    def __init__(self, *args):

        SourceHandler.__init__(self, *args)

    #--------------------------------------------------------------------------------
    ## implicit term is on LHS, so return zeros
    #
    #  @param[in] i  element id
    #
    def evalImplicit(self, i, **kwargs):

        return np.zeros(4)

    #--------------------------------------------------------------------------------
    ## Computes old reaction term, \f$\sigma_t^n\Psi^{\pm,n}\f$
    #
    #  @param[in] i         element id
    #  @param[in] psim_old  old angular flux in minus direction, \f$\Psi^{-,n}\f$
    #  @param[in] psip_old  old angular flux in plus  direction, \f$\Psi^{+,n}\f$
    #  @param[in] cx_old    old cross sections
    #
    def evalOld(self, i, psim_old=None, psip_old=None, cx_old=None, **kwargs):

        # get local indices
        iLm = getLocalIndex("L","-") # dof L,-
        iLp = getLocalIndex("L","+") # dof L,+
        iRm = getLocalIndex("R","-") # dof R,-
        iRp = getLocalIndex("R","+") # dof R,+

        # left and right cross sections
        sig_t_L = cx_old[i][0].sig_t
        sig_t_R = cx_old[i][1].sig_t

        # compute reaction source
        Q  = np.zeros(4)
        Q[iLm] = -1.*psim_old[i][0] * sig_t_L
        Q[iRm] = -1.*psim_old[i][1] * sig_t_R
        Q[iLp] = -1.*psip_old[i][0] * sig_t_L
        Q[iRp] = -1.*psip_old[i][1] * sig_t_R

        return Q

    #--------------------------------------------------------------------------------
    ## Computes older reaction term, \f$\sigma_t^{n-1}\Psi^{\pm,n-1}\f$
    #
    #  @param[in] i           element id
    #  @param[in] psim_older  older angular flux in minus direction,
    #                        \f$\Psi^{-,n-1}\f$
    #  @param[in] psip_older  older angular flux in plus  direction,
    #                        \f$\Psi^{+,n-1}\f$
    #
    #  @param[in] cx_older    older cross sections
    #
    def evalOlder(self, i, psim_older=None, psip_older=None,
                  cx_older=None, **kwargs):

        # Use old function but with older arguments.
        # Note that you cannot pass in **kwargs or it will duplicate some arguments
        return self.evalOld(i, psim_old=psim_older, psip_old=psip_older,
                            cx_old=cx_older)

#====================================================================================
## Derived class for computing scattering source term, \f$\frac{\sigma_s}{2}\phi\f$
class ScatteringSrc(SourceHandler):

    #-------------------------------------------------------------------------------
    ## Constructor
    def __init__(self, *args):

        # call base class constructor
        SourceHandler.__init__(self, *args)

        # save speed of light
        self.c = GC.SPD_OF_LGT

    #--------------------------------------------------------------------------------
    ## implicit term is on LHS, so return zeros
    #
    #  @param[in] i  element id
    #
    def evalImplicit(self, i, **kwargs):

        return np.zeros(4)

    #--------------------------------------------------------------------------------
    ## Computes old scattering source term, \f$\frac{\sigma_s^n}{2}\phi^n\f$
    #
    #  @param[in] i       element id
    #  @param[in] E_old   old radiation energy, \f$\mathcal{E}^n\f$
    #  @param[in] cx_old  old cross sections
    #
    def evalOld(self, i, E_old=None, cx_old=None, **kwargs):

        # left and right scattering cross section
        sig_s_L = cx_old[i][0].sig_s
        sig_s_R = cx_old[i][1].sig_s

        # left and right scalar fluxes
        phi_L = E_old[i][0]*self.c
        phi_R = E_old[i][1]*self.c

        # get local indices
        iLm = getLocalIndex("L","-") # dof L,-
        iLp = getLocalIndex("L","+") # dof L,+
        iRm = getLocalIndex("R","-") # dof R,-
        iRp = getLocalIndex("R","+") # dof R,+

        # compute scattering source
        Q = np.zeros(4)
        Q[iLm] = 0.5*phi_L*sig_s_L
        Q[iRm] = 0.5*phi_R*sig_s_R
        Q[iLp] = 0.5*phi_L*sig_s_L
        Q[iRp] = 0.5*phi_R*sig_s_R

        return Q

    #--------------------------------------------------------------------------------
    ## Computes older scattering source term,
    #  \f$\frac{\sigma_s^{n-1}}{2}\phi^{n-1}\f$
    #
    #  @param[in] i         element id
    #  @param[in] E_older   older radiation energy, \f$\mathcal{E}^{n-1}\f$
    #  @param[in] cx_older  older cross sections
    #
    def evalOlder(self, i, E_older=None, cx_older=None, **kwargs):

        # Use old function but with older arguments
        return self.evalOld(i, E_old=E_older, cx_old=cx_older)

#====================================================================================
## Derived class for computing source term, \f$\mathcal{Q}\f$
class SourceSrc(SourceHandler):

    #-------------------------------------------------------------------------------
    ## Constructor
    def __init__(self, *args):

        # call base class constructor
        SourceHandler.__init__(self, *args)

    #--------------------------------------------------------------------------------
    ## Computes implicit source term, \f$\mathcal{Q}^{\pm,k}\f$
    #
    #  @param[in] i      element id
    #  @param[in] Q_new  implicit source term, \f$\mathcal{Q}^{\pm,k}\f$,
    #                    provided if source is not solution-dependent
    #
    def evalImplicit(self, i, Q_new=None, **kwargs):

        # Use old function but with new arguments
        return self.evalOld(i, Q_old=Q_new)

    #--------------------------------------------------------------------------------
    ## Computes old source term, \f$\mathcal{Q}^{\pm,n}\f$
    #
    #  @param[in] i      element id
    #  @param[in] Q_old  old source term, \f$\mathcal{Q}^{\pm,n}\f$, provided if
    #                    source is not solution-dependent
    #
    def evalOld(self, i, Q_old=None, **kwargs):

        # for now, sources cannot be solution-dependent, so raise an error if
        # no source is provided
        if Q_old is None:
           raise NotImplementedError("Solution-dependent sources not yet implemented")
        else:
           # get global indices
           iLm = getIndex(i,"L","-") # dof i,L,-
           iLp = getIndex(i,"L","+") # dof i,L,+
           iRm = getIndex(i,"R","-") # dof i,R,-
           iRp = getIndex(i,"R","+") # dof i,R,+

           # get local indices
           Lm = getLocalIndex("L","-") # dof L,-
           Lp = getLocalIndex("L","+") # dof L,+
           Rm = getLocalIndex("R","-") # dof R,-
           Rp = getLocalIndex("R","+") # dof R,+

           # return local Q values
           Q_local = np.zeros(4)
           Q_local[Lm] = Q_old[iLm]
           Q_local[Lp] = Q_old[iLp]
           Q_local[Rm] = Q_old[iRm]
           Q_local[Rp] = Q_old[iRp]

        return Q_local

    #--------------------------------------------------------------------------------
    ## Computes older source term, \f$\mathcal{Q}^{\pm,n-1}\f$
    #
    #  @param[in] i        element id
    #  @param[in] Q_older  older source term, \f$\mathcal{Q}^{\pm,n-1}\f$,
    #                      provided if source is not solution-dependent
    #
    def evalOlder(self, i, Q_older=None, **kwargs):

        # Use old function but with older arguments
        return self.evalOld(i, Q_old=Q_older)

