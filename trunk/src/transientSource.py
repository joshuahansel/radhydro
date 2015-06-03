## @package src.transientSource
#  This file contains all of the transient source builder functions 
#
# Each source is responsible for implementing several functions to build the
# source for each of the time stepping methods.  The derived classes will, in
# general, only need to define each of several ``virtual functions''.  Each derived
# class inherits a "computeTerm" function. This is the primary function, responsible
# for building the entire right hand side for that term in the equation. 
#
# NOTE: TRT related source builders are located in src/newtonStateHandler.py because
#
# In general, each derived class is responsible for overriding the different
# functions for the different time stepping methods. Each source term
# will be its own derived class.  Most classes will use the default computeTerm
# function, which calls several functions to help build the source on the RHS. In
# particular, they assume the following forms of each of the time stepping methods:
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
# responsible for writing functions that evaluate \f$A^n,\;A^{n-1},\f$ and
# \f$Y^{n+1}\f$, which are functions evalOld, evalOlder, and evalImplicit,
# respectively.
#
# In implementation, \f$Y^n/c\Delta t\f$
# is also on the right hand side of the equation as a source term. This is
# the only term for which its derived class overrides computeTerm
# rather than implement the eval*** functions, since the term is the same for all
# stepping algorithms


import re
import numpy as np
from math import sqrt
from mesh import Mesh
import globalConstants as GC
from utilityFunctions import getIndex, getLocalIndex
from radUtilities import mu

## Class to compute the full transient source
#
class TransientSource:
   
   ## Constructor
   #
   #  @param[in] mesh          mesh object
   #  @param[in] time_stepper  string identifier for the chosen time-stepper,
   #                           e.g., 'CN'
   #  @param[in] problem_type  type of transient problem being run. options are:
   #                           rad_only: transient no material coupling, just for
   #                                     tests. Default value.
   #                           trt     : radiation coupled to material internal
   #                                     energy. No material motion terms
   #                           rad_hydro: not implemented yet. may need something
   #                                     more complicated
   #  @param[in] src_term      if you want to have an external source term, you need
   #                           to pass in true, otherwise this is ignored
   #
   #  
   #
   def __init__(self, mesh, time_stepper, problem_type='rad_only',
           src_term=False, newton_handler=None):

      #keep track of the newton handler separately as it may need remembered
      #to ensure that same instance is used
      self.newton_handler = newton_handler
      
      self.mesh = mesh

      # create transient source terms. TODO Since the only state remembered in this class
      # is the time stepper and the mesh, this doesnt really need to be a class,
      # it can just be a function that is called with mesh and time stepper passed 
      # along to evaluate
      self.terms = [OldIntensityTerm(mesh, time_stepper), 
               StreamingTerm   (mesh, time_stepper),
               ReactionTerm    (mesh, time_stepper),
               ScatteringTerm  (mesh, time_stepper)]

      #Add extra terms if necessary
      if src_term:
          self.terms.append(SourceTerm(mesh, time_stepper))  
      
      #Check that newton handler was passed in if TRT active
      if problem_type in ['trt', 'rad_hydro']:

         if newton_handler == None:
             raise IOError("You must pass in a newton handler to TransientSource "\
                "constructor for a TRT problem\n")
         else:

             #make sure newton handler is instance of TransientSource
             if not isinstance(newton_handler,TransientSourceTerm):
                 raise NotImplementedError("Newton handler must inherit from TransientSourceTerm")
             else:
                 self.terms.append(self.newton_handler)
               

   ## Function to evaluate the full transient source
   #
   #  Creates a list of transient sources and loops
   #  over them to add to the full transient source.
   #
   def evaluate(self, **kwargs):

      #Debugging check to notify since change made
      if 'Q_new' in kwargs:
          if not any([isinstance(i,SourceTerm) for i in self.terms]):
              raise ValueError("You passed in a Q_new to source builder, but didnt "\
                  "specify source term in constructor, check RadiationTimeStepper "\
                  "constructor")


      # build the transient source
      n = self.mesh.n_elems * 4
      Q_tr = np.zeros(n)
      for term in self.terms:
          # build source for this handler
          Q_term = term.computeTerm(**kwargs)
          # Add elementwise the src to the total
          Q_tr += Q_term

      return Q_tr

#=================================================================================
## Base class for source handlers. More info given in package documentation. Here, 
#  the evaluate functions are only implemented to raise errors
#  in case developer incorrectly creates a derived class
class TransientSourceTerm:

    #---------------------------------------------------------------------------
    ## Constructor
    #
    #  @param[in] mesh          mesh object
    #  @param[in] time_stepper  string identifier for the chosen time-stepper,
    #                           e.g., 'CN'
    #
    def __init__(self, mesh, time_stepper):

        self.mesh = mesh
        self.time_stepper = time_stepper

        # Determine which time stepping function to use in derived class
        self.func = None
        if re.search("BE", time_stepper):
            self.func = self.evalBE
        elif re.search("CN", time_stepper):
            self.func = self.evalCN
        elif re.search("BDF2", time_stepper):
            self.func = self.evalBDF2
        else:
           raise NotImplementedError("Specified an invalid time-stepper")

    #----------------------------------------------------------------------------
    ## Function to evaluate source at all cells in the mesh. This is the main 
    #  function to be called on all sources.  
    #
    def computeTerm(self, **kwargs):

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

#===================================================================================
# All derived classes
#===================================================================================

## Derived class for computing old intensity term,
#  \f$\frac{\Psi^{\pm,n}}{c\Delta t}\f$
#
class OldIntensityTerm(TransientSourceTerm):
    
    ## Constructor
    def __init__(self, *args):

        # call base class constructor
        TransientSourceTerm.__init__(self, *args)
    
    ## Override the build source function, since old intensity term is the same
    #  for all time steppers
    #
    #  @param[in] dt        time step size
    #  @param[in] psi_old   old angular flux, stored as array with
    #                       global dof indexing
    #
    def computeTerm(self, dt=None, psi_old=None, **kwargs):

        # Loop over all cells and build source 
        Q = np.array([])
        for i in range(self.mesh.n_elems):

            # Evaluate source of element i
            Q_local = self.computeOldIntensityTerm(i, dt=dt, psi_old=psi_old)

            # Append source from element i
            Q = np.append(Q, Q_local)

        return Q

    ## Computes old intensity term, \f$\frac{\Psi^{\pm,n}}{c\Delta t}\f$
    #
    #  @param[in] i         element id
    #  @param[in] dt        time step size
    #  @param[in] psi_old   old angular flux, stored as array with
    #                       global dof indexing
    #
    def computeOldIntensityTerm(self, i, dt=None, psi_old=None):

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

        # compute c*dt
        c_dt = GC.SPD_OF_LGT * dt

        # compute old intensity term
        Q_local = np.zeros(4)
        Q_local[Lm] = psi_old[iLm] / c_dt
        Q_local[Lp] = psi_old[iLp] / c_dt
        Q_local[Rm] = psi_old[iRm] / c_dt
        Q_local[Rp] = psi_old[iRp] / c_dt

        return Q_local


#====================================================================================
## Derived class for computing streaming term,
#  \f$\mu^\pm \frac{\partial\Psi}{\partial x}\f$
#
class StreamingTerm(TransientSourceTerm):

    #-------------------------------------------------------------------------------
    ## Constructor
    def __init__(self, *args):

        TransientSourceTerm.__init__(self, *args)

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
    #  @param[in] psi_old   old angular flux, stored as array with
    #                       global dof indexing
    #
    def evalOld(self, i, psi_old=None, **kwargs):

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

        # psip_{i-1/2}
        if i == 0: # left boundary
            psip_Lface = kwargs['bc_flux_left']
        else:
            psip_Lface = psi_old[getIndex(i-1,"R","+")]

        # psim_{i+1/2}
        if i == self.mesh.n_elems - 1: # right boundary
            psim_Rface = kwargs['bc_flux_right']
        else:
            psim_Rface  = psi_old[getIndex(i+1,"L","-")]

        psim_L = psi_old[iLm] # psim_{i,L}
        psip_L = psi_old[iLp] # psip_{i,L}
        psim_R = psi_old[iRm] # psim_{i,R}
        psip_R = psi_old[iRp] # psip_{i,R}
        psim_Lface = psim_L   # psim_{i-1/2}
        psip_Rface = psip_R   # psip_{i+1/2}

        # compute cell center values
        psim_i = 0.5*(psim_L + psim_R)
        psip_i = 0.5*(psip_L + psip_R)

        # mesh size divided by 2
        h_over_2 = self.mesh.getElement(i).dx/2.0

        # compute streaming source
        Q_local = np.zeros(4)
        Q_local[Lm] = -1.*mu["-"]*(psim_i     - psim_Lface)/h_over_2
        Q_local[Lp] = -1.*mu["+"]*(psip_i     - psip_Lface)/h_over_2
        Q_local[Rm] = -1.*mu["-"]*(psim_Rface - psim_i)    /h_over_2
        Q_local[Rp] = -1.*mu["+"]*(psip_Rface - psip_i)    /h_over_2

        return Q_local

    #--------------------------------------------------------------------------------
    ## Computes older streaming term,
    #  \f$\mu^\pm\frac{\partial\Psi^{n-1}}{\partial x}\f$
    #
    #  @param[in] i           element id
    #  @param[in] psi_older   older angular flux, stored as array with
    #                         global dof indexing
    #
    def evalOlder(self, i, psi_older=None, **kwargs):

        # Use old function but with older arguments.
        # carefully pass in **kwargs to avoid duplicating
        return self.evalOld(i, psi_old=psi_older,
                  bc_flux_left=kwargs['bc_flux_left'],
                  bc_flux_right=kwargs['bc_flux_right'])


#====================================================================================
## Derived class for computing reaction term, \f$\sigma_t\Psi^\pm\f$
class ReactionTerm(TransientSourceTerm):

    #-------------------------------------------------------------------------------
    ## Constructor
    def __init__(self, *args):

        TransientSourceTerm.__init__(self, *args)

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
    #  @param[in] psi_old   old angular flux, stored as array with
    #                       global dof indexing
    #  @param[in] cx_old    old cross sections
    #
    def evalOld(self, i, psi_old=None, cx_old=None, **kwargs):

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

        # left and right cross sections
        sig_t_L = cx_old[i][0].sig_t
        sig_t_R = cx_old[i][1].sig_t

        # compute reaction source
        Q_local = np.zeros(4)
        Q_local[Lm] = -1.*psi_old[iLm] * sig_t_L
        Q_local[Rm] = -1.*psi_old[iRm] * sig_t_R
        Q_local[Lp] = -1.*psi_old[iLp] * sig_t_L
        Q_local[Rp] = -1.*psi_old[iRp] * sig_t_R

        return Q_local

    #--------------------------------------------------------------------------------
    ## Computes older reaction term, \f$\sigma_t^{n-1}\Psi^{\pm,n-1}\f$
    #
    #  @param[in] i           element id
    #  @param[in] psi_older   older angular flux, stored as array with
    #                         global dof indexing
    #  @param[in] cx_older    older cross sections
    #
    def evalOlder(self, i, psi_older=None, cx_older=None, **kwargs):

        # Use old function but with older arguments.
        # Note that you cannot pass in **kwargs or it will duplicate some arguments
        return self.evalOld(i, psi_old=psi_older, cx_old=cx_older)

#====================================================================================
## Derived class for computing scattering source term, \f$\frac{\sigma_s}{2}\phi\f$
class ScatteringTerm(TransientSourceTerm):

    #-------------------------------------------------------------------------------
    ## Constructor
    def __init__(self, *args):

        # call base class constructor
        TransientSourceTerm.__init__(self, *args)

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
    #  @param[in] i        element id
    #  @param[in] psi_old  old angular flux, stored as array with
    #                      global dof indexing
    #  @param[in] cx_old   old cross sections
    #
    def evalOld(self, i, psi_old=None, cx_old=None, **kwargs):

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

        # left and right scattering cross section
        sig_s_L = cx_old[i][0].sig_s
        sig_s_R = cx_old[i][1].sig_s

        # left and right scalar fluxes
        phi_L = psi_old[iLm] + psi_old[iLp]
        phi_R = psi_old[iRm] + psi_old[iRp]

        # compute scattering source
        Q_local = np.zeros(4)
        Q_local[Lm] = 0.5*phi_L*sig_s_L
        Q_local[Rm] = 0.5*phi_R*sig_s_R
        Q_local[Lp] = 0.5*phi_L*sig_s_L
        Q_local[Rp] = 0.5*phi_R*sig_s_R

        return Q_local

    #--------------------------------------------------------------------------------
    ## Computes older scattering source term,
    #  \f$\frac{\sigma_s^{n-1}}{2}\phi^{n-1}\f$
    #
    #  @param[in] i          element id
    #  @param[in] psi_older  older angular flux, stored as array with
    #                        global dof indexing
    #  @param[in] cx_older   older cross sections
    #
    def evalOlder(self, i, psi_older=None, cx_older=None, **kwargs):

        # Use old function but with older arguments
        return self.evalOld(i, psi_old=psi_older, cx_old=cx_older)

#====================================================================================
## Derived class for computing source term, \f$\mathcal{Q}\f$
class SourceTerm(TransientSourceTerm):

    #-------------------------------------------------------------------------------
    ## Constructor
    def __init__(self, *args):

        # call base class constructor
        TransientSourceTerm.__init__(self, *args)

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

#====================================================================================
## Derived class for computing drift term,
#  \f$-\frac{1}{2}\sigma_t\frac{u}{c}\mathcal{F}_0\f$
#
class DriftTerm(TransientSourceTerm):

    #-------------------------------------------------------------------------------
    ## Constructor
    #
    def __init__(self, *args):

        # call base class constructor
        TransientSourceTerm.__init__(self, *args)

    #--------------------------------------------------------------------------------
    ## Computes implicit drift term,
    #  \f$-\frac{1}{2}\sigma_t^k\frac{u^k}{c}\mathcal{F}_0^k\f$
    #
    #  @param[in] i           element id
    #  @param[in] cx_prev     previous cross sections \f$\sigma^k\f$
    #  @param[in] hydro_prev  previous hydro states \f$\mathbf{H}^k\f$
    #  @param[in] psi_prev    previous angular fluxes \f$\Psi^{\pm,k}\f$
    #
    def evalImplicit(self, i, cx_prev=None, hydro_prev=None,
           psi_prev=None, **kwargs):

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

        # get left and right total cross sections
        cxtL = cx_prev[i][0].sig_t
        cxtR = cx_prev[i][1].sig_t

        # compute left and right velocities
        uL = hydro_prev[i][0].u
        uR = hydro_prev[i][1].u

        # get left and right angular fluxes
        psimL = psi_prev[iLm]
        psipL = psi_prev[iLp]
        psimR = psi_prev[iRm]
        psipR = psi_prev[iRp]

        # compute left and right radiation energies and fluxes
        c = GC.SPD_OF_LGT
        EL = (psipL + psimL) / c
        ER = (psipR + psimR) / c
        FL = (psipL - psimL) / sqrt(3.0)
        FR = (psipR - psimR) / sqrt(3.0)
        F0L = FL - 4.0/3.0*EL*uL
        F0R = FR - 4.0/3.0*ER*uR
        
        # return local Q values
        Q_local = np.zeros(4)
        Q_local[Lm] = -cxtL*uL/c*F0L
        Q_local[Lp] = -cxtL*uL/c*F0L
        Q_local[Rm] = -cxtR*uR/c*F0R
        Q_local[Rp] = -cxtR*uR/c*F0R

        return Q_local

    #--------------------------------------------------------------------------------
    ## Computes old drift term,
    #  \f$-\frac{1}{2}\sigma_t^n\frac{u^n}{c}\mathcal{F}_0^n\f$
    #
    #  @param[in] i          element id
    #  @param[in] cx_old     old cross sections \f$\sigma^n\f$
    #  @param[in] hydro_old  old hydro states \f$\mathbf{H}^n\f$
    #  @param[in] psi_old    old angular fluxes \f$\Psi^{\pm,n}\f$
    #
    def evalOld(self, i, cx_old=None, hydro_old=None,
           psi_old=None, **kwargs):

       return self.evalImplicit(i, cx_prev=cx_old, hydro_prev=hydro_old,
           psi_prev=psi_old, **kwargs):


    #--------------------------------------------------------------------------------
    ## Computes older drift term,
    #  \f$-\frac{1}{2}\sigma_t^{n-1}\frac{u^{n-1}}{c}\mathcal{F}_0^{n-1}\f$
    #
    #  @param[in] i            element id
    #  @param[in] cx_older     older cross sections \f$\sigma^{n-1}\f$
    #  @param[in] hydro_older  older hydro states \f$\mathbf{H}^{n-1}\f$
    #  @param[in] psi_older    older angular fluxes \f$\Psi^{\pm,n-1}\f$
    #
    def evalOlder(self, i, cx_older=None, hydro_older=None,
           psi_older=None, **kwargs):

       return self.evalImplicit(i, cx_prev=cx_older, hydro_prev=hydro_older,
           psi_prev=psi_older, **kwargs):

#====================================================================================
## Derived class for computing anisotropic source term,
#  \f$2\mu^\pm\sigma_t\mathcal{E}u\f$
#
class AnisotropicSourceTerm(TransientSourceTerm):

    #-------------------------------------------------------------------------------
    ## Constructor
    #
    def __init__(self, *args):

        # call base class constructor
        TransientSourceTerm.__init__(self, *args)

    #--------------------------------------------------------------------------------
    ## Computes previous anisotropic source term,
    #  \f$2\mu^\pm\sigma_t^k\mathcal{E}^k u^k\f$
    #
    #  @param[in] i           element id
    #  @param[in] cx_prev     previous cross sections \f$\sigma^k\f$
    #  @param[in] hydro_prev  previous hydro states \f$\mathbf{H}^k\f$
    #  @param[in] psi_prev    previous angular fluxes \f$\Psi^{\pm,k}\f$
    #
    def evalImplicit(self, i, cx_prev=None, hydro_prev=None,
           psi_prev=None, **kwargs):

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

        # get left and right total cross sections
        cxtL = cx_prev[i][0].sig_t
        cxtR = cx_prev[i][1].sig_t

        # compute left and right velocities
        uL = hydro_prev[i][0].u
        uR = hydro_prev[i][1].u

        # get left and right angular fluxes
        psimL = psi_prev[iLm]
        psipL = psi_prev[iLp]
        psimR = psi_prev[iRm]
        psipR = psi_prev[iRp]

        # compute left and right radiation energies and fluxes
        c = GC.SPD_OF_LGT
        EL = (psipL + psimL) / c
        ER = (psipR + psimR) / c
        
        # return local Q values
        Q_local = np.zeros(4)
        Q_local[Lm] = 2.0*mu["-"]*cxtL*EL*uL
        Q_local[Lp] = 2.0*mu["+"]*cxtL*EL*uL
        Q_local[Rm] = 2.0*mu["-"]*cxtR*ER*uR
        Q_local[Rp] = 2.0*mu["+"]*cxtR*ER*uR

        return Q_local

    #--------------------------------------------------------------------------------
    ## Computes old anisotropic source term,
    #  \f$2\mu^\pm\sigma_t^n\mathcal{E}^n u^n\f$
    #
    #  @param[in] i           element id
    #  @param[in] cx_old     old cross sections \f$\sigma^n\f$
    #  @param[in] hydro_old  old hydro states \f$\mathbf{H}^n\f$
    #  @param[in] psi_old    old angular fluxes \f$\Psi^{\pm,n}\f$
    #
    def evalOld(self, i, cx_old=None, hydro_old=None,
           psi_old=None, **kwargs):

       return self.evalImplicit(i, cx_prev=cx_old, hydro_prev=hydro_old,
           psi_prev=psi_old, **kwargs):

    #--------------------------------------------------------------------------------
    ## Computes older anisotropic source term,
    #  \f$2\mu^\pm\sigma_t^{n-1}\mathcal{E}^{n-1} u^{n-1}\f$
    #
    #  @param[in] i            element id
    #  @param[in] cx_older     older cross sections \f$\sigma^{n-1}\f$
    #  @param[in] hydro_older  older hydro states \f$\mathbf{H}^{n-1}\f$
    #  @param[in] psi_older    older angular fluxes \f$\Psi^{\pm,n-1}\f$
    #
    def evalOlder(self, i, cx_older=None, hydro_older=None,
           psi_older=None, **kwargs):

       return self.evalImplicit(i, cx_prev=cx_older, hydro_prev=hydro_older,
           psi_prev=psi_older, **kwargs):

