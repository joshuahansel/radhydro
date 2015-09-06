## @package src.transientSource
#  Contains classes and functions used to evaluate the transient radiation source
#
# Each source is responsible for implementing several functions to build the
# source for each of the time stepping methods.  The derived classes will, in
# general, only need to define each of several ``virtual functions''.  Each derived
# class inherits a "computeTerm" function. This is the primary function, responsible
# for building the entire right hand side for that term in the equation. 
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
#
# for the general problem 
# \f[
#   \frac{\partial Y}{c\partial t} = A[ Y(t)]
# \f]
# where \f$A\f$ is a operator that is a function of \f$Y,t\f$. NOTE: we keep the
# \f$1/c\Delta t\f$ term on the left hand terms. Each derived class is thus
# responsible for writing functions that evaluate \f$A^n,\;A^{n-1},\f$ and
# \f$A^{n+1}\f$, which are functions evalOld, evalOlder, and evalImplicit,
# respectively.
#
# In implementation, \f$Y^n/c\Delta t\f$
# is also on the right hand side of the equation as a source term. This is
# the only term for which its derived class overrides computeTerm
# rather than implement the eval*** functions, since the term is the same for all
# stepping algorithms
#
# TODO: There is a lot of extra passing of argumetns around to functions here.
# Although it would be less clean looking, moving the loops over all elements
# into the evaluate terms can probably speed up the code overall quite a bit.
# I would recommend looking at the profiler unittest to see where the time
# is being spent
#

import re
import numpy as np
from math import sqrt
from mesh import Mesh
import globalConstants as GC
from radUtilities import mu
from utilityFunctions import getIndex, getLocalIndex, computeRadiationVector
from utilityFunctions import getNu, computeEdgeVelocities, computeEdgeTemperatures,\
   computeEdgeDensities 

## Computes the radiation transient source
#
#  Loops over the list of transient source terms and adds them
#  to the full transient source.
#  The problem_type parameter may take one of the following values:
#
#  * 'rad_only':  radiation with no material coupling. Default value.
#     a           The S-2 source in this case is provided:
#  \f[
#    \mathcal{Q}^\pm = \mathcal{Q}^\pm(x,t)
#  \f]
#  * 'rad_mat':   radiation coupled to material internal
#                 energy with no material motion.
#                 The S-2 source in this case is
#  \f[
#    \mathcal{Q}^\pm = \frac{1}{2}\sigma_a a c T^4
#  \f]
#  * 'rad_hydro': radiation coupled to hydrodynamics.
#
#  @param[in] mesh          mesh object
#  @param[in] time_stepper  string identifier for the chosen time-stepper,
#                           e.g., 'CN'
#  @param[in] problem_type  type of transient problem being run. options are
#                           'rad_only', 'rad_mat', or 'rad_hydro'.
#                           Descriptions are above.
#
def computeRadiationSource(mesh, time_stepper, problem_type,
   **kwargs):

   # create list of transient source terms
   terms = [OldIntensityTerm(mesh, time_stepper), 
            StreamingTerm   (mesh, time_stepper),
            ReactionTerm    (mesh, time_stepper),
            ScatteringTerm  (mesh, time_stepper),
            SourceTerm      (mesh, time_stepper)]

   # create list of source terms according to problem type
   if problem_type == 'rad_only':

       # no terms need to be added
       pass

   elif problem_type == 'rad_mat':

       terms.append(PlanckianTerm(mesh, time_stepper))

   elif problem_type == 'rad_hydro':

       terms.extend([PlanckianTerm(mesh, time_stepper),
                     DriftTerm(mesh, time_stepper),
                     AnisotropicTerm(mesh, time_stepper)])

   else:

       raise NotImplementedError('Invalid problem_type specified')

   # compute the transient source
   n = mesh.n_elems * 4
   Q_tr = np.zeros(n)

   for term in terms:
       # compute source contribution for this term
       Q_term = term.computeTerm(**kwargs)
       # add to total
       Q_tr += Q_term

   return Q_tr

#=================================================================================
## Base class for source handlers. More info given in package documentation. Here, 
#  the evaluate functions are only implemented to raise errors
#  in case developer incorrectly creates a derived class
#
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

        for i in xrange(self.mesh.n_elems):
            
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
    #  @param[in] rad_old   old radiation
    #
    def computeTerm(self, dt, rad_old, **kwargs):

        # Loop over all cells and build source 
        Q = np.array([])
        for i in xrange(self.mesh.n_elems):

            # Evaluate source of element i
            Q_local = self.computeOldIntensityTerm(i, dt=dt, rad_old=rad_old)

            # Append source from element i
            Q = np.append(Q, Q_local)

        return Q

    ## Computes old intensity term, \f$\frac{\Psi^{\pm,n}}{c\Delta t}\f$
    #
    #  @param[in] i         element id
    #  @param[in] dt        time step size
    #  @param[in] rad_old   old radiation
    #
    def computeOldIntensityTerm(self, i, dt, rad_old):

        # get local indices
        Lm = getLocalIndex("L","-") # dof L,-
        Lp = getLocalIndex("L","+") # dof L,+
        Rm = getLocalIndex("R","-") # dof R,-
        Rp = getLocalIndex("R","+") # dof R,+

        # compute c*dt
        c_dt = GC.SPD_OF_LGT * dt

        # compute old intensity term
        Q_local = np.zeros(4)
        Q_local[Lm] = rad_old.psim[i][0] / c_dt
        Q_local[Lp] = rad_old.psip[i][0] / c_dt
        Q_local[Rm] = rad_old.psim[i][1] / c_dt
        Q_local[Rp] = rad_old.psip[i][1] / c_dt

        return Q_local


#====================================================================================
## Derived class for computing streaming term,
#  \f$\mu^\pm \frac{\partial\Psi^\pm}{\partial x}\f$
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
    ## Compute RHS term. This function has to be separate function just because the boundary
    #  fluxes may be variable in time and evalOld and evalOlder only have one BC
    #  object which knows old values
    #
    def evalRHSTerm(self, i, rad_old, rad_BC, time_step, **kwargs):

        #If it is a time dependent boundary condition, make a copy and
        #update to the old time
        if time_step == "old":
           psi_left, psi_right = rad_BC.getOldIncidentFluxes()
        elif time_step == "older":
           psi_left, psi_right = rad_BC.getOlderIncidentFluxes()
        else:
           raise IOError("Invalid call of evalRHSTerm BC time")

        # get local indices
        Lm = getLocalIndex("L","-") # dof L,-
        Lp = getLocalIndex("L","+") # dof L,+
        Rm = getLocalIndex("R","-") # dof R,-
        Rp = getLocalIndex("R","+") # dof R,+

        # psip_{i-1/2}
        if i == 0: # left boundary
            if rad_BC.bc_type == "periodic":
               psip_Lface = rad_old.psip[self.mesh.n_elems-1][1] # psip_{N-1,R}
            else:
               psip_Lface = psi_left
        else:
            psip_Lface = rad_old.psip[i-1][1]

        # psim_{i+1/2}
        if i == self.mesh.n_elems - 1: # right boundary
            if rad_BC.bc_type == "periodic":
               psim_Rface = rad_old.psim[0][0] #psim_{0,L}
            else:
               psim_Rface = psi_right
        else:
            psim_Rface  = rad_old.psim[i+1][0]

        psim_L = rad_old.psim[i][0] # psim_{i,L}
        psip_L = rad_old.psip[i][0] # psip_{i,L}
        psim_R = rad_old.psim[i][1] # psim_{i,R}
        psip_R = rad_old.psip[i][1] # psip_{i,R}
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
    ## Computes old streaming term,
    #  \f$\mu^\pm\frac{\partial\Psi^{\pm,n}}{\partial x}\f$
    #
    #  @param[in] i         element id
    #  @param[in] rad_old   old radiation
    #
    def evalOld(self, i, rad_old, rad_BC, t_old=None, **kwargs):

        return self.evalRHSTerm(i,
                  rad_old   = rad_old,
                  rad_BC    = rad_BC,
                  time_step = "old")


    #--------------------------------------------------------------------------------
    ## Computes older streaming term,
    #  \f$\mu^\pm\frac{\partial\Psi^{\pm,n-1}}{\partial x}\f$
    #
    #  @param[in] i           element id
    #  @param[in] rad_older   older radiation
    #
    def evalOlder(self, i, rad_older, rad_BC, **kwargs):

        # Use old function but with older arguments.
        # carefully pass in **kwargs to avoid duplicating
        return self.evalRHSTerm(i,
                  rad_old   = rad_older,
                  rad_BC    = rad_BC,
                  time_step = "older")



#====================================================================================
## Derived class for computing reaction term, \f$\sigma_t\Psi^\pm\f$
#
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
    #  @param[in] rad_old   old radiation
    #  @param[in] cx_old    old cross sections
    #
    def evalOld(self, i, rad_old, cx_old, **kwargs):

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
        Q_local[Lm] = -1.*rad_old.psim[i][0] * sig_t_L
        Q_local[Lp] = -1.*rad_old.psip[i][0] * sig_t_L
        Q_local[Rm] = -1.*rad_old.psim[i][1] * sig_t_R
        Q_local[Rp] = -1.*rad_old.psip[i][1] * sig_t_R

        return Q_local

    #--------------------------------------------------------------------------------
    ## Computes older reaction term, \f$\sigma_t^{n-1}\Psi^{\pm,n-1}\f$
    #
    #  @param[in] i           element id
    #  @param[in] rad_older   older radiation
    #  @param[in] cx_older    older cross sections
    #
    def evalOlder(self, i, rad_older, cx_older, **kwargs):

        # Use old function but with older arguments.
        return self.evalOld(i, rad_old=rad_older, cx_old=cx_older)

#====================================================================================
## Derived class for computing scattering source term, \f$\frac{\sigma_s}{2}\phi\f$
#
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
    #  @param[in] rad_old  old radiation
    #  @param[in] cx_old   old cross sections
    #
    def evalOld(self, i, rad_old, cx_old, **kwargs):

        # get local indices
        Lm = getLocalIndex("L","-") # dof L,-
        Lp = getLocalIndex("L","+") # dof L,+
        Rm = getLocalIndex("R","-") # dof R,-
        Rp = getLocalIndex("R","+") # dof R,+

        # left and right scattering cross section
        sig_s_L = cx_old[i][0].sig_s
        sig_s_R = cx_old[i][1].sig_s

        # left and right scalar fluxes
        phi_L = rad_old.phi[i][0]
        phi_R = rad_old.phi[i][1]

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
    #  @param[in] rad_older  older radiation
    #  @param[in] cx_older   older cross sections
    #
    def evalOlder(self, i, rad_older, cx_older, **kwargs):

        # Use old function but with older arguments
        return self.evalOld(i, rad_old=rad_older, cx_old=cx_older)

#====================================================================================
## Derived class for computing source term, \f$\mathcal{Q}\f$
#
class SourceTerm(TransientSourceTerm):

    #-------------------------------------------------------------------------------
    ## Constructor
    def __init__(self, *args):

        # call base class constructor
        TransientSourceTerm.__init__(self, *args)

    #--------------------------------------------------------------------------------
    ## Computes implicit source term, \f$\mathcal{Q}^{\pm,k}\f$
    #
    #  @param[in] i         element id
    #  @param[in] Qpsi_new  implicit source term, \f$\mathcal{Q}^{\pm,k}\f$,
    #                       provided if source is not solution-dependent
    #
    def evalImplicit(self, i, Qpsi_new, **kwargs):

        # Use old function but with new arguments
        return self.evalOld(i, Qpsi_old=Qpsi_new)

    #--------------------------------------------------------------------------------
    ## Computes old source term, \f$\mathcal{Q}^{\pm,n}\f$
    #
    #  @param[in] i         element id
    #  @param[in] Qpsi_old  old source term, \f$\mathcal{Q}^{\pm,n}\f$, provided if
    #                       source is not solution-dependent
    #
    def evalOld(self, i, Qpsi_old, **kwargs):

        # for now, sources cannot be solution-dependent, so raise an error if
        # no source is provided
        if Qpsi_old is None:
           raise NotImplementedError("Solution-dependent sources not yet implemented")

        return Qpsi_old[4*i:4*i+4]

    #--------------------------------------------------------------------------------
    ## Computes older source term, \f$\mathcal{Q}^{\pm,n-1}\f$
    #
    #  @param[in] i           element id
    #  @param[in] Qpsi_older  older source term, \f$\mathcal{Q}^{\pm,n-1}\f$,
    #                         provided if source is not solution-dependent
    #
    def evalOlder(self, i, Qpsi_older, **kwargs):

        # Use old function but with older arguments
        return self.evalOld(i, Qpsi_old=Qpsi_older)


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
    #  @param[in] rad_prev    previous radiation
    #
    def evalImplicit(self, i, cx_prev, hydro_prev, rad_prev, slopes_old, **kwargs):

        # get local indices
        Lm = getLocalIndex("L","-") # dof L,-
        Lp = getLocalIndex("L","+") # dof L,+
        Rm = getLocalIndex("R","-") # dof R,-
        Rp = getLocalIndex("R","+") # dof R,+

        # get left and right total cross sections
        cxtL = cx_prev[i][0].sig_t
        cxtR = cx_prev[i][1].sig_t

        # compute left and right velocities
        u = computeEdgeVelocities(i, hydro_prev[i], slopes_old)

        # compute left and right radiation energies and fluxes
        c = GC.SPD_OF_LGT
        EL = rad_prev.E[i][0]
        ER = rad_prev.E[i][1]
        FL = rad_prev.F[i][0]
        FR = rad_prev.F[i][1]
        F0L = FL - 4.0/3.0*EL*u[0]
        F0R = FR - 4.0/3.0*ER*u[1]
        
        # return local Q values
        Q_local = np.zeros(4)
        Q_local[Lm] = -cxtL*u[0]/c*F0L*0.5 #0.5 because it is isotropic
        Q_local[Lp] = -cxtL*u[0]/c*F0L*0.5
        Q_local[Rm] = -cxtR*u[1]/c*F0R*0.5
        Q_local[Rp] = -cxtR*u[1]/c*F0R*0.5

        return Q_local

    #--------------------------------------------------------------------------------
    ## Computes old drift term,
    #  \f$-\frac{1}{2}\sigma_t^n\frac{u^n}{c}\mathcal{F}_0^n\f$
    #
    #  @param[in] i          element id
    #  @param[in] cx_old     old cross sections \f$\sigma^n\f$
    #  @param[in] hydro_old  old hydro states \f$\mathbf{H}^n\f$
    #  @param[in] rad_old    old radiation
    #
    def evalOld(self, i, cx_old, hydro_old, rad_old, slopes_old, **kwargs):

       return self.evalImplicit(i, cx_prev=cx_old, hydro_prev=hydro_old,
           rad_prev=rad_old, slopes_old=slopes_old)

    #--------------------------------------------------------------------------------
    ## Computes older drift term,
    #  \f$-\frac{1}{2}\sigma_t^{n-1}\frac{u^{n-1}}{c}\mathcal{F}_0^{n-1}\f$
    #
    #  @param[in] i            element id
    #  @param[in] cx_older     older cross sections \f$\sigma^{n-1}\f$
    #  @param[in] hydro_older  older hydro states \f$\mathbf{H}^{n-1}\f$
    #  @param[in] rad_older    older radiation
    #
    def evalOlder(self, i, cx_older, hydro_older, rad_older, slopes_older, **kwargs):

       return self.evalImplicit(i, cx_prev=cx_older, hydro_prev=hydro_older,
           rad_prev=rad_older, slopes_old=slopes_older)

#====================================================================================
## Derived class for computing anisotropic source term,
#  \f$2\mu^\pm\sigma_t\mathcal{E}u\f$
#
class AnisotropicTerm(TransientSourceTerm):

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
    #  @param[in] rad_prev    previous radiation
    #
    def evalImplicit(self, i, cx_prev, hydro_prev, rad_prev, slopes_old, **kwargs):

        # get local indices
        Lm = getLocalIndex("L","-") # dof L,-
        Lp = getLocalIndex("L","+") # dof L,+
        Rm = getLocalIndex("R","-") # dof R,-
        Rp = getLocalIndex("R","+") # dof R,+

        # get left and right total cross sections
        cxtL = cx_prev[i][0].sig_t
        cxtR = cx_prev[i][1].sig_t

        # compute left and right velocities
        u = computeEdgeVelocities(i, hydro_prev[i], slopes_old)

        # compute left and right radiation energies and fluxes
        EL = rad_prev.E[i][0]
        ER = rad_prev.E[i][1]
        
        # return local Q values
        Q_local = np.zeros(4)
        Q_local[Lm] = 2.0*mu["-"]*cxtL*EL*u[0]
        Q_local[Lp] = 2.0*mu["+"]*cxtL*EL*u[0]
        Q_local[Rm] = 2.0*mu["-"]*cxtR*ER*u[1]
        Q_local[Rp] = 2.0*mu["+"]*cxtR*ER*u[1]

        return Q_local

    #--------------------------------------------------------------------------------
    ## Computes old anisotropic source term,
    #  \f$2\mu^\pm\sigma_t^n\mathcal{E}^n u^n\f$
    #
    #  @param[in] i          element id
    #  @param[in] cx_old     old cross sections \f$\sigma^n\f$
    #  @param[in] hydro_old  old hydro states \f$\mathbf{H}^n\f$
    #  @param[in] rad_old    old radiation
    #
    def evalOld(self, i, cx_old, hydro_old, rad_old, slopes_old, **kwargs):

       return self.evalImplicit(i, cx_prev=cx_old, hydro_prev=hydro_old,
           rad_prev=rad_old, slopes_old=slopes_old)

    #--------------------------------------------------------------------------------
    ## Computes older anisotropic source term,
    #  \f$2\mu^\pm\sigma_t^{n-1}\mathcal{E}^{n-1} u^{n-1}\f$
    #
    #  @param[in] i            element id
    #  @param[in] cx_older     older cross sections \f$\sigma^{n-1}\f$
    #  @param[in] hydro_older  older hydro states \f$\mathbf{H}^{n-1}\f$
    #  @param[in] rad_older    older radiation
    #
    def evalOlder(self, i, cx_older, hydro_older, rad_older, slopes_older, **kwargs):

       return self.evalImplicit(i, cx_prev=cx_older, hydro_prev=hydro_older,
           rad_prev=rad_older, slopes_old=slopes_older)

#====================================================================================
## Derived class for computing Planckian emission source term,
#  \f$\frac{1}{2}\sigma_a a c T^4\f$
#
#  The evalImplicit function requires a passed in argument planckian_new. Note that we
#  cannot simply use T_new^4 because it will not be consistent with the linearization,
#  which will result in inaccurate energy conservation.
#
class PlanckianTerm(TransientSourceTerm):

    #-------------------------------------------------------------------------------
    ## Constructor
    #
    def __init__(self, *args):

        # call base class constructor
        TransientSourceTerm.__init__(self, *args)

    #--------------------------------------------------------------------------------
    ## Computes the RHS portion of the implicit linearized Planckian term,
    #  \f$\frac{1}{2}\sigma_a^k a c (T^{k+1})^4 - \frac{1}{2}\sigma_a^k c \nu^k
    #    \mathcal{E}^{k+1}\f$.
    #
    #  The term \f$\frac{1}{2}\sigma_a^k c \nu^k \mathcal{E}^{k+1}\f$
    #  is not evaluated here but is instead evaluated as part of the scattering
    #  term \f$\sigma_s^k\phi^{k+1}\f$ by making the substitution
    #  \[
    #     \sigma_s^k\phi^{k+1} \rightarrow \tilde{\sigma_s^k}\phi^{k+1},
    #  \]
    #  where \f$\tilde{\sigma_s^k} \equiv \sigma_s^k + \nu^k\sigma_a^k\f$.
    #   
    #  @param[in] i             element id
    #  @param[in] cx_prev       previous cross sections \f$\sigma^k\f$
    #
    def evalImplicit(self, i, dt, cx_prev=None, hydro_prev=None, hydro_star=None,
            E_slopes_star=None, QE=None, slopes_old=None, e_rad_prev=None, hydro_new=None, **kwargs):

        # get coefficient corresponding to time-stepper
        scales = {"CN":0.5, "BE":1., "BDF2":2./3.}
        scale = scales[self.time_stepper]
      
        # get constants
        a = GC.RAD_CONSTANT
        c = GC.SPD_OF_LGT

        # get previous hydro state and specific heat
        state_prev = hydro_prev[i]
        state_star = hydro_star[i]
        spec_heat = state_prev.spec_heat

        #Compute edge velocities
        u_new = computeEdgeVelocities(i, hydro_new[i], slopes_old)

        #Compute left and right star energyes
        E_star = [state_star.E() - 0.5*E_slopes_star[i],
                  state_star.E() + 0.5*E_slopes_star[i]]

        # compute edge quantities, use newest density
        rho = computeEdgeDensities(i, hydro_new[i], slopes_old)
        T = computeEdgeTemperatures(spec_heat, e_rad_prev[i])

        # compute Planckian term for each edge on element
        planckian = [0.0,0.0]
        for edge in range(2):

            # get cross section
            sig_a = cx_prev[i][edge].sig_a

            # compute effective scattering fraction
            nu = getNu(T[edge], sig_a, rho[edge], spec_heat, dt, scale)
            QE_elem = QE[i][edge]

            # compute Planckian
            emission = (1.0 - nu)*sig_a*a*c*T[edge]**4
            planckian[edge] = emission \
                -   nu/(scale*dt)*( rho[edge]*(e_rad_prev[i][edge] +0.5*u_new[edge]**2)  \
                -   E_star[edge] ) \
                + nu*QE_elem/scale 

        # get local indices
        Lm = getLocalIndex("L","-") # dof L,-
        Lp = getLocalIndex("L","+") # dof L,+
        Rm = getLocalIndex("R","-") # dof R,-
        Rp = getLocalIndex("R","+") # dof R,+

        # return local Q values
        Q_local = np.zeros(4)
        Q_local[Lm] = 0.5*planckian[0]
        Q_local[Lp] = 0.5*planckian[0]
        Q_local[Rm] = 0.5*planckian[1]
        Q_local[Rp] = 0.5*planckian[1]

        return Q_local

    #--------------------------------------------------------------------------------
    ## Evaluate old Planckian.
    #  \f$\frac{1}{2}\sigma_a^n a c (T^n)^4\f$
    #
    #  @param[in] i          element id
    #  @param[in] cx_old     old cross sections \f$\sigma^n\f$
    #  @param[in] hydro_old  old hydro states \f$\mathbf{H}^n\f$
    #
    def evalOld(self, i, hydro_old, cx_old, e_rad_old=None, **kwargs):

        #use function forward to external function
        planckian = evalPlanckianOld(i, hydro_old, cx_old, e_rad_old)

        #Store the (isotropic) sources in correct index
        Q = np.zeros(4)
        Q[getLocalIndex("L","-")] = 0.5*planckian[0]
        Q[getLocalIndex("L","+")] = 0.5*planckian[0]
        Q[getLocalIndex("R","-")] = 0.5*planckian[1]
        Q[getLocalIndex("R","+")] = 0.5*planckian[1]

        return Q

    #--------------------------------------------------------------------------------
    ## Evaluate older term. Just call the evalOld function as in other source terms
    #
    def evalOlder(self, i, hydro_older, cx_older, e_rad_older, **kwargs):

        # Use old function but with older arguments.
        return self.evalOld(i, hydro_old=hydro_older, cx_old=cx_older,
           e_rad_old=e_rad_older)


#=====================================================================================
# Functions used by source term builders as well as newton state handler. Thus they
#  are external functions from which objects that need them use function forwarding
#  to access.


## Evaluates a Planckian term \f$\sigma_a^n a c (T^n)^4\f$.
#
#  @param[in] i          element id
#  @param[in] hydro_old  old hydro states \f$\mathbf{H}^n\f$
#  @param[in] cx_old     old cross sections \f$\sigma^n\f$
#  @param[in] e_rad_old  value of e at edge values from radiation solve
#
def evalPlanckianOld(i, hydro_old, cx_old, e_rad_old):

    # compute edge temperatures
    T = computeEdgeTemperatures(hydro_old[i].spec_heat, e_rad_old[i])

    #calculate at left and right, isotropic emission source
    planckian = [0.0,0.0]
    for edge in range(2):

        #Cross section
        sig_a = cx_old[i][edge].sig_a

        #Calculate planckian (with isotropic term included)
        planckian[edge] = sig_a*GC.RAD_CONSTANT*GC.SPD_OF_LGT*T[edge]**4.

    return tuple(planckian)


## Computes an extraneous source vector for radiation.
#
#  The input function handles are functions of (x,t), and the output source
#  is given in the ordering of radiation dofs.
#
#  @param[in] psim_src  function handle for the \f$\Psi^-\f$ extraneous source
#  @param[in] psip_src  function handle for the \f$\Psi^+\f$ extraneous source
#  @param[in] mesh      mesh
#  @param[in] t         time at which to evaluate the function
#
#  @return source vector, in the ordering of radiation dofs
#
def computeRadiationExtraneousSource(psim_src, psip_src, mesh, t):

   # call radiation vector evaluation function
   return computeRadiationVector(psim_src, psip_src, mesh, t)








