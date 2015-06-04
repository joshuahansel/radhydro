## @package src.newtonStateHandler
#  This file contains a class to handle computing linearized source
#  terms and handles temperature updates during a non-linear TRT solve.
#  Also contains auxilary functions for evaluating TRT functions of interest
#  NOTE: this class should not do the entire non-linear solve because it is also
#  responsible for evaluating the planckian source in the solver, so it wouldnt make
#  sense for it to pass itsself to the source builder
#
#  This class is eventually passed to the source builder as source. Although it is
#  has additional features, it is easier to have this class directly implement the
#  evalImplicit etc. functions than using forwarding functions and potentially having
#  the wrong states.
#
#  The documentation for the linearization and solution of non-linear system 
#  derives in detail the equations being generated here.
#
from musclHancock    import HydroState
from transientSource import TransientSourceTerm
from crossXInterface import CrossXInterface
from copy            import deepcopy
import globalConstants as GC
import numpy as np
import utilityFunctions as UT

#===================================================================================
## Main class to handle newton solve and temperature updates, etc.
#
class NewtonStateHandler(TransientSourceTerm):

    #--------------------------------------------------------------------------------
    ## Constructor
    #
    # @param [in] mesh          spatial mesh 
    # @param [in] delta_t       time step size, required throughout class
    # @param [in] time_stepper  specifier for time-stepping method, e.g., 'BE'
    # @param [in] hydro_guess   initial guess for new hydro state,
    #                           usually taken to be old hydro state.
    #                           This WILL be modified.
    #                               TODO have the constructor adjust star to use rad
    #                               slopes
    #
    def __init__(self,mesh,cx_new=None,hydro_guess=None,time_stepper='BE'):

        TransientSourceTerm.__init__(self,mesh,time_stepper)

        #store scale coefficient from time discretization for computing nu
        scale = {"CN":0.5, "BE":1., "BDF2":2./3.}
        self.mesh = mesh
        self.time_stepper = time_stepper
        self.scale = scale[time_stepper]

        #initialize the new hydro state to the provided guess state. These WILL be modified
        self.hydro_states = hydro_guess

        #Store the cross sections, you will be updating these 
        self.cx_new = cx_new


    #---------------------------------------------------------------------------------
    ## Returns the new hydro states that are not converged
    def getNewHydroStates(self):

        return deepcopy(self.hydro_states)

    #--------------------------------------------------------------------------------
    ## Function to generate effective cross sections for linearization. 
    #  
    #  The effective re-emission source is included as a scattering cross section
    #  given by
    #  \f[
    #      \tilde{\sigma_s} = \sigma_s + \sigma_a(T^k)\nu
    #  \f]
    #  It is noted that \f$\nu\f$ is different for the different time stepping algorithms
    #
    # @param [in] cx_orig    Pass in the original cross sections, these are NOT
    #                        modified. Return effective cross section
    #
    def getEffectiveOpacities(self, dt):

        cx_effective = []
        cx_orig = self.cx_new #local reference

        # loop over cells:
        for i in range(len(cx_orig)):

            cx_i = []

            # calculate at left and right
            for x in range(2): 

                #Get the guessed temperature from states
                state = self.hydro_states[i][x]
                T     = state.getTemperature()
                rho   = state.rho

                #Make sure cross section is updated with temperature
                cx_orig[i][x].updateCrossX(rho=rho, temp=T) #if constant this call does nothing

                #Calculate the effective scattering cross sections
                sig_a_og = cx_orig[i][x].sig_a
                sig_s_og = cx_orig[i][x].sig_s
                nu    = getNu(T,sig_a_og,state.rho,state.spec_heat,dt,self.scale)
            
                #Create new FIXED cross section instance
                sig_s_new = nu*sig_a_og+sig_s_og
                cx_i.append( CrossXInterface(sig_s_new, sig_s_og+sig_a_og) )

            cx_effective.append(tuple(cx_i))

        return cx_effective

    #--------------------------------------------------------------------------------
    ## Function to evaluate \f$Q_E^k\f$ in documentation. This is essentially everything
    #  else from the linearization that isnt included elsewhere
    #
    def getQE():

        if self.time_stepper == "BE":

            return 0.0
        else:

            raise NotImplementedError("Not really sure what to do here yet")

    ## Computes new velocities \f$u^{k+1}\f$
    #
    def updateVelocity():

       return

    #--------------------------------------------------------------------------------
    ## Computes a new internal energy in each of the states, based on a passed in
    #  solution for E^{k+1}
    #
    def updateIntEnergy(self, E, dt, hydro_star=None):

        #constants
        a = GC.RAD_CONSTANT
        c = GC.SPD_OF_LGT

        if self.time_stepper != 'BE':

            raise NotImplementedError("This only works for BE currently")

        #loop over cells
        for i in xrange(len(self.hydro_states)):

            #loop over left and right value
            for x in range(2):

                #get temperature for this cell, at indice k, we are going to k+1
                state = self.hydro_states[i][x]
                T_prev  = state.getTemperature()
                e_prev = state.e

                #old state from hydro
                state_star = hydro_star[i][x]
                e_star = state_star.e

                sig_a = self.cx_new[i][x].sig_a
                nu    = getNu(T_prev,sig_a,state.rho,state.spec_heat,dt,self.scale)

                #Calculate planckian from T_prev (t_k)
                planck_prev = sig_a*a*c*T_prev**4.

                #Will need scale factor
                scale = self.scale
                
                #Calculate a new internal energy and store it (NEED SCALE FACTORS)
                e_new = (1.-nu) * scale* dt/state.rho * (sig_a*c*E[i][x] - planck_prev) \
                        + (1.-nu)*e_star + nu*e_prev
                self.hydro_states[i][x].e = e_new

                #print "New temps: ", i, self.hydro_states[i][x].getTemperature()

        c = GC.SPD_OF_LGT
    

    #================================================================================
    #   The following functions are for the TransientSourceTerm class
    #--------------------------------------------------------------------------------
    ## Evaluate new emission term \f$\sigma_a^k a c(T^{k+1})^4\f$
    #
    #  @param [in] hydro_star  If there is no material motion, this is simply
    #                                 the hydro states at t_n. But if there is material
    #                                 motion these come from the MUSCL hancock. It is
    #                                 assumed they have been adjusted to use the
    #                                 correct slope before this function is called
    #
    def evalImplicit(self, i, hydro_star=None, dt=None,**kwargs):
        
        #calculate at left and right, isotropic emission source
        cx_new = self.cx_new #local reference
        planckian = [0.0,0.0]
        for edge in range(2):

            #get temperature for this cell, at index k
            state = self.hydro_states[i][edge]
            T     = state.getTemperature()

            #old state from hydro
            state_star = hydro_star[i][edge]

            #Update cross section just in case
            cx_new[i][edge].updateCrossX(state.rho,T)

            sig_a = cx_new[i][edge].sig_a
            nu    = getNu(T,sig_a,state.rho,state.spec_heat,dt,self.scale)

            #Calculate planckian
            emission = (1. - nu )*sig_a*GC.RAD_CONSTANT*GC.SPD_OF_LGT*T**4.
            
            #add in additional term from internal energy
            planckian[edge] = emission - (nu*state.rho/(self.scale*dt) 
                    * (state.e  - state_star.e ) )

        #Store the (isotropic) sources in correct index
        Q = np.zeros(4)
        Q[UT.getLocalIndex("L","-")] = 0.5*planckian[0]
        Q[UT.getLocalIndex("L","+")] = 0.5*planckian[0]
        Q[UT.getLocalIndex("R","-")] = 0.5*planckian[1]
        Q[UT.getLocalIndex("R","+")] = 0.5*planckian[1]
        
        return Q

    #--------------------------------------------------------------------------------
    ## Evaluate old term
    #
    def evalOld(self, **kwargs):

        raise NotImplementedError("not done")

    #--------------------------------------------------------------------------------
    ## Evaluate older term
    #
    def evalOlder(self, **kwargs):

        raise NotImplementedError("not done")

#=====================================================================================
# Miscellaneous functions that do not need to be a member function
#=====================================================================================
#
#--------------------------------------------------------------------------------
## Computes effective scattering fraction \f$\nu^k\f$ in linearization at an
#  arbitrary temperature \f$T^k\f$
#
#  @param[in] T      Previous iteration temperature \f$T^k\f$
#  @param[in] sig_a  Previous iteration absorption cross section \f$\sigma_a^k\f$
#  @param[in] rho    New density \f$\rho^{n+1}\f$
#  @param[in] spec_heat   Previous iteration specific heat \f$c_v^k\f$
#  @param[in] dt          time step size \f$\Delta t\f$
#  @param[in] scale       coefficient corresponding to time-stepper \f$\gamma\f$
#
def getNu(T, sig_a, rho, spec_heat, dt, scale):

    ## compute \f$c\Delta t\f$
    c_dt = GC.SPD_OF_LGT*dt

    ## compute \f$\beta^k\f$
    beta = 4.*GC.RAD_CONSTANT * T * T * T / spec_heat

    # Evaluate numerator
    num  = scale*sig_a*c_dt*beta/rho

    return num/(1. + num)


