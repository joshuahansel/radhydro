## @package src.newtonStateHandler
#  This file contains a class to handle computing linearized source
#  terms and handles temperature updates during a non-linear TRT solve.
#  Also contains auxilary functions for evaluating TRT functions of interest
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
from crossXInterface import CrossXInterface
from copy            import deepcopy
import globalConstants as GC
import numpy as np
import utilityFunctions as UT

#===================================================================================
## Main class to handle newton solve and temperature updates, etc.
#
class NewtonStateHandler(object):

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


        #store scale coefficient from time discretization for computing nu
        scale = {"CN":0.5, "BE":1., "BDF2":2./3.}
        self.mesh = mesh
        self.time_stepper = time_stepper
        self.scale = scale[time_stepper]

        #initialize the new hydro state to the provided guess state. These WILL be modified
        self.hydro_states = hydro_guess

        #Store the cross sections, you will be updating these 
        self.cx_new = cx_new

        #Create and store the known source terms Q_E^k. This will be updated
        #self.E = luate


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
            
                #Create new FIXED cross section instance. No need to add scale term
                #here because it will be included in scattering source term
                sig_s_new = nu*sig_a_og+sig_s_og
                cx_i.append( CrossXInterface(sig_s_new, sig_s_og+sig_a_og) )

            cx_effective.append(tuple(cx_i))

        return cx_effective

    #--------------------------------------------------------------------------------
    ## Function to evaluate \f$Q_E^k\f$ in documentation. This is essentially everything
    #  else from the linearization that isnt included elsewhere that is part of
    #  lagged material motion and emission sources
    #
    # NOTE: some of the terms will be duplicated here and in construction of the
    # source. For example, for CN, \f$-sigma_a a c T^{4,n}\f$ is here (but multiplied
    # by \nu), but it is also present from the time discretization (without the nu
    # and of opposite sign) in the actual transport equation. This will result in
    # (1-nu) factors multiplying the sources, but we dont worry about that here, just
    # build it as \f$Q_E^k\f$
    def evalMatCouplingQE():

        if self.time_stepper == "BE":

            #Add

            raise NotImplementedError("Just need to add Q from docs")

        elif self.time_stepper == "BDF2":


            raise NotImplementedError("Just need to add Q from docs")


        elif self.time_stepper == "CN":

            #add in term from planckian and absorption old

            raise NotImplementedError("Just need to add Q from docs")

        else:

            raise NotImplementedError("Invalid time stepper")

    ## Computes new velocities \f$u^{k+1}\f$
    #
    def updateVelocity():

       # TODO: do this
       return

    #--------------------------------------------------------------------------------
    ## Computes a new internal energy \f$e^{k+1}\f$ in each of the states, based on a passed in
    #  solution for E^{k+1}
    #
    def updateIntEnergy(self, E_new, dt, hydro_star=None):

        # constants
        a = GC.RAD_CONSTANT
        c = GC.SPD_OF_LGT

        #Evaluate knwon srt term QE for all the elements
        #QE_elems = evalMatCouplingQE(

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

                # compute a*T^4
                aT4 = a*T_prev**4.

                #Will need scale factor
                scale = self.scale

                #Evaluate extra terms from rad hydro
                QE = 0.
                

                #Calculate a new internal energy and store it (TODO: NEEDS Q AND KIN. ERG. TERM INCLUDED)
                e_new = (1.-nu)*scale*dt/state.rho * (sig_a*c*(E_new[i][x] - aT4)\
                   + 2.0*QE) + (1.-nu)*e_star + nu*e_prev
                self.hydro_states[i][x].e = e_new


    #================================================================================
    #   The following functions are for the TransientSourceTerm class
    #--------------------------------------------------------------------------------
    ## Evaluate new emission term \f$\frac{1}{2}\sigma_a^k a c(T^{k+1})^4\f$
    #
    #  @param [in] hydro_star  If there is no material motion, this is simply
    #                                 the hydro states at t_n. But if there is material
    #                                 motion these come from the MUSCL hancock. It is
    #                                 assumed they have been adjusted to use the
    #                                 correct slope before this function is called
    #  @param[in] dt           time step
    #
    def evalPlanckianImplicit(self, hydro_star=None, dt=None,**kwargs):
        
        #calculate at left and right, isotropic emission source
        cx_new = self.cx_new #local reference

        # get constants
        a = GC.RAD_CONSTANT
        c = GC.SPD_OF_LGT

        #loop over cells
        Q = []
        for i in range(self.mesh.n_elems):

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
                emission = (1. - nu )*sig_a*a*c*T**4.
                
                #add in additional term from internal energy
                planckian[edge] = emission - (
                        nu*state.rho/(self.scale*dt)* (state.e  - state_star.e ) )

                planckian[edge] *= 0.5

            #Store the (isotropic) sources in correct index
            Q.append(tuple(planckian))

        return Q


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


