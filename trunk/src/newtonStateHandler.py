## @package src.newtonStateHanlder
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
from transientSource import TransientSourceTerm
from crossXInterface import CrossXInterface
from copy            import deepcopy
import globalConstants as GC

#===================================================================================
## Main class to handle newton solve and temperature updates, etc.
#
class NewtonStateHandler(TransientSourceTerm):

    #--------------------------------------------------------------------------------
    ## Constructor
    #
    # @param [in] mesh            spatial mesh 
    # @param [in] delta_t         The time step size is required throughout class
    # @param [in] time_stepper    What type of time stepping method, e.g., 'BE'
    # @param [in] hydro_states_implicit  The initial guess for hydro states at end of
    #                             time step. These WILL be modified. Stored in usual
    #                             tuple format (L,R)
    #
    def __init__(self,mesh,hydro_states_implicit=None,time_stepper='BE'):

        TransientSourceTerm.__init__(self,mesh,time_stepper)

        #store scale coefficient from time discretization for computing nu
        scale = {"CN":0.5, "BE":1., "BDF2":2./3.}
        self.mesh = mesh
        self.time_stepper = time_stepper
        self.scale = scale[time_stepper]

        #Store the (initial) implicit hydro_states at t_{n+1}. These WILL be modified
        self.hydro_states = hydro_states_implicit

    #---------------------------------------------------------------------------------
    def getFinalHydroStates():

        #destroy the hydro states so no one accidentally reuses these for now
        temp_states = self.hydro_states
        self.hydro_states = []

        return temp_states


    #--------------------------------------------------------------------------------
    ## Function to generate effective cross sections for linearization. 
    #  
    #  The effective re-emission source is included as a scattering cross section
    #  given by
    #  \f[
    #      \tilde{\sigma_s} = \sigma_s + \sigma_a(T^k)*\nu
    #  \f]
    #  It is noted that \nu is different for the different time stepping algorithms
    #
    # @param [in] cx_orig    Pass in the original cross sections, these are NOT
    #                        modified. Return effective cross section
    #
    def getEffectiveOpacities(self, cx_orig, dt):

        cx_effective = []

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
                cx_orig[i][x].updateCrossX(rho,T) #if constant this call does nothing

                #Calculate the effective scattering cross sections
                sig_a_og = cx_orig[i][x].sig_a
                sig_s_og = cx_orig[i][x].sig_s
                nu    = getNu(T,sig_a_og,state.rho,state.spec_heat,dt,self.scale)
                print T, sig_a_og, state.rho, state.spec_heat, dt, self.scale
            
                #Create new FIXED cross section instance
                cx_i.append( CrossXInterface(sig_a_og, sig_s_og + nu*sig_a_og) )

            cx_effective.append(tuple(cx_i))

        return cx_effective

    #================================================================================
    #   The following functions are for the TransientSourceTerm class
    #--------------------------------------------------------------------------------
    ## Evaluate implicit term
    #
    def evalImplicit(self, i, **kwargs):
        
        T =  5.0375116316e-02 
        rho = 1.0000000000e+01 
        spec_heat = 1.0000000000e-01
        dt = 1.0000000000e-03 
        sigma_a = 2.0000000000e+03
        nu  =  4.1888035417e-03
        Sigma_s  = 8.3776070833e+00

        print "Actual nu: ", nu
        print "Actual sigs: ", Sigma_s
        print "My nu: ", getNu(T,sigma_a,rho, spec_heat, dt, self.scale)

        raise NotImplementedError("not done")

    #--------------------------------------------------------------------------------
    ## Evaluate implicit term
    #
    def evalOld(self, **kwargs):

        raise NotImplementedError("not done")

    #--------------------------------------------------------------------------------
    ## Evaluate BDF2 older term
    #
    def evalOlder(self, **kwargs):

        raise NotImplementedError("not done")

#=====================================================================================
# Miscellaneous functions that do not need to be a member function
#=====================================================================================
#
#--------------------------------------------------------------------------------
## Evaluate nu in linearization at a arbitrary temperature 'T'
#
def getNu(T, sig_a, rho, spec_heat, dt, scale):

    c_dt = GC.SPD_OF_LGT*dt
    beta = 4.*GC.RAD_CONSTANT * T * T * T / spec_heat #DU_R/De

    #Evaluate numerator
    num  = scale*sig_a*c_dt*beta/rho

    return num/(1. + num)




