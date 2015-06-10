from crossXInterface import CrossXInterface
from transientSource import TransientSourceTerm, evalPlanckianOld
from copy            import deepcopy
import globalConstants as GC
import numpy as np
import utilityFunctions as UT
from utilityFunctions import getNu

#--------------------------------------------------------------------------------
## Updates velocities.
#
#  param[in,out] hydro_new
#
def updateVelocity(mesh, time_stepper, dt, hydro_star, hydro_new, **kwargs):
 
    # compute source
    src_handler = VelocityUpdateSourceHandler(mesh, time_stepper)
    Q = src_handler.computeTerm(**kwargs)

    # loop over cells
    for i in range(mesh.n_elems):

        # loop over edges
        for x in range(2):

            # update velocity
            hydro_new[i][x].u = hydro_star[i][x].u + dt/hydro_new[i][x].rho*Q[i][x]


#--------------------------------------------------------------------------------
## Updates internal energy.
#
#  param[in,out] hydro_new  contains new velocities
#
def updateInternalEnergy(time_stepper, dt, QE, cx_prev, rad_new, hydro_new,
    hydro_prev, hydro_star):
 
    # constants
    a = GC.RAD_CONSTANT
    c = GC.SPD_OF_LGT

    # get coefficient corresponding to time-stepper
    scales = {"CN":0.5, "BE":1., "BDF2":2./3.}
    scale = scales[time_stepper]

    # loop over cells
    for i in xrange(len(hydro_new)):

        #loop over left and right value
        for x in range(2):

            # get new quantities
            state_new = hydro_new[i][x]
            u_new = state_new.u
            E = rad_new.E[i][x]

            # get previous quantities
            state_prev = hydro_prev[i][x]
            T_prev  = state_prev.getTemperature()
            aT4 = a*T_prev**4
            e_prev = state_prev.e
            sig_a = cx_prev[i][x].sig_a
            spec_heat = state_prev.spec_heat

            # get star quantities
            state_star = hydro_star[i][x]
            e_star = state_star.e
            u_star = state_star.u
            rho = state_star.rho

            # compute effective scattering ratio
            nu = getNu(T_prev, sig_a, rho, spec_heat, dt, scale)

            #Evaluate extra terms from rad hydro
            QE_elem = QE[i][x]

            # compute new internal energy
            e_new = (1.0-nu)*scale*dt/rho * (sig_a*c*(E - aT4) + QE_elem/scale)\
               + (1.0-nu)*e_star + nu*e_prev - 0.5*(1.0-nu)*(u_new**2 - u_star**2)

            # put new internal energies in new hydro
            hydro_new[i][x].e = e_new


#-----------------------------------------------------------------------------------
## Computes estimated energy gain \f$Q\f$ due to the coupling to radiation energy field
#  based on estimated co-moving frame flux. This is used in the energy update
#  equation in the computation of \f$Q_E\f$:
#  \f[
#      Q = \sigma_t \frac{u}{c} \left(\mathcal{F} - \frac{4}{3}\mathcal{E} u\right)
#  \f]
#
def evalEnergyExchange(i, rad, hydro, cx):

    momentum_exchange_term = evalMomentumExchange(i, rad, hydro, cx)
    return [momentum_exchange_term[x]*hydro[i][x].u for x in xrange(2)]


#------------------------------------------------------------------------------------
## Compute estimated momentum exchange \f$Q\f$ due to the coupling to radiation,
#  used the in the velocity update equation:
#  \f[
#      Q = \frac{\sigma_t}{c} \left(F - \frac{4}{3}E u\right)
#  \f]
#
def evalMomentumExchange(i, rad, hydro, cx):

    return [cx[i][x].sig_t/GC.SPD_OF_LGT*(rad.F[i][x]
       - 4.0/3.0*rad.E[i][x]*hydro[i][x].u) for x in xrange(2)]


#------------------------------------------------------------------------------------
## Computes \f$\sigma_a\phi\f$
#
def evalEnergyAbsorption(i, rad, cx):

    return [rad.phi[i][x]*cx[i][x].sig_a for x in xrange(2)]

#=====================================================================================
## Class to simplify evaluating the Q_E^k term in linearization. It is similar to 
#  other terms but with the implicit Planckian removed and angularly integrated
#  quantities.
#
# It utilizes the TransientSource term class to build terms mostly just for
# simplicity of having access to the time stepping algorithms. The ultimate
# return of this function is just a list of tuples of QE for each elem
#
class QEHandler(TransientSourceTerm):

    #-------------------------------------------------------------------------------
    ## Constructor
    #
    def __init__(self, *args):

        # call base class constructor
        TransientSourceTerm.__init__(self, *args)

    #----------------------------------------------------------------------------
    ## Function to evaluate source at all cells in the mesh. Call the base clas
    #  version first, then angularly integrate and form the tuple array we need  
    #
    def computeTerm(self, **kwargs):

        # loop over all cells and build source 
        Q = [[0.0,0.0] for i in range(self.mesh.n_elems)]
        for i in range(self.mesh.n_elems):
            
            # add the source from element i
            Q_elem = list(self.func(i, **kwargs))
            for x in range(2):

                Q[i][x] += Q_elem[x]

        return Q

    #--------------------------------------------------------------------------------
    ## Computes implicit terms in QE^k. Only an energy exchange term, there is 
    #  no planckian 
    #
    def evalImplicit(self, i, rad_prev, hydro_prev, cx_prev, **kwargs):

        Q_local = np.array(evalEnergyExchange(i, rad=rad_prev, hydro=hydro_prev,
           cx=cx_prev))
        return Q_local

    #--------------------------------------------------------------------------------
    ## Evaluate old term. This includes Planckian, as well as energy exchange term
    # 
    def evalOld(self, i, rad_old, hydro_old, cx_old, **kwargs):

        Q_local = np.array(evalEnergyAbsorption(i, rad=rad_old, cx=cx_old))\
           - np.array(evalPlanckianOld(i, hydro=hydro_old, cx=cx_old))\
           + np.array(evalEnergyExchange(i, rad=rad_old, hydro=hydro_old, cx=cx_old))
        return Q_local

    #--------------------------------------------------------------------------------
    ## Evaluate older term. Just call the evalOld function as in other source terms
    #
    def evalOlder(self, i, rad_older, hydro_older, cx_older, **kwargs):

        return self.evalOld(i, rad_old=rad_older, hydro_old=hydro_older,
           cx_old=cx_older)


## Handles velocity update source term.
#
#
class VelocityUpdateSourceHandler(TransientSourceTerm):

    #-------------------------------------------------------------------------------
    ## Constructor
    #
    def __init__(self, *args):

        # call base class constructor
        TransientSourceTerm.__init__(self, *args)

    #----------------------------------------------------------------------------
    def computeTerm(self, **kwargs):

        # loop over all cells and build source 
        Q = [[0.0,0.0] for i in range(self.mesh.n_elems)]
        for i in range(self.mesh.n_elems):
            
            # add the source from element i
            Q_elem = list(self.func(i, **kwargs))
            for x in range(2):

                Q[i][x] += Q_elem[x]

        return Q

    #--------------------------------------------------------------------------------
    def evalImplicit(self, i, rad_prev, hydro_prev, cx_prev, **kwargs):

        Q_local = np.array(evalMomentumExchange(i, rad=rad_prev, hydro=hydro_prev,
           cx=cx_prev))
        return Q_local

    #--------------------------------------------------------------------------------
    def evalOld(self, i, rad_old, hydro_old, cx_old, **kwargs):

        return self.evalImplicit(i, rad_prev=rad_old, hydro_prev=hydro_old,
           cx_prev=cx_old)

    #--------------------------------------------------------------------------------
    def evalOlder(self, i, rad_older, hydro_older, cx_older, **kwargs):

        return self.evalImplicit(i, rad_prev=rad_older, hydro_prev=hydro_older,
           cx_prev=cx_older)


