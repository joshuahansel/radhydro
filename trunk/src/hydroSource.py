
from crossXInterface import CrossXInterface
from transientSource import TransientSourceTerm, evalPlanckianOld
from copy            import deepcopy
import globalConstants as GC
import numpy as np
import utilityFunctions as UT
from utilityFunctions import getNu, computeEdgeVelocities, computeEdgeTemperatures,\
   computeEdgeDensities, computeEdgeInternalEnergies

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

        # update velocity
        u_new = hydro_star[i].u + dt/hydro_new[i].rho*Q[i]
        hydro_new[i].updateVelocity(u_new)


#--------------------------------------------------------------------------------
## Updates internal energy.
#
#  param[in,out] hydro_new  contains new velocities
#
def updateInternalEnergy(time_stepper, dt, QE, cx_prev, rad_new, hydro_new,
    hydro_prev, hydro_star, slopes_old, e_slopes_old):
 
    # constants
    a = GC.RAD_CONSTANT
    c = GC.SPD_OF_LGT

    # get coefficient corresponding to time-stepper
    scales = {"CN":0.5, "BE":1., "BDF2":2./3.}
    scale = scales[time_stepper]

    # initialize new internal energy slopes
    e_slopes_new = np.zeros(len(hydro_new))

    # loop over cells
    for i in xrange(len(hydro_new)):

        # get hydro states
        state_new  = hydro_new[i]
        state_prev = hydro_prev[i]
        state_star = hydro_star[i]

        # compute edge densities
        rho = computeEdgeDensities(i, state_star, slopes_old)

        # compute edge velocities
        u_new  = computeEdgeVelocities(i, state_new,  slopes_old)
        u_star = computeEdgeVelocities(i, state_star, slopes_old)

        # compute edge temperatures
        T_prev = computeEdgeTemperatures(state_prev, e_slopes_old[i])

        # compute edge internal energies
        e_prev = computeEdgeInternalEnergies(state_prev, e_slopes_old[i])
        e_star = computeEdgeInternalEnergies(state_star, e_slopes_old[i])

        # loop over edges to compute new internal energies
        e_new = np.zeros(2)
        for x in range(2):

            # get new quantities
            Er = rad_new.E[i][x]

            # get previous quantities
            aT4 = a*T_prev[x]**4
            sig_a = cx_prev[i][x].sig_a
            spec_heat = state_prev.spec_heat

            # compute effective scattering ratio
            nu = getNu(T_prev[x], sig_a, rho[x], spec_heat, dt, scale)

            # evaluate additional source terms
            QE_elem = QE[i][x]

            # compute new internal energy
            e_new[x] = (1.0-nu)*scale*dt/rho[x] * (sig_a*c*(Er - aT4) + QE_elem/scale)\
               + (1.0-nu)*e_star[x] + nu*e_prev[x]\
               - 0.5*(1.0-nu)*(u_new[x]**2 - u_star[x]**2)

        # compute new average internal energy
        e_new_avg = 0.5*e_new[0] + 0.5*e_new[1]

        # put new internal energy in the new hydro state
        hydro_new[i].updateStateDensityInternalEnergy(state_star.rho, e_new_avg)

        # compute new internal energy slope
        e_slopes_new[i] = e_new[1] - e_new[0]

    # return new internal energy slopes
    return e_slopes_new


#-----------------------------------------------------------------------------------
## Computes estimated energy gain \f$Q\f$ due to the coupling to radiation energy field
#  based on estimated co-moving frame flux. This is used in the energy update
#  equation in the computation of \f$Q_E\f$:
#  \f[
#      Q = \sigma_t \frac{u}{c} \left(\mathcal{F} - \frac{4}{3}\mathcal{E} u\right)
#  \f]
#
def evalEnergyExchange(i, rad, hydro, cx, slopes):

    # compute edge velocities
    u = computeEdgeVelocities(i, hydro[i], slopes)

    # compute momentum exchange term
    momentum_exchange_term = evalMomentumExchange(i, rad, hydro, cx, slopes)

    return [momentum_exchange_term[x]*u[x] for x in xrange(2)]


#------------------------------------------------------------------------------------
## Compute estimated momentum exchange \f$Q\f$ due to the coupling to radiation,
#  used the in the velocity update equation:
#  \f[
#      Q = \frac{\sigma_t}{c} \left(F - \frac{4}{3}E u\right)
#  \f]
#
def evalMomentumExchange(i, rad, hydro, cx, slopes):

    # compute edge velocities
    u = computeEdgeVelocities(i, hydro[i], slopes)

    return [cx[i][x].sig_t/GC.SPD_OF_LGT*(rad.F[i][x]
       - 4.0/3.0*rad.E[i][x]*u[x]) for x in xrange(2)]


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
    def evalImplicit(self, i, rad_prev, hydro_prev, cx_prev, slopes_old,
       Qerg_new, **kwargs):

        Q_local = np.array(evalEnergyExchange(i, rad=rad_prev, hydro=hydro_prev,
                           cx=cx_prev, slopes=slopes_old))\
           + np.array(Qerg_new[i])
        return Q_local

    #--------------------------------------------------------------------------------
    ## Evaluate old term. This includes Planckian, as well as energy exchange term
    # 
    def evalOld(self, i, rad_old, hydro_old, cx_old, slopes_old, e_slopes_old,
       Qerg_old, **kwargs):

        Q_local = np.array(evalEnergyAbsorption(i, rad=rad_old, cx=cx_old))\
           - np.array(evalPlanckianOld(i, hydro_old=hydro_old, cx_old=cx_old,
                      e_slopes_old=e_slopes_old))\
           + np.array(evalEnergyExchange(i, rad=rad_old, hydro=hydro_old, cx=cx_old,
                      slopes=slopes_old))\
           + np.array(Qerg_old[i])
        return Q_local

    #--------------------------------------------------------------------------------
    ## Evaluate older term. Just call the evalOld function as in other source terms
    #
    def evalOlder(self, i, rad_older, hydro_older, cx_older, slopes_older,
       e_slopes_older, Qerg_older, **kwargs):

        return self.evalOld(i, rad_old=rad_older, hydro_old=hydro_older,
           cx_old=cx_older, slopes_old=slopes_older, e_slopes_old=e_slopes_older,
           Qerg_old=Qerg_older)


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
        Q = [0.0 for i in range(self.mesh.n_elems)]
        for i in range(self.mesh.n_elems):
            
            # add the source from element i
            Q_elem = self.func(i, **kwargs)
            Q[i] += Q_elem

        return Q

    #--------------------------------------------------------------------------------
    def evalImplicit(self, i, rad_prev, hydro_prev, cx_prev, Qmom_new, **kwargs):

        Q_local = evalMomentumExchangeAverage(i, rad=rad_prev, hydro=hydro_prev,
           cx=cx_prev) + Qmom_new[i]

        return Q_local

    #--------------------------------------------------------------------------------
    def evalOld(self, i, rad_old, hydro_old, cx_old, Qmom_old, **kwargs):

        Qreturn = self.evalImplicit(i, rad_prev=rad_old, hydro_prev=hydro_old,
           cx_prev=cx_old, Qmom_new=Qmom_old)

        return Qreturn

    #--------------------------------------------------------------------------------
    def evalOlder(self, i, rad_older, hydro_older, cx_older, Qmom_older, **kwargs):

        return self.evalImplicit(i, rad_prev=rad_older, hydro_prev=hydro_older,
           cx_prev=cx_older, Qmom_new=Qmom_older)


#------------------------------------------------------------------------------------
## Compute estimated momentum exchange \f$Q\f$ due to the coupling to radiation,
#  used the in the velocity update equation:
#  \f[
#      Q = \frac{\sigma_t}{c} \left(F - \frac{4}{3}E u\right)
#  \f]
#
def evalMomentumExchangeAverage(i, rad, hydro, cx):

    # compute average cross section
    sig_t = 0.5*cx[i][0].sig_t + 0.5*cx[i][1].sig_t

    # compute average radiation quantities
    E = 0.5*rad.E[i][0] + 0.5*rad.E[i][1]
    F = 0.5*rad.F[i][0] + 0.5*rad.F[i][1]

    return sig_t/GC.SPD_OF_LGT*(F - 4.0/3.0*E*hydro[i].u)


## Computes an extraneous source vector for the momentum equation
#
#  @param[in] mom_src  function handle for the momentum extraneous source
#  @param[in] mesh     mesh
#  @param[in] t        time at which to evaluate the function
#
#  @return list of the momentum extraneous source function evaluated
#          at each cell center
#
def computeMomentumExtraneousSource(mom_src, mesh, t):

   # evaluate momentum source function at each cell center
   return [mom_src(mesh.getElement(i).x_cent,t) for i in xrange(mesh.n_elems)]


## Computes an extraneous source vector for the energy equation
#
#  @param[in] erg_src  function handle for the energy extraneous source
#  @param[in] mesh     mesh
#  @param[in] t        time at which to evaluate the function
#
#  @return list of tuples of the energy extraneous source function evaluated
#          at each edge on each cell
#
def computeEnergyExtraneousSource(erg_src, mesh, t):

   # evaluate energy source function at each edge of each cell
   return [(erg_src(mesh.getElement(i).xl,t), erg_src(mesh.getElement(i).xr,t))
      for i in xrange(mesh.n_elems)]


