
import numpy as np
from math import isinf

## Class for handling slope and computing edge internal energies to encapsulate slope
# changes due to ensuring the diffusion limit
#
class RadSlopes:

    ## Constructor
    #
    #  @param[in] states_star   the raw averages from musl solver
    #  @param[in] slopes_star   the raw slopes from MUSCL solver
    #  @param[in] e_rad         the edge values of e from radiation
    #  TODO this can be modified to be done purely in terms of slopes,
    #  but I did it this way just to have something to check the slope
    #  way remotely works
    #
    def __init__(self, states_in, slopes_in):

        #keep list names
        self.states = states_in
        self.slopes = slopes_in
        self.n      = len(states_in)

    ## Function to compute Delta E_* that is consistent with e_rad values
    # 
    #  @param[in] e_avg      the value of e at centers corresponding to slope
    #  @param[in] e_slope    the edge values of e from radiation solve, in theory
    #                      we could/should? be updating the E slope after each radiation solve
    def getTotalEnergySlopes(self, e_avg, e_slope):

       #Use conserved variables to construct 
       E_slopes = np.zeros(self.n)
       for i in xrange(len(self.states)):

           rho, mom, erg = self.states[i].getConservativeVariables()

           #Need all values at edges
           rho_l = rho - 0.5*self.slopes.rho_slopes[i]
           rho_r = rho + 0.5*self.slopes.rho_slopes[i]
           mom_l = mom - 0.5*self.slopes.mom_slopes[i]
           mom_r = mom + 0.5*self.slopes.mom_slopes[i]
           erg_l = erg - 0.5*self.slopes.erg_slopes[i]
           erg_r = erg + 0.5*self.slopes.erg_slopes[i]

           # compute internal energies at left and right edges
           u_l = mom_l/rho_l
           u_r = mom_r/rho_r

           #Compute total energy on edge, using e_rad values
           e_l = e_avg[i] - 0.5*e_slope[i]
           e_r = e_avg[i] + 0.5*e_slope[i]

           #Edge total energies
           E_l = rho_l*(e_l + 0.5*u_l**2)
           E_r = rho_r*(e_r + 0.5*u_r**2)

           #Slope to return
           E_slopes[i] = E_r - E_l
            
       # compute edge temperature using internal energy slope
       return E_slopes
