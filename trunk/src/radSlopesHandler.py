
import numpy as np
from math import isinf

## Function to compute Delta E_* that is consistent with e_rad values
# 
# 
#  @param[in] states     Hydro states we want to generate corresponding edge
#                        values for
#  @param[in] hydro_slopes Corresponding raw hydro slopes
#  @param[in] e_rad      Values from radiation for internal energy at edges
def computeTotalEnergySlopes(states, slopes, erad):

   #Use conserved variables to construct 
   E_slopes = np.zeros(len(states))
   for i in xrange(len(states)):

       rho, mom, erg = states[i].getConservativeVariables()

       #Need all values at edges
       rho_l = rho - 0.5*slopes.rho_slopes[i]
       rho_r = rho + 0.5*slopes.rho_slopes[i]
       mom_l = mom - 0.5*slopes.mom_slopes[i]
       mom_r = mom + 0.5*slopes.mom_slopes[i]
       erg_l = erg - 0.5*slopes.erg_slopes[i]
       erg_r = erg + 0.5*slopes.erg_slopes[i]

       # compute internal energies at left and right edges
       u_l = mom_l/rho_l
       u_r = mom_r/rho_r

       #Compute total energy on edge, using e_rad values
       #for slope, but average from hydro
       e_avg = states[i].e

       de = erad[i][1]-erad[i][0]
       e_l = e_avg - 0.5*de
       e_r = e_avg + 0.5*de

       #Edge total energies
       E_l = rho_l*(e_l + 0.5*u_l**2)
       E_r = rho_r*(e_r + 0.5*u_r**2)

       #Slope to return
       E_slopes[i] = E_r - E_l
        
   # compute edge temperature using internal energy slope
   return E_slopes
