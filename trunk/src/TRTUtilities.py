## @package src.TRTUtilities
#  Contains material and rad utilities useful in TRT problems, e.g., unit conversions

import globalConstants as GC  # global constants
from radUtilities import computeEnergyDensity
from math import sqrt


## Function to convert specific heat from ergs/(ev-g) to jks/(keV-g) (the true units)
#
#  @param[in] spec_heat specific heat in units of (ergs/ev-g)
# 
#  @return    spec_heat in jks/(keV*g), this is the standard unit in the code
#
def convSpecHeatErgsEvToJksKev(spec_heat):
   
   return spec_heat*(GC.JOULES_PER_ERG*GC.JK_PER_JOULES) * GC.EV_PER_KEV

## Function to compute equivalent isotropic intensity for a given energy
#
# @param[in] temp_rad   equivalent temperature in keV desired
# @return    the equivalent isotropic radiation intensity for the temperature
def computeEquivIntensity(temperature):
 
   return 0.5*GC.SPD_OF_LGT*GC.RAD_CONSTANT * (temperature**4.0) 


## Function to compute equivalent radiation temperature from psi_plus and psi_minus:
#
#  \[
#     T_r = \left(\frac{E}{a}\right)^{1/4}
#  \]
#
def computeRadTemp(psi_minus, psi_plus):
    
    # compute radiation energy
    E = computeEnergyDensity(psi_minus,psi_plus)
    a = GC.RAD_CONSTANT

    #compute equivalent temperature
    T_rad  = [((E_elem[0]/a)**0.25,(E_elem[1]/a)**0.25) for E_elem in E]
    
    return T_rad


## Takes a step of the radiation hydrodynamics scheme
#
def radHydroStep(hydro_old, psi_old, dt, bc_shit):

   # compute time step size, considering CFL condition
   dt = computeTimeStepSize(hydro_old, psi_old)

   # take first stage of MUSCL-Hancock step
   hydro_star = MUSCLHancockEvolve(hydro_old, 0.5*dt)

   # nonlinear iteration loop
   not_converged = True
   hydro_new = hydro_old
   while not_converged:

      # update velocities with new momentum deposition
      hydro_new = updateVelocity(hydro_old, hydro_new, hydro_star, 
                     psi_old, psi_new, 0.25*dt)

      # update angular fluxes
      psi_new = updateRadiation(hydro_old, hydro_new, psi_old, psi_new, 0.25*dt)
