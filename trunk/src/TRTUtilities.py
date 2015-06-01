## @package src.materialUtilities
#  Contains material and rad utilities useful in TRT problems, e.g., unit conversions

import globalConstants as GC  # global constants
from radUtilities import computeEnergyDensity
from math import sqrt


## Function to convert specific heat from ergs/(ev-g) to jks/(keV-g) (the true units)
#
#  @param[in] spec_heat spec heat in units of (ergs/ev-g)
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


## Function to compute equvialent radiation temperature from psi_plus and psi_minus
#
def computeRadTemp(psi_minus,psi_plus ):
    
    # compute scalar flux first
    E = computeEnergyDensity(psi_minus,psi_plus)
    a = GC.RAD_CONSTANT

    #compute evialent temperature
    T_rad  = [((i[0]/a)**0.25,(i[1]/a)**0.25) for i in E]
    
    return T_rad


