## @package src.radUtilities
#  Contains radiation utilities

import globalConstants as GC  # global constants
import operator               # for adding tuples to each other elementwise
from math import sqrt

## directions for S-2, \f$\mu^\pm\f$
mu = {"-" : -1.0/sqrt(3.0), "+" : 1.0/sqrt(3.0)}

## Function to compute scalar flux from angular fluxes.
#
#  @param[in] psi_minus S-2 angular flux solution for the minus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^-\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_minus[i]\f$=(\Psi^-_{i,L},
#                       \Psi^-_{i,R})\f$
#  @param[in] psi_plus  S-2 angular flux solution for the plus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^+\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_plus[i]\f$=(\Psi^+_{i,L},
#                       \Psi^+_{i,R})\f$
#  @return  scalar flux solution, \f$\phi\f$, as an array of tuples of left
#           and right values, e.g., scalar_flux[i]\f$=(\phi_{i,L},\phi_{i,R})\f$
#
def computeScalarFlux(psi_minus, psi_plus):
   scalar_flux = [tuple(y for y in tuple(map(operator.add, psi_minus[i], psi_plus[i]))
                       ) for i in xrange(len(psi_minus))]
   return scalar_flux

## Function to compute energy density E from angular fluxes.
#
#  @param[in] psi_minus S-2 angular flux solution for the minus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^-\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_minus[i]\f$=(\Psi^-_{i,L},
#                       \Psi^-_{i,R})\f$
#  @param[in] psi_plus  S-2 angular flux solution for the plus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^+\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_plus[i]\f$=(\Psi^+_{i,L},
#                       \Psi^+_{i,R})\f$
#  @return  energy density, \f$E\f$, as an array of tuples of left
#           and right values, e.g., E[i]\f$=(E_{i,L},E_{i,R})\f$
#
def computeEnergyDensity(psi_minus, psi_plus):
   E = [tuple((1./GC.SPD_OF_LGT)*y for y in tuple(map(operator.add, psi_minus[i], psi_plus[i]))
                       ) for i in xrange(len(psi_minus))]
   return E
