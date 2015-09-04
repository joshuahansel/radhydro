## @package src.globalConstants
#  Contains global constants necessary for solving radiative transfer problems
#
#  Constants may be accessed like a C++ namespace as follows:
#
# import globalConstants as GC
#
# a = GC.RAD_CONSTANT # get radiation constant
# c = GC.SPD_OF_LGT   # get speed of light, etc.
#
# Our standard units are length: cm, time: shakes (sh),
# energy and temperature: keV, mass: g, volume: cm^3
#
# Temperatures are done as an equivalent temperature, so 
# 
# T = T (K) * Boltzmann Constant (keV/K)
# 
# All constants will are in units so that temperatures can just be treated as in 
# keV.  This is confusing for specific heats, which will have units of
#
# C_V (energy/(temp * volume)) has units (keV/(keV*cm^3))

##shakes per second
SH_PER_S = 1.E+8  

##Speed of light in units of cm/shake
SPD_OF_LGT = 299.7924580

##Boltzman constant k in keV/K, for converting to equivalent energy of T
BOLTZMAN_CONSTANT = 8.617343E-08 

##grey radiation constant, a, in jk/(cm^3*keV^4)
RAD_CONSTANT = 0.013720172

##jk/J ("jerks" per Joules)
JK_PER_JOULES = 1.E-09 

##eV/keV
EV_PER_KEV = 1000. 

##Joules per KEV to convert some specific heat constants
JOULES_PER_KEV = 1.602176565e-16

##Some times specific heats are given in ergs
JOULES_PER_ERG = 1.E-07

##To convert from jerks to energy in KEV, our standard energy unit is KEV
JK_PER_KEV = 1.602176565e-25

##Plancks constant in jk-sh
PLANCKS_CONSTANT = 6.6260693E-35 
