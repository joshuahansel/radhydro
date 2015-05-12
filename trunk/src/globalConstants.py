## This module contains all the global constants that are necessary for solving
#  radiative transfer problems. Access them like a C++ namespace as follows
#
#
# import globalConstants as GC
#
# a = GC.RAD_CONST #get radiation constant
# c = GC.SPD_OF_LIGHT #get speed of light, etc.
#
#

# All constants are at global scope in this module, please dont change them :)

SH_PER_S = 1.E+8  #shakes per second
SPD_OF_LGT = 299.792458L #Units of cm/shake
BOLTZMAN_CONSTANT = 8.617343E-08 #keV/K
RAD_CONSTANT = 0.01372 #grey radiation constant, a, in jk/(cm^3*keV^4)
JK_PER_JOULES = 1.E-09 #jk/J ("jerks" per Joules)
EV_PER_KEV = 1000. #eV/keV
JOULES_PER_KEV = 1.602176565e-16
JOULES_PER_ERG = 1.E-07 #Some times specific heats are given in ergs
JK_PER_KEV = 1.602176565e-25
PLANCKS_CONSTANT = 6.6260693E-35 #jk-sh
