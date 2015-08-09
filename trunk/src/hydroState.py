## @package src.hydroState
#  Contains class for describing a hydrodynamic state.

from math import sqrt

## Class for defining the hydrodynamic state at a point
#
class HydroState:

    ## Constructor
    #
    def __init__(self, rho, u, gamma, spec_heat, p=None,
            e=None, T=None, E=None):

        self.u = u
        self.rho = rho
        self.gamma = gamma
        self.spec_heat = spec_heat

        second_intensive_property_specified = False

        # ensure that only 2 intensive properties have been supplied
        # (one is density)
        if p != None:

            assert not second_intensive_property_specified, 'Only 2\
               independent, intensive properties may be specified'
            second_intensive_property_specified = True
                
            self.p = p
            self.e = getIntErg(gamma,rho,p)

        if e != None:

            assert not second_intensive_property_specified, 'Only 2\
               independent, intensive properties may be specified'
            second_intensive_property_specified = True

            self.e = e
            self.p = getPressure(gamma,rho,self.e)

        if T != None:

            assert not second_intensive_property_specified, 'Only 2\
               independent, intensive properties may be specified'
            second_intensive_property_specified = True

            self.e = self.computeIntEnergy(T)
            self.p = getPressure(gamma,rho,self.e)

        if E != None:

            assert not second_intensive_property_specified, 'Only 2\
               independent, intensive properties may be specified'
            second_intensive_property_specified = True

            self.e = self.computeInternalEnergyFromTotalEnergy(E)
            self.p = getPressure(gamma,rho,self.e)

        if not second_intensive_property_specified:
            raise IOError("You must specify pressure, internal energy,\
               temperature, or total energy in HydroState constructor")


    #solve for new values based on a consState variables
    def updateState(self, rho, mom, erg):

        self.rho = rho
        self.u = mom/rho
        self.e = erg/rho - 0.5*self.u*self.u
        self.p = getPressure(self.gamma, self.rho, self.e) 

    ## Updates velocity.
    #
    #  @param[in] u  new velocity \f$u\f$
    #
    def updateVelocity(self, u):
       self.u = u

    ## Updates state based on density and internal energy.
    #
    #  This function is used by the radiation-hydrodynamics scheme to update
    #  the hydro state with new internal energies when they are computed.
    #  It is assumed that the velocity has already been updated earlier in
    #  the time step with the updateVelocity() function. At this point,
    #  both the new velocity field and the new thermodynamic state of the
    #  fluid have been fully defined, so an update to all thermodynamic
    #  quantities is performed in this function.
    #
    #  @param[in] rho  new density \f$\rho\f$
    #  @param[in] e    new internal energy \f$e\f$
    #
    def updateStateDensityInternalEnergy(self, rho, e):

       self.rho = rho
       self.e   = e
       self.p = getPressure(self.gamma, self.rho, self.e) 

    ## Computes sound speed.
    #
    def getSoundSpeed(self):

        return sqrt(self.gamma*self.p/self.rho)

    #-------------------------------------------------------------------------------
    # Get temperature based on internal energy and constant specific heat
    #
    def getTemperature(self):

        return self.e/(self.spec_heat)

    ## Computes specific internal energy from temperature
    #
    def computeIntEnergy(self,temp):

        return temp*self.spec_heat

    ## Computes specific internal energy \f$e\f$ from total energy density \f$E\f$
    #
    def computeInternalEnergyFromTotalEnergy(self,E):

        return E/self.rho - 0.5*self.u**2

    ## Compare function
    #
    def __eq__(self, other):

        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    ## Print definition
    #
    def __str__(self):

        return "u: %.15f rho: %.15f e: %.15f p: %.15f E: %.15f" % (self.u, self.rho,
                self.e, self.p, self.rho*(self.u**2.*0.5 + self.e))

    ## Prints conservative variables
    #
    def printConservativeVariables(self):

       rho, mom, erg = self.getConservativeVariables()
       print "%f %f %f" % (rho, mom, erg)

    ## Returns conservative variables
    #
    #  @return conservative variables:
    #     -# \f$\rho\f$
    #     -# \f$\rho u\f$
    #     -# \f$E\f$
    #
    def getConservativeVariables(self):
    
       rho = self.rho
       u   = self.u
       mom = rho * u
       erg = rho * (0.5*u*u + self.e)
    
       return rho, mom, erg


## Computes volume
#
def getVolume(x1,x2):

    return (x2-x1)*1.0

## Computes pressure
#
def getPressure(gamma,rho,e):

    return (gamma-1.0)*rho*e

## Computes internal energy
#
def getIntErg(gamma, rho, p):

    return p/((gamma-1.0)*rho)

## Computes velocity
#
def computeVelocity(rho, mom):

   return mom / rho

## Computes internal energy
#
def computeIntEnergy(rho, mom, erg):

   u = computeVelocity(rho=rho, mom=mom)
   return erg / rho - 0.5*u*u

## Computes pressure
#
def computePressure(rho, mom, erg, gam):

   e = computeIntEnergy(rho=rho, mom=mom, erg=erg)
   return rho*e*(gam - 1.0)

