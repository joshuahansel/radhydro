from math import sqrt

## Class for defining the hydrodynamic state at a point
#
class HydroState:

    ## Constructor
    #
    def __init__(self, u=None ,rho=None, p=None, gamma=None, 
            spec_heat=None, int_energy=None, temp=None):

        self.u = u
        self.rho = rho
        self.gamma = gamma
        self.spec_heat = spec_heat

        # ensure p, T, and e are all consistent and not duplicate
        if p != None:
                
            self.p = p
            self.e = getIntErg(gamma,rho,p)

        elif int_energy != None or temp != None: #check if T or e specificied

            if temp != None:
                self.e = computeIntEnergy(temp) #update int erg to be consistent with temp
            else:
                self.e = int_energy
             
            self.p = getPressure(gamma,rho,self.e)

        else:
            raise IOError("You must specify pressure, energy, or temperature in"
                " HydroState constructor")

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

    #-------------------------------------------------------------------------------
    # Update internal energy based on temperature
    def computeIntEnergy(self,temp):

        return temp*(self.rho*self.spec_heat)

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

        return "u: %.4f rho: %.4f e: %.4f p: %.4f" % (self.u, self.rho, self.e, self.p)


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

