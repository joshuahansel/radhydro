## @package hydroSlopes
#  Contains class for computing and storing hydro slopes.

import numpy as np
from math import isinf

## Class for computing and storing hydro slopes
#
class HydroSlopes:

    ## Constructor
    #
    def __init__(self, states):

       # extract vectors of conservative variables
       rho = [s.rho                     for s in states]
       mom = [s.rho*s.u                 for s in states]
       erg = [s.rho*(0.5*s.u*s.u + s.e) for s in states]

       # compute slopes
       self.rho_slopes = self.slopeReconstruction(rho)
       self.mom_slopes = self.slopeReconstruction(mom)
       self.erg_slopes = self.slopeReconstruction(erg)

    ## Extracts slopes for each conservative variable
    #
    def extractSlopes(self):

       return self.rho_slopes, self.mom_slopes, self.erg_slopes

    ## Reconstructs slopes for a single conservative variable
    #
    def slopeReconstruction(self,u):
    
        limiter = "vanleer"
    
        omega = 0.
    
        # compute slopes
        u_slopes = np.zeros(len(u))
        for i in range(len(u)):
            
            if i==0:
                del_i = u[i+1]-u[i]
            elif i==len(u)-1:
                del_i = u[i]-u[i-1]
            else:
                del_rt = u[i+1]-u[i]
                del_lt = u[i] - u[i-1]
                del_i = 0.5*(1.+omega)*del_lt + 0.5*(1.-omega)*del_rt
    
                #Simple slope limiters
                if limiter == "minmod":
    
                    del_i = minMod(del_rt,del_lt)
    
                elif limiter == "vanleer":
    
                    beta = 1.
    
                    #Catch if divide by zero
                    if abs(u[i+1] - u[i]) < 0.000000001*(u[i]+0.00001):
                        if abs(u[i]-u[i-1]) < 0.000000001*(u[i]+0.00001):
                            r = 1.0
                        else:
                            r = -1.
                    else:
                        r = del_lt/del_rt
                            
                    if r < 0.0 or isinf(r):
                        del_i = 0.0
                    else:
    
                        zeta =  min(2.*r/(1.+r),2.*beta/(1.-omega+(1.+omega)*r))
                        del_i = zeta*del_i
    
                elif limiter == "none":
                    del_i = del_i
    
                elif limiter == "step":
                    del_i = 0.0
    
                else:
                    raise ValueError("Invalid slope limiter\n")
    
            # save slope
            u_slopes[i] = del_i
                
        return u_slopes

