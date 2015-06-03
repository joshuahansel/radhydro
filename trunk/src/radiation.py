## @package src.radiation
#  Provides Radiation class
#

from math import sqrt
from copy import deepcopy

from radUtilities import mu
from globalConstants import SPD_OF_LGT as c

## Stores and handles radiation quantities
#
#  Angular fluxes \f$\Psi^\pm\f$, scalar flux \f$\phi\f$,
#  radiation energy \f$\mathcal{E}\f$, and radiation flux \f$\mathcal{F}\f$
#  are computed from a solution vector.
#
class Radiation(object):

   ## Constructor
   #
   #  @param[in] psi  S-2 solution vector
   #
   def __init__(self, psi):

      # update quantities
      self.update(psi)

   ## Updates all radiation quantities
   #
   #  Updates angular fluxes \f$\Psi^\pm\f$, scalar flux \f$\phi\f$,
   #  radiation energy \f$\mathcal{E}\f$, and radiation flux \f$\mathcal{F}\f$
   #
   #  @param[in] psi  S-2 solution vector
   #
   def update(self, psi):

      # copy solution vector
      self.psi = deepcopy(psi)

      # get number of dofs
      self.n_dofs = len(psi)

      # assert that number of dofs is multiple of 4
      if self.n_dofs % 4 != 0:
         raise ValueError('input vector must have a length\
            that is a multiple of 4')

      # compute number of elements
      self.n_elems = self.n_dofs / 4

      # update angular fluxes
      self.psim = [(psi[4*i],  psi[4*i+2]) for i in xrange(self.n_elems)]
      self.psip = [(psi[4*i+1],psi[4*i+3]) for i in xrange(self.n_elems)]

      # update scalar flux
      self.phi = [(self.psim[i][0]+self.psip[i][0],
                   self.psim[i][1]+self.psip[i][1])
                   for i in xrange(self.n_elems)]

      # update radiation energy
      self.E = [(self.phi[i][0]/c, self.phi[i][1]/c) for i in xrange(self.n_elems)]

      # update radiation flux
      self.F = [((self.psip[i][0]-self.psim[i][0])/sqrt(3.0),
                 (self.psip[i][1]-self.psim[i][1])/sqrt(3.0))
                 for i in xrange(self.n_elems)]

