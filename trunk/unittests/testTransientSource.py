## @package testTransientSource
#  Tests the transient source builder with Crank-Nicolson.
#
#  The sources are built using the steady-state solution as
#  the old solution. Considering that the steady-state
#  residual in the transient source should be zero, the transient
#  source should reduce to
#  \f[
#    \mathcal{Q}_{i,L}^{\pm,k} = \frac{\Psi_{i,L}^{\pm,n}}{c\Delta t}
#    + \frac{1}{2}Q_{i,L}^{\pm,k}
#  \f]
#  and similarly for \f$\mathcal{Q}_{i,R}^{\pm,k}\f$.
#  This equality is tested to 14 decimal places.
#
import sys
sys.path.append('../src')

from random import random
import numpy as np
import unittest

from mesh import Mesh
from crossXInterface import ConstantCrossSection
from radiationSolveSS import radiationSolveSS
from transientSource import * 
import globalConstants as GC

## Derived unittest class to test source builder
#
class TestTransientSource(unittest.TestCase):

   def setUp(self):
      pass

   def tearDown(self):
      pass

   def test_TransientSourceCN(self):

      # number of decimal places to test
      n_decimal_places = 13

      # create mesh
      n_elems = 5
      mesh = Mesh(n_elems,random())

      # compute uniform cross sections
      cross_sects = [(ConstantCrossSection(random(), random()),
                      ConstantCrossSection(random(), random()))
                      for i in xrange(mesh.n_elems)]
  
      # time step size and c*dt
      dt = random()
      c_dt = GC.SPD_OF_LGT*dt
  
      # boundary fluxes
      psi_left  = random()
      psi_right = random()
  
      # create the steady-state source
      n = 4*mesh.n_elems
      Q = np.zeros(n)
      for i in xrange(mesh.n_elems):
         Q[getIndex(i,"L","+")] = random()
         Q[getIndex(i,"R","+")] = random()
         Q[getIndex(i,"L","-")] = random()
         Q[getIndex(i,"R","-")] = random()
  
      # compute the steady-state solution
      rad_ss = radiationSolveSS(mesh, cross_sects, Q,
         bc_psi_right = psi_right, bc_psi_left = psi_left)
  
      # time-stepper
      time_stepper = "CN"
  
      # compute the transient source
      transient_source = TransientSource(mesh, time_stepper, src_term=True)
      Q_tr = transient_source.evaluate(
         dt            = dt,
         bc_flux_left  = psi_left,
         bc_flux_right = psi_right,
         cx_old        = cross_sects,
         rad_old       = rad_ss,
         Q_old         = Q,
         Q_new         = Q)

      # loop over elements and test that sources are what they should be
      for i in xrange(mesh.n_elems):
         iLm = getIndex(i,"L","-")
         iLp = getIndex(i,"L","+")
         iRm = getIndex(i,"R","-")
         iRp = getIndex(i,"R","+")
         QiLm_expected = rad_ss.psim[i][0]/c_dt + 0.5*Q[iLm]
         QiLp_expected = rad_ss.psip[i][0]/c_dt + 0.5*Q[iLp]
         QiRm_expected = rad_ss.psim[i][1]/c_dt + 0.5*Q[iRm]
         QiRp_expected = rad_ss.psip[i][1]/c_dt + 0.5*Q[iRp]
         self.assertAlmostEqual(Q_tr[iLm], QiLm_expected, n_decimal_places)
         self.assertAlmostEqual(Q_tr[iLp], QiLp_expected, n_decimal_places)
         self.assertAlmostEqual(Q_tr[iRm], QiRm_expected, n_decimal_places)
         self.assertAlmostEqual(Q_tr[iRp], QiRp_expected, n_decimal_places)
  
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()
