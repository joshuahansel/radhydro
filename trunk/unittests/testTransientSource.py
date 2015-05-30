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
from crossXInterface import CrossXInterface
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
   def test_CNTransientSource(self):
      # number of decimal places to test
      n_decimal_places = 14

      # create mesh
      n_elems = 5
      mesh = Mesh(n_elems,random())

      # compute uniform cross sections
      cross_sects = [(CrossXInterface(random(), random()),
                      CrossXInterface(random(), random()))
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
      psim_ss, psip_ss, E_ss, F_ss = radiationSolveSS(mesh, cross_sects, Q,
              bc_psi_right = psi_right, bc_psi_left = psi_left)
  
      # time-stepper
      time_stepper = "CN"
  
      # compute the transient source
      transient_source = TransientSource(mesh, time_stepper)
      Q_tr = transient_source.evaluate(
         dt            = dt,
         psim_old      = psim_ss,
         psip_old      = psip_ss,
         bc_flux_left  = psi_left,
         bc_flux_right = psi_right,
         cx_old        = cross_sects,
         E_old         = E_ss,
         Q_old         = Q,
         Q_new         = Q)

      # loop over elements and test that sources are what they should be
      for i in xrange(mesh.n_elems):
         iLminus = getIndex(i,"L","-")
         iLplus  = getIndex(i,"L","+")
         iRminus = getIndex(i,"R","-")
         iRplus  = getIndex(i,"R","+")
         QiLminus_expected = psim_ss[i][0]/c_dt + 0.5*Q[iLminus]
         QiLplus_expected  = psip_ss[i][0]/c_dt + 0.5*Q[iLplus]
         QiRminus_expected = psim_ss[i][1]/c_dt + 0.5*Q[iRminus]
         QiRplus_expected  = psip_ss[i][1]/c_dt + 0.5*Q[iRplus]
         self.assertAlmostEqual(Q_tr[iLminus],QiLminus_expected,n_decimal_places)
         self.assertAlmostEqual(Q_tr[iLplus], QiLplus_expected, n_decimal_places)
         self.assertAlmostEqual(Q_tr[iRminus],QiRminus_expected,n_decimal_places)
         self.assertAlmostEqual(Q_tr[iRplus], QiRplus_expected, n_decimal_places)
  
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()
