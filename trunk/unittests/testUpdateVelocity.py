## @package unittests.testUpdateVelocity
#  Contains unittest class to test the velocity update function.

# add source directory to module search path
import sys
sys.path.append('../src')

# numpy
import numpy as np

# unit test package
import unittest

from random import random

# local packages
from mesh import Mesh
from hydroState import HydroState
from radiation import Radiation
from crossXInterface import ConstantCrossSection
from hydroSource import updateVelocity
import globalConstants as GC

## Class to test the velocity update function.
#
class TestUpdateVelocity(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_UpdateVelocity(self):
      
      n_elems = 20
      width = 1.0
      mesh = Mesh(n_elems, width)
      dt = 0.001
      time_stepper = 'BDF2'

      # create randomized cross sections
      def createRandomCrossSections(n):
         cx = list()
         for i in xrange(n):
            sigsL = random()
            sigaL = random()
            sigsR = random()
            sigaR = random()
            cx.append((ConstantCrossSection(sigsL, sigsL+sigaL),
                       ConstantCrossSection(sigsR, sigsR+sigaR)))
         return cx
      cx_older = createRandomCrossSections(n_elems)
      cx_old   = createRandomCrossSections(n_elems)
      cx_prev  = createRandomCrossSections(n_elems)

      # create random radiation
      def createRandomRadiation(n):
         rad_vector = np.random.random(n*4)
         rad = Radiation(rad_vector)
         return rad
      rad_older = createRandomRadiation(n_elems)
      rad_old   = createRandomRadiation(n_elems)
      rad_prev  = createRandomRadiation(n_elems)

      # create random hydro
      spec_heat = 1.0
      gamma = 1.4
      def createRandomHydro(n):
         hydro = list()
         for i in xrange(n):
            rho = random()
            u   = random()
            p   = random()
            hydro.append(HydroState(rho=rho,u=u,gamma=gamma,\
               spec_heat=spec_heat,p=p))
         return hydro
      hydro_older = createRandomHydro(n_elems)
      hydro_old   = createRandomHydro(n_elems)
      hydro_star  = createRandomHydro(n_elems)
      hydro_prev  = createRandomHydro(n_elems)
      hydro_new   = createRandomHydro(n_elems)

      # create random extraneous sources
      Qmom_older = np.random.random(n_elems)
      Qmom_old   = np.random.random(n_elems)
      Qmom_new   = np.random.random(n_elems)

      # perform velocity update
      updateVelocity(
         mesh         = mesh,
         time_stepper = time_stepper,
         dt           = dt,
         hydro_star   = hydro_star,
         hydro_new    = hydro_new,
         cx_older     = cx_older,
         cx_old       = cx_old,
         cx_prev      = cx_prev,
         rad_older    = rad_older,
         rad_old      = rad_old,
         rad_prev     = rad_prev,
         hydro_older  = hydro_older,
         hydro_old    = hydro_old,
         hydro_prev   = hydro_prev,
         Qmom_new     = Qmom_new,
         Qmom_old     = Qmom_old,
         Qmom_older   = Qmom_older)

      # number of decimal places to check
      n_decimal_places = 15

      # check that states are all equal to the IC
      c = GC.SPD_OF_LGT
      for i in xrange(n_elems):

         sigt_older = 0.5*(cx_older[i][0].sig_t + cx_older[i][1].sig_t)
         sigt_old   = 0.5*(cx_old[i][0].sig_t   + cx_old[i][1].sig_t)
         sigt_prev  = 0.5*(cx_prev[i][0].sig_t  + cx_prev[i][1].sig_t)

         Fr_older = 0.5*(rad_older.F[i][0] + rad_older.F[i][1])
         Fr_old   = 0.5*(rad_old.F[i][0]   + rad_old.F[i][1])
         Fr_prev  = 0.5*(rad_prev.F[i][0]  + rad_prev.F[i][1])

         Er_older = 0.5*(rad_older.E[i][0] + rad_older.E[i][1])
         Er_old   = 0.5*(rad_old.E[i][0]   + rad_old.E[i][1])
         Er_prev  = 0.5*(rad_prev.E[i][0]  + rad_prev.E[i][1])

         u_older = hydro_older[i].u
         u_old   = hydro_old[i].u
         u_prev  = hydro_prev[i].u

         older_src = sigt_older/c*(Fr_older - 4.0/3.0*Er_older*u_older) \
            + Qmom_older[i]
         old_src   = sigt_old  /c*(Fr_old   - 4.0/3.0*Er_old  *u_old) \
            + Qmom_old[i]
         prev_src  = sigt_prev /c*(Fr_prev  - 4.0/3.0*Er_prev *u_prev) \
            + Qmom_new[i]

         state_star = hydro_star[i]
         rho    = state_star.rho
         u_star = state_star.u

         u_new = u_star + dt/rho*(1.0/6.0*older_src + 1.0/6.0*old_src \
            + 2.0/3.0*prev_src)

         self.assertAlmostEqual(hydro_new[i].u, u_new, n_decimal_places)


# run main function from unittest module
if __name__ == '__main__':
   unittest.main()


