## @package testTRTOnly
#  Contains unit test for a 2-material TRT problem, using BE time discretization.

import sys
sys.path.append('../src')

import numpy as np
import unittest

from mesh import Mesh
from crossXInterface import ConstantCrossSection
from hydroState import HydroState
from TRTUtilities import convSpecHeatErgsEvToJksKev, \
                         computeEquivIntensity, \
                         computeRadTemp
from plotUtilities import plotTemperatures
from radiation import Radiation
from transient import runNonlinearTransient

## Unit test class for a 2-material TRT problem, using BE time discretization.
#
class TestTRTOnly(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_TRTBE(self):

      # create mesh
      n_elems = 200
      mesh = Mesh(n_elems,1.)

      # time step size and transient start and end times
      dt      = 0.001
      t_start = 0.
      t_end   = 0.04

      # initialize temperature
      T_init = 0.05
      T_l    = 0.5
      T_r    = 0.05

      # gamma constant
      gam = 1.4
 
      # material 1 properties and IC
      sig_s1 = 0.0
      sig_a1 = 0.2
      c_v1   = convSpecHeatErgsEvToJksKev(1.E+12)
      rho1   = 0.01
      e1     = T_init*c_v1

      # material 2 properties and IC
      sig_s2 = 0.0
      sig_a2 = 2000.
      c_v2   = c_v1
      rho2   = 10.
      e2     = T_init*c_v2

      # construct cross sections and hydro IC
      cross_sects = list()
      hydro_IC = list()
      for i in range(mesh.n_elems):  
      
         if mesh.getElement(i).x_cent < 0.5: # material 1
            cross_sects.append( (ConstantCrossSection(sig_s1, sig_s1+sig_a1),
                                 ConstantCrossSection(sig_s1, sig_s1+sig_a1)) )
            hydro_IC.append( (
               HydroState(u=0,rho=rho1,int_energy=e1,gamma=gam,spec_heat=c_v1),
               HydroState(u=0,rho=rho1,int_energy=e1,gamma=gam,spec_heat=c_v1)) )
         else: # material 2
            cross_sects.append((ConstantCrossSection(sig_s2, sig_a2+sig_s2),
                                ConstantCrossSection(sig_s2, sig_a2+sig_s2)))
            hydro_IC.append( (
               HydroState(u=0,rho=rho2,int_energy=e2,spec_heat=c_v2, gamma=gam), 
               HydroState(u=0,rho=rho2,spec_heat=c_v2,int_energy=e2, gamma=gam)) )
  
      # initialize radiation to equilibrium solution
      psi_left  = computeEquivIntensity(T_l)
      psi_right = computeEquivIntensity(T_r)
      rad_IC    = Radiation([psi_right for i in range(n_elems*4)])

      # time-stepper
      time_stepper = "BE"

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbose = True

      # run transient
      rad_new, hydro_new = runNonlinearTransient(
         mesh         = mesh,
         time_stepper = time_stepper,
         problem_type = 'rad_mat',
         dt_option    = 'constant',
         dt_constant  = dt,
         t_start      = t_start,
         t_end        = t_end,
         psi_left     = psi_left,
         psi_right    = psi_right,
         cross_sects  = cross_sects,
         rad_IC       = rad_IC,
         hydro_IC     = hydro_IC,
         verbose      = verbose)

      # plot solutions if run standalone
      if __name__ == "__main__":
          plotTemperatures(mesh, rad_new.E, hydro_states=hydro_new, print_values=True)

  
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

