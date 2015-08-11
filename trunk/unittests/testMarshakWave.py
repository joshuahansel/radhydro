## @package unittests.testMarshakWave
#  Contains unit test for a 2-material TRT problem, using BE time discretization.

import sys
sys.path.append('../src')

import numpy as np
import unittest

from mesh import Mesh
from crossXInterface import InvCubedCrossX
from hydroState import HydroState
from TRTUtilities import convSpecHeatErgsEvToJksKev, \
                         computeEquivIntensity, \
                         computeRadTemp
from plotUtilities import plotTemperatures
from radiation import Radiation
from transient import runNonlinearTransient
from hydroBC import HydroBC

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
      mesh = Mesh(n_elems,2.)

      # time step size and transient start and end times
      dt      = 0.001
      t_start = 0.
      t_end   = 5.0

      # initialize temperature
      T_init = 2.5E-05
      T_l    = 0.150
      T_r    = T_init

      # gamma constant
      gam = 1.4
 
      # material 1 properties and IC
      sig_s1 = 0.0
      sig_a1 = 0.001
      c_v1   = convSpecHeatErgsEvToJksKev(1.3784E+11)
      rho1   = 1.0
      e1     = T_init*c_v1

      # construct cross sections and hydro IC
      cross_sects = list()
      hydro_IC = list()
      for i in range(mesh.n_elems):  
      
            hydro_IC.append(
               HydroState(u=0,rho=rho1,e=e1,spec_heat=c_v1,gamma=gam))
            cross_sects.append( (InvCubedCrossX(sig_s1,
                hydro_IC[-1],scale_coeff=0.001),InvCubedCrossX(sig_s1,
                hydro_IC[-1],scale_coeff=0.001)) )
  
      # create hydro BC
      hydro_BC = HydroBC(bc_type='reflective', mesh=mesh)
  
      # initialize radiation to equilibrium solution
      psi_left  = computeEquivIntensity(T_l)
      psi_right = computeEquivIntensity(T_r)
      rad_IC    = Radiation([psi_right for i in range(n_elems*4)])

      # time-stepper
      time_stepper = "CN"

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbosity = 2
      else:
         verbosity = 0

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
         hydro_BC     = hydro_BC,
         verbosity    = verbosity,
         check_balance= True)

      # plot solutions if run standalone
      if __name__ == "__main__":
          plotTemperatures(mesh, rad_new.E, hydro_states=hydro_new, print_values=True)

  
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

