## @package unittests.testHydro
#  Contains unit test class for testing a pure hydrodynamics test problem.

import sys
sys.path.append('../src')

import numpy as np
import unittest

from mesh import Mesh
from crossXInterface import ConstantCrossSection
from hydroState import HydroState
from plotUtilities import plotHydroSolutions
from radiation import Radiation
from transient import runNonlinearTransient
from hydroBC import HydroBC

## Unit test class
#
class TestHydro(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_Hydro(self):

      # create mesh
      n_elems = 100
      width = 1.0
      mesh = Mesh(n_elems,width)
      x_diaphragm = 0.3

      # time step size and transient start and end times
      CFL     = 0.5
      t_start = 0.0
      t_end   = 0.2

      # constant properties
      sig_s = 1.0 # arbitrary
      sig_a = 0.0 # set to zero to ensure no emission
      sig_t = sig_s + sig_a
      c_v   = 3.0 # arbitrary
      gam = 1.4

      # hydro IC values for left half of domain
      rhoL = 1.0
      uL   = 0.75
      pL   = 1.0

      # hydro IC values for right half of domain
      rhoR = 0.125
      uR   = 0.0
      pR   = 0.1

      # compute left and right internal energies
      eL = pL/(rhoL*(gam - 1.0))
      eR = pR/(rhoR*(gam - 1.0))

      # construct cross sections and hydro IC
      cross_sects = list()
      hydro_IC = list()
      for i in range(mesh.n_elems):  

         # cross section is constant
         cross_sects.append( (ConstantCrossSection(sig_s, sig_t),
                              ConstantCrossSection(sig_s, sig_t)) )
      
         # IC for left half of domain
         if mesh.getElement(i).x_cent < x_diaphragm:
            hydro_IC.append(
               HydroState(u=uL,rho=rhoL,e=eL,gamma=gam,spec_heat=c_v) )

         # IC for right half of domain
         else:
            hydro_IC.append(
               HydroState(u=uR,rho=rhoR,e=eR,gamma=gam,spec_heat=c_v) )

      # create hydro BC
      hydro_BC = HydroBC(bc_type='reflective', mesh=mesh)
  
      # initialize radiation to zero solution to give pure hydrodynamics
      psi_left  = 0.0
      psi_right = 0.0
      rad_IC    = Radiation([0.0 for i in range(n_elems*4)])

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbose = True

      # run transient
      rad_new, hydro_new = runNonlinearTransient(
         mesh         = mesh,
         time_stepper = 'BE',
         problem_type = 'rad_hydro',
         dt_option    = 'CFL',
         CFL          = 0.5,
         use_2_cycles = True,
         t_start      = t_start,
         t_end        = t_end,
         psi_left     = psi_left,
         psi_right    = psi_right,
         cross_sects  = cross_sects,
         rad_IC       = rad_IC,
         hydro_IC     = hydro_IC,
         hydro_BC     = hydro_BC,
         verbose      = verbose)



      # plot solutions if run standalone
      if __name__ == "__main__":

         # get exact solutions
         #get the exact values
         f = open('exact_testHydro.txt', 'r')
         x_e = []
         u_e = []
         p_e = []
         rho_e = []
         e_e = []
         for line in f:
             if len(line.split())==1:
                 t = line.split()
             else:
                 data = line.split()
                 x_e.append(float(data[0]))
                 u_e.append(float(data[1]))
                 p_e.append(float(data[2]))
                 rho_e.append(float(data[4]))
                 e_e.append(float(data[3]))

         hydro_exact = []
         for i in range(len(x_e)):
            hydro_exact.append(HydroState(u=u_e[i],rho=rho_e[i],e=e_e[i],gamma=gam,spec_heat=c_v) )


         plotHydroSolutions(mesh, hydro_new,x_exact=x_e,exact=hydro_exact)

         # print out the states
         for i in hydro_new:

             print i

  
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

