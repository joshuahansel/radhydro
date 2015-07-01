## @package testRadHydroMMS
#  Contains unittest class to test an MMS problem with the full
#  radiation-hydrodynamics scheme

# add source directory to module search path
import sys
sys.path.append('../src')

# symbolic math packages
from sympy import symbols, exp, sin, pi
from sympy.utilities.lambdify import lambdify

# numpy
import numpy as np

# unit test package
import unittest

# local packages
from mesh import Mesh
from hydroState import HydroState
from radiation import Radiation
from plotUtilities import plotHydroSolutions, plotTemperatures
from utilityFunctions import computeRadiationVector, computeAnalyticHydroSolution
from crossXInterface import ConstantCrossSection
from transient import runNonlinearTransient
from hydroBC import HydroBC
import globalConstants as GC

## Derived unittest class to test the MMS source creator functions
#
class TestRadHydroShock(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_RadHydroShock(self):
      
      
      # numeric values
      cv_value    = 1.0
      gamma_value = 1.4
      
      # create uniform mesh
      n_elems = 100
      width = 0.02
      mesh = Mesh(n_elems, width)

      # compute radiation BC; assumes BC is independent of timea
      Erad_left = 1.372E-06
      Erad_right = 2.7955320762182542e-06
      c = GC.SPD_OF_LGT
      psi_left  = 0.5*c*Erad_left
      psi_right = 0.5*c*Erad_right
      
      # gamma constant
      gam = 1.4
 
      # material 1 properties and IC
      sig_a1 = 390.71164263502122
      sig_s1 = 8.5314410158161809E+2-sig_a1
      c_v1   = 1.2348000000000001e-01
      rho1   = 1.0
      E1     = 2.2226400000000000e-02
      u1     = 1.4055888445772469e-01
      e1     = E1/rho1 - 0.5*u1*u1
      T1     = 0.1

      # material 2 properties and IC
      sig_a2 = sig_a1
      sig_s2 = sig_s1
      rho2   = 1.2973213452231311
      c_v2   = c_v1
      E2     = 2.6753570531538713e-002
      u2     = 1.0834546504247138e-001
      e2     = E2/rho2 - 0.5*u2*u2
      T2     = 1.1947515210501813e-001

      # construct cross sections and hydro IC
      cross_sects = list()
      hydro_IC = list()
      psi_IC = list()

      for i in range(mesh.n_elems):  
      
         if mesh.getElement(i).x_cent < 0.5*width: # material 1
            cross_sects.append( (ConstantCrossSection(sig_s1, sig_s1+sig_a1),
                                 ConstantCrossSection(sig_s1, sig_s1+sig_a1)) )
            hydro_IC.append(
               HydroState(u=u1,rho=rho1,e=e1,spec_heat=c_v1,gamma=gam))

            psi_IC += [psi_left for dof in range(4)]

         else: # material 2
            cross_sects.append((ConstantCrossSection(sig_s2, sig_a2+sig_s2),
                                ConstantCrossSection(sig_s2, sig_a2+sig_s2)))
            hydro_IC.append(
               HydroState(u=u2,rho=rho2,e=e2,spec_heat=c_v2,gamma=gam))

            psi_IC += [psi_right for dof in range(4)]

         print hydro_IC[-1]
         print hydro_IC[-1].getTemperature(), T2

      rad_IC = Radiation(psi_IC)

      # create hydro BC
      hydro_BC = HydroBC(bc_type='reflective', mesh=mesh)
  
      # transient options
      t_start  = 0.0
      t_end = 0.5

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbose = True
      
      # run the rad-hydro transient
      rad_new, hydro_new = runNonlinearTransient(
         mesh         = mesh,
         problem_type = 'rad_hydro',
         dt_option    = 'CFL',
         time_stepper = 'BE',
         CFL          = 0.3,
         use_2_cycles = False,
         t_start      = t_start,
         t_end        = t_end,
         psi_left     = psi_left,
         psi_right    = psi_right,
         cross_sects  = cross_sects,
         rad_IC       = rad_IC,
         hydro_IC     = hydro_IC,
         hydro_BC     = hydro_BC,
         verbose      = verbose)

      # plot
      if __name__ == '__main__':

         # plot radiation solution

         # compute exact hydro solution
     #    hydro_exact = computeAnalyticHydroSolution(mesh, t=t_end,
     #       rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)
         hydro_exact = None

         # plot hydro solution
         plotHydroSolutions(mesh, hydro_new, exact=hydro_exact)

         plotTemperatures(mesh, rad_new.E, hydro_states=hydro_new, print_values=False)

# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

