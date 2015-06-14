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
from createMMSSourceFunctions import createMMSSourceFunctionsRadHydro
from mesh import Mesh
from hydroState import HydroState
from radiation import Radiation
from plotUtilities import plotHydroSolutions
from utilityFunctions import computeRadiationVector

## Derived unittest class to test the MMS source creator functions
#
class TestRadHydroMMS(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_RadHydroMMS(self):
      
      # declare symbolic variables
      x, t, alpha = symbols('x t alpha')
      
      # create solution for thermodynamic state and flow field
      rho = exp(-1*alpha*t)*sin(pi*x) + 2
      u   = exp(-2*alpha*t)*sin(pi*x)
      E   = exp(-3*alpha*t)*sin(pi*x) + 3
      
      # create solution for radiation field
      psim = 2*t*sin(pi*(1-x))
      psip = t*sin(pi*x)
      
      # numeric values
      alpha_value = 0.01
      cv_value    = 1.0
      gamma_value = 1.4
      sigma_s_value = 1.0
      sigma_a_value = 1.0
      
      # create MMS source functions
      rho_src, mom_src, E_src, psim_src, psip_src = createMMSSourceFunctionsRadHydro(
         rho           = rho,
         u             = u,
         E             = E,
         psim          = psim,
         psip          = psip,
         sigma_s_value = sigma_s_value,
         sigma_a_value = sigma_a_value,
         gamma_value   = gamma_value,
         cv_value      = cv_value,
         alpha_value   = alpha_value,
         display_equations = False)

      # create functions for exact solutions
      substitutions = dict()
      substitutions['alpha'] = alpha_value
      rho = rho.subs(substitutions)
      u   = u.subs(substitutions)
      E   = E.subs(substitutions)
      rho_f  = lambdify((symbols('x'),symbols('t')), rho,  "numpy")
      u_f    = lambdify((symbols('x'),symbols('t')), u,    "numpy")
      E_f    = lambdify((symbols('x'),symbols('t')), E,    "numpy")
      psim_f = lambdify((symbols('x'),symbols('t')), psim, "numpy")
      psip_f = lambdify((symbols('x'),symbols('t')), psip, "numpy")
      
      # create uniform mesh
      n_elems = 50
      width = 1.0
      mesh = Mesh(n_elems, width)

      # compute radiation IC
      psi_IC = computeRadiationVector(psim_f, psip_f, mesh, t=0.0)
      rad_IC = Radiation(psi_IC)

      # compute radiation BC; assumes BC is independent of time
      psi_left  = psip_f(x=0.0,   t=0.0)
      psi_right = psim_f(x=width, t=0.0)

      # compute hydro IC
      hydro_IC = list()
      for i in xrange(n_elems):

         # get cell center
         x_i = mesh.getElement(i).x_cent

         # evaluate functions at cell center
         rho_i = rho_f(x=x_i, t=0.0)
         u_i   =   u_f(x=x_i, t=0.0)
         E_i   =   E_f(x=x_i, t=0.0)
         e_i = E_i / rho_i - 0.5*u_i**2

         # add hydro state for cell
         hydro_IC.append(HydroState(rho=rho_i, u=u_i, int_energy=e_i,
            spec_heat=cv_value, gamma=gamma_value))

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbose = True
      
      # run the rad-hydro transient
#      rad_new, hydro_new = runNonlinearTransient(
#         mesh         = mesh,
#         time_stepper = 'BE',
#         problem_type = 'rad_hydro',
#         dt_option    = 'CFL',
#         CFL          = 0.5,
#         use_2_cycles = True,
#         t_start      = t_start,
#         t_end        = t_end,
#         psi_left     = psi_left,
#         psi_right    = psi_right,
#         cross_sects  = cross_sects,
#         rad_IC       = rad_IC,
#         hydro_IC     = hydro_IC,
#         rho_src      = rho_src,
#         mom_src      = mom_src,
#         E_src        = E_src,
#         psim_src     = psim_src,
#         psip_src     = psip_src,
#         verbose      = verbose)

      # plot
      if __name__ == '__main__':

         # plot radiation solution

         # plot hydro solution
         plotHydroSolutions(mesh.getCellCenters(), hydro_IC)


# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

