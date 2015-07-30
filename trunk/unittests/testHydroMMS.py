## @package testHydroMMS
#  Contains unittest class to test an MMS problem with only hydrodynamics

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
from createMMSSourceFunctions import createMMSSourceFunctionsHydroOnly
from mesh import Mesh
from hydroState import HydroState
from radiation import Radiation
from plotUtilities import plotHydroSolutions, plotAngularFlux
from utilityFunctions import computeRadiationVector, computeAnalyticHydroSolution
from crossXInterface import ConstantCrossSection
from transient import runNonlinearTransient
from hydroBC import HydroBC

## Derived unittest class
#
class TestHydroMMS(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_HydroMMS(self):
      
      # declare symbolic variables
      x, t, alpha = symbols('x t alpha')
      
      # create solution for thermodynamic state and flow field
      rho = symbols('1')
      u   = symbols('1')
      E   = symbols('10')
      #rho = 1+x-t
      #u   = symbols('1')
      #E   = 5 + 5*(x-0.5)**2
      #rho = exp(x+t)
      #u   = exp(-x)*sin(t) - 1
      #E   = symbols('10')
      
      # create solution for radiation field
      psim = 0
      psip = 0
      
      # numeric values
      alpha_value = 0.01
      cv_value    = 1.0
      gamma_value = 1.4
      sig_s = 0.0
      sig_a = 0.0
      
      # create MMS source functions
      rho_src, mom_src, E_src, psim_src, psip_src = createMMSSourceFunctionsHydroOnly(
         rho           = rho,
         u             = u,
         E             = E,
         gamma_value   = gamma_value,
         cv_value      = cv_value,
         alpha_value   = alpha_value,
         display_equations = True)

      # create functions for exact solutions
      substitutions = dict()
      substitutions['alpha'] = alpha_value
      rho = rho.subs(substitutions)
      u   = u.subs(substitutions)
      mom = rho*u
      E   = E.subs(substitutions)
      rho_f  = lambdify((symbols('x'),symbols('t')), rho,  "numpy", dummify=False)
      u_f    = lambdify((symbols('x'),symbols('t')), u,    "numpy", dummify=False)
      mom_f  = lambdify((symbols('x'),symbols('t')), mom,  "numpy", dummify=False)
      E_f    = lambdify((symbols('x'),symbols('t')), E,    "numpy", dummify=False)
      psim_f = lambdify((symbols('x'),symbols('t')), psim, "numpy", dummify=False)
      psip_f = lambdify((symbols('x'),symbols('t')), psip, "numpy", dummify=False)
      
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
      hydro_IC = computeAnalyticHydroSolution(mesh, t=0.0,
         rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)

      # create hydro BC
      hydro_BC = HydroBC(bc_type='dirichlet', mesh=mesh, rho_BC=rho_f,
         mom_BC=mom_f, erg_BC=E_f)
  
      # create cross sections
      cross_sects = [(ConstantCrossSection(sig_s, sig_s+sig_a),
                      ConstantCrossSection(sig_s, sig_s+sig_a))
                      for i in xrange(mesh.n_elems)]

      # transient options
      t_start  = 0.0
      t_end = 0.1
      dt_constant = 0.05

      # slope limiter option
      slope_limiter = "minmod"

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbose = True
      
      # run the rad-hydro transient
      rad_new, hydro_new = runNonlinearTransient(
         mesh         = mesh,
         problem_type = 'rad_hydro',
         dt_option    = 'CFL',
    #     dt_option    = 'constant',
         CFL          = 0.5,
    #     dt_constant  = dt_constant,
         slope_limiter = slope_limiter,
         use_2_cycles = False,
         t_start      = t_start,
         t_end        = t_end,
         psi_left     = psi_left,
         psi_right    = psi_right,
         hydro_BC     = hydro_BC,
         cross_sects  = cross_sects,
         rad_IC       = rad_IC,
         hydro_IC     = hydro_IC,
         mom_src      = mom_src,
         E_src        = E_src,
         psim_src     = psim_src,
         psip_src     = psip_src,
         verbose      = verbose)

      # plot
      if __name__ == '__main__':

         # compute exact hydro solution
         hydro_exact = computeAnalyticHydroSolution(mesh, t=t_end,
            rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)

         # plot hydro solution
         plotHydroSolutions(mesh, hydro_new, x_exact=mesh.getCellCenters(), exact=hydro_exact)


# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

