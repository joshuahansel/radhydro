## @package unittests.testHydroUniformIC
#  Contains unittest class to test a pure hydro problem with uniform
#  initial conditions.

# add source directory to module search path
import sys
sys.path.append('../src')

# symbolic math packages
from sympy import symbols, exp, sin, pi, sympify
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
from utilityFunctions import computeRadiationVector, computeAnalyticHydroSolution,\
   computeHydroError, computeConvergenceRates, printConvergenceTable
from crossXInterface import ConstantCrossSection
from transient import runNonlinearTransient
from hydroBC import HydroBC

## Class to test a pure hydro problem with uniform initial conditions.
#
#  With uniform initial conditions and boundary conditions, it is
#  impossible for any flux to be generated, so taking any number of
#  time steps should yield the initial solution.
#
class TestHydroUniformIC(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_HydroUniformIC(self):
      
      # declare symbolic variables
      x, t, alpha = symbols('x t alpha')
      
      # create solution for thermodynamic state and flow field
      rho = sympify('1')
      u   = sympify('1')
      E   = sympify('10')
      
      # create solution for radiation field
      psim = sympify('0')
      psip = sympify('0')
      
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
         display_equations = False)

      # create functions for exact solutions
      substitutions = dict()
      substitutions['alpha'] = alpha_value
      rho = rho.subs(substitutions)
      u   = u.subs(substitutions)
      mom = rho*u
      E   = E.subs(substitutions)
      rho_f  = lambdify((symbols('x'),symbols('t')), rho,  "numpy")
      u_f    = lambdify((symbols('x'),symbols('t')), u,    "numpy")
      mom_f  = lambdify((symbols('x'),symbols('t')), mom,  "numpy")
      E_f    = lambdify((symbols('x'),symbols('t')), E,    "numpy")
      psim_f = lambdify((symbols('x'),symbols('t')), psim, "numpy")
      psip_f = lambdify((symbols('x'),symbols('t')), psip, "numpy")
      
      # spatial and temporal domains
      width = 1.0
      t_start = 0.0
      t_end = 0.005
      dt_constant = 0.001

      # create mesh
      n_elems = 20
      mesh = Mesh(n_elems, width)

      # compute radiation BC; assumes BC is independent of time
      psi_left  = psip_f(x=0.0,   t=0.0)
      psi_right = psim_f(x=width, t=0.0)

      # compute radiation IC
      psi_IC = computeRadiationVector(psim_f, psip_f, mesh, t=0.0)
      rad_IC = Radiation(psi_IC)

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

      # slope limiter option
      slope_limiter = "vanleer"

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbosity = 2
      else:
         verbosity = 0
      
      # run the rad-hydro transient
      rad_new, hydro_new = runNonlinearTransient(
         mesh         = mesh,
         problem_type = 'rad_hydro',
         dt_option    = 'constant',
         dt_constant  = dt_constant,
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
         verbosity    = verbosity)

      # number of decimal places to check
      n_decimal_places = 15

      # check that states are all equal to the IC
      for i in xrange(n_elems):
         state    = hydro_new[i]
         state_IC = hydro_IC[i]
         rho, mom, erg = state.getConservativeVariables()
         rho0, mom0, erg0 = state_IC.getConservativeVariables()
         self.assertAlmostEqual(rho, rho0, n_decimal_places)
         self.assertAlmostEqual(mom, mom0, n_decimal_places)
         self.assertAlmostEqual(erg, erg0, n_decimal_places)


# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

