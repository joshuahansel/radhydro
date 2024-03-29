## @package unittests.testHydroMMS
#  Contains unittest class to test an MMS problem with only hydrodynamics

# add source directory to module search path
import sys
sys.path.append('../src')

# symbolic math packages
from sympy import symbols, exp, sin, pi, sympify, cos
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
   computeHydroL2Error, computeHydroConvergenceRates, printHydroConvergenceTable
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
      
      # number of elements in first cycle
      n_elems = 25

      # number of refinement cycles
      n_cycles = 4

      # spatial and temporal domains
      width = 1.0
      t_start  = 0.0
      t_end = 0.02
      
      # declare symbolic variables
      x, t, alpha = symbols('x t alpha')
      
      # physics parameters (floats, not sympy)
      alpha_value = 0.01
      cv_value    = 1.0
      gamma_value = 1.4
      sig_s = 0.0
      sig_a = 0.0

      # create solution for thermodynamic state and flow field
#      rho = sympify('4')
#      u   = sympify('1.3')
#      E   = sympify('10')
      rho = 2. + sin(2*pi*x)
      u   = 2. + cos(2*pi*x) 
      p   = 2. + cos(2*pi*x) 
      e = p/(rho*(gamma_value-1.))
      E = 0.5*rho*u*u + rho*e

      # create solution for radiation field
      psim = sympify('0')
      psip = sympify('0')
      
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
      rho_f  = lambdify((symbols('x'),symbols('t')), rho,  "numpy")
      u_f    = lambdify((symbols('x'),symbols('t')), u,    "numpy")
      mom_f  = lambdify((symbols('x'),symbols('t')), mom,  "numpy")
      E_f    = lambdify((symbols('x'),symbols('t')), E,    "numpy")
      psim_f = lambdify((symbols('x'),symbols('t')), psim, "numpy")
      psip_f = lambdify((symbols('x'),symbols('t')), psip, "numpy")
      
      # compute radiation BC; assumes BC is independent of time
      psi_left  = psip_f(x=0.0,   t=0.0)
      psi_right = psim_f(x=width, t=0.0)

      # initialize lists for mesh size and L1 error for each cycle
      max_dx = list()
      err = list()

      # loop over refinement cycles
      for cycle in xrange(n_cycles):

         if __name__ == '__main__':
            print("\nCycle %d of %d: n_elems = %d" % (cycle+1,n_cycles,n_elems))

         # create uniform mesh
         mesh = Mesh(n_elems, width)

         # append max dx for this cycle to list
         max_dx.append(mesh.max_dx)
   
         # compute radiation IC
         psi_IC = computeRadiationVector(psim_f, psip_f, mesh, t=0.0)
         rad_IC = Radiation(psi_IC)
   
         # compute hydro IC
         hydro_IC = computeAnalyticHydroSolution(mesh, t=0.0,
            rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)
   
         # create hydro BC
         hydro_BC = HydroBC(bc_type='periodic', mesh=mesh)

         # create cross sections
         cross_sects = [(ConstantCrossSection(sig_s, sig_s+sig_a),
                         ConstantCrossSection(sig_s, sig_s+sig_a))
                         for i in xrange(mesh.n_elems)]
   
         # slope limiter option
         slope_limiter = "none"
   
         # if run standalone, then be verbose
         if __name__ == '__main__':
            if n_cycles == 1:
               verbosity = 2
            else:
               verbosity = 1
         else:
            verbosity = 0

         # run the rad-hydro transient
         rad_new, hydro_new = runNonlinearTransient(
            mesh         = mesh,
            problem_type = 'rad_hydro',
            dt_option    = 'CFL',
            CFL          = 0.5,
            slope_limiter = slope_limiter,
            time_stepper = 'BDF2',
            use_2_cycles = True,
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
            rho_src      = rho_src,
            verbosity    = verbosity,
            check_balance = True,
            rho_f =rho_f,
            u_f = u_f,
            E_f = E_f,
            gamma_value = gamma_value,
            cv_value = cv_value  )
   
         # compute exact hydro solution
         hydro_exact = computeAnalyticHydroSolution(mesh, t=t_end,
            rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)
   
         # compute error
         err.append(computeHydroL2Error(hydro_new, hydro_exact))

         # double number of elements for next cycle
         n_elems *= 2

      # compute convergence rates
      rates = computeHydroConvergenceRates(max_dx,err)

      # print convergence table and plot
      if __name__ == '__main__':

         # print convergence table
         if n_cycles > 1:
            printHydroConvergenceTable(max_dx,err,rates=rates,
               dx_desc='dx',err_desc='L2')

         # plot hydro solution
         plotHydroSolutions(mesh, hydro_new, x_exact=mesh.getCellCenters(),
            exact=hydro_exact)


# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

