## @package unittests.testRadHydroMMS
#  Contains unittest class to test an MMS problem with the full
#  radiation-hydrodynamics scheme

# add source directory to module search path
import sys
sys.path.append('../src')

# symbolic math packages
from sympy import symbols, exp, sin, pi, sympify, cos, diff
from sympy.utilities.lambdify import lambdify

# numpy
import numpy as np
import math

# unit test package
import unittest

# local packages
from createMMSSourceFunctions import createMMSSourceFunctionsRadHydro
from mesh import Mesh
from hydroState import HydroState
from radiation import Radiation
from plotUtilities import plotHydroSolutions, plotTemperatures, plotRadErg
from utilityFunctions import computeRadiationVector, computeAnalyticHydroSolution
from crossXInterface import ConstantCrossSection
from transient import runNonlinearTransient
from hydroBC import HydroBC
from radBC import RadBC
import globalConstants as GC
import radUtilities as RU

## Derived unittest class to test the MMS source creator functions
##
class TestRadHydroMMS(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_RadHydroMMS(self):
      
      # declare symbolic variables
      x, t, alpha, c, a, mu  = symbols('x t alpha c a mu')
      
      # numeric values
      alpha_value = 0.01
      cv_value    = 1.0
      gamma_value = 1.4
      sig_s = 1.0
      sig_a = 1000.0
      sig_t = sig_s + sig_a

      # create solution for thermodynamic state and flow field
 #     rho = sympify('4.0')
 #     u   = sympify('1.0')
 #     E   = sympify('10.0')
      rho = 2. + sin(2*pi*x-t)
      u   = 2. + cos(2*pi*x+t)
      p   = 2. + cos(2*pi*x+t)
      e = p/(rho*(gamma_value-1.))
      E = 0.5*rho*u*u + rho*e
      
      # create solution for radiation field based on solution for F 
      # that is the leading order diffusion limit solution

      #Equilibrium diffusion solution
      T = e/cv_value
      Er = a*T**4
      Fr = -1./(3.*sig_t)*diff(Er*c,x) + 4./3.*Er*u

      #Form psi+ and psi- from Fr and Er
      psip = (Er*c*mu + Fr)/(2.*mu)
      psim = (Er*c*mu - Fr)/(2.*mu)

      # create functions for exact solutions
      substitutions = dict()
      substitutions['alpha'] = alpha_value
      substitutions['c']     = GC.SPD_OF_LGT
      substitutions['a']     = GC.RAD_CONSTANT
      substitutions['mu']    = RU.mu["+"]
      rho = rho.subs(substitutions)
      u   = u.subs(substitutions)
      mom = rho*u
      E   = E.subs(substitutions)
      psim = psim.subs(substitutions)
      psip = psip.subs(substitutions)
      rho_f  = lambdify((symbols('x'),symbols('t')), rho,  "numpy")
      u_f    = lambdify((symbols('x'),symbols('t')), u,    "numpy")
      mom_f  = lambdify((symbols('x'),symbols('t')), mom,  "numpy")
      E_f    = lambdify((symbols('x'),symbols('t')), E,    "numpy")
      psim_f = lambdify((symbols('x'),symbols('t')), psim, "numpy")
      psip_f = lambdify((symbols('x'),symbols('t')), psip, "numpy")

      # create MMS source functions
      rho_src, mom_src, E_src, psim_src, psip_src = createMMSSourceFunctionsRadHydro(
         rho           = rho,
         u             = u,
         E             = E,
         psim          = psim,
         psip          = psip,
         sigma_s_value = sig_s,
         sigma_a_value = sig_a,
         gamma_value   = gamma_value,
         cv_value      = cv_value,
         alpha_value   = alpha_value,
         display_equations = True)

      # create uniform mesh
      n_elems = 50
      width = 1.0
      mesh = Mesh(n_elems, width)

      # compute radiation IC
      psi_IC = computeRadiationVector(psim_f, psip_f, mesh, t=0.0)
      rad_IC = Radiation(psi_IC)

      # create rad BC object
      rad_BC = RadBC(mesh, 'periodic')

      # compute hydro IC
      hydro_IC = computeAnalyticHydroSolution(mesh,t=0.0,
         rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)

      # create hydro BC
      hydro_BC = HydroBC(bc_type='periodic', mesh=mesh)
  
      # create cross sections
      cross_sects = [(ConstantCrossSection(sig_s, sig_s+sig_a),
                      ConstantCrossSection(sig_s, sig_s+sig_a))
                      for i in xrange(mesh.n_elems)]

      # transient options
      t_start  = 0.0
      t_end = 0.000001*math.pi

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbosity = 2
      else:
         verbosity = 0

      #slope limiter
      limiter = 'vanleer'
      
      # run the rad-hydro transient
      rad_new, hydro_new = runNonlinearTransient(
         mesh         = mesh,
         problem_type = 'rad_hydro',
         dt_option    = 'CFL',
         CFL          = 0.5,
       #  dt_option    = 'constant',
       #  dt_constant  = 0.0002,
         slope_limiter = limiter,
         time_stepper = 'BDF2',
         use_2_cycles = True,
         t_start      = t_start,
         t_end        = t_end,
         rad_BC       = rad_BC,
         cross_sects  = cross_sects,
         rad_IC       = rad_IC,
         hydro_IC     = hydro_IC,
         hydro_BC     = hydro_BC,
         mom_src      = mom_src,
         E_src        = E_src,
         rho_src      = rho_src,
         psim_src     = psim_src,
         psip_src     = psip_src,
         verbosity    = verbosity,
         rho_f =rho_f,
         u_f = u_f,
         E_f = E_f,
         gamma_value = gamma_value,
         cv_value = cv_value,
         check_balance = True)

      # plot
      if __name__ == '__main__':

         # plot radiation solution

         # compute exact hydro solution
         hydro_exact = computeAnalyticHydroSolution(mesh, t=t_end,
            rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)

         # plot hydro solution
         plotHydroSolutions(mesh, hydro_new, x_exact=mesh.getCellCenters(),exact=hydro_exact)

         #plot exact and our E_r
         Er_exact_fn = 1./GC.SPD_OF_LGT*(psim + psip)
         Er_exact = []
         x = mesh.getCellCenters()
         for xi in x:
             
             substitutions = {'x':xi, 't':t_end}
             Er_exact.append(Er_exact_fn.subs(substitutions))

         plotRadErg(mesh, rad_new.E, exact_Er=Er_exact)

# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

