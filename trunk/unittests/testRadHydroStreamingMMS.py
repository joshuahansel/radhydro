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

class TestRadHydroMMS(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_RadHydroMMS(self):
      
      # declare symbolic variables
      x, t, A, B, C, Cnondim, c, cv, gamma, mu, alpha = \
         symbols('x t A B C Cnondim c cv gamma mu alpha')
      
      # MMS solutions
      rho = A*(sin(B*x-C*t)+2)
      u   = 1/(A*(sin(B*x-C*t)+2))
      p   = A*alpha*(sin(B*x-C*t)+2)
      Er  = alpha*(sin(B*x-Cnondim*C*t)+2)
      Fr  = alpha*(sin(B*x-Cnondim*C*t)+2)
      #Er = 0.5*(sin(2*pi*x - 10.*t) + 2.)/c
      #Fr = 0.5*(sin(2*pi*x - 10.*t) + 2.)

      # derived solutions
      T = gamma*p/rho
      e = cv * T
      E = rho*(u*u/2 + e)
      psip = (Er*c + Fr/mu)/2
      psim = (Er*c - Fr/mu)/2

      # numeric values
      A_value = 1.0
      B_value = 1.0
      C_value = 1.0
      alpha_value = 1.0
      gamma_value = 5.0/3.0
      Cnondim_value = 10.0
      sig_s_value = 0.0
      sig_a_value = 1.0
      P_value = 0.001
      cv_value = 1.0

      # create list of substitutions
      substitutions = dict()
      substitutions['A']     = A_value
      substitutions['B']     = B_value
      substitutions['C']     = C_value
      substitutions['Cnondim'] = Cnondim_value
      substitutions['c']     = GC.SPD_OF_LGT
      substitutions['cv']    = cv_value
      substitutions['gamma'] = gamma_value
      substitutions['mu']    = RU.mu["+"]
      substitutions['alpha'] = alpha_value

      # make substitutions
      rho  = rho.subs(substitutions)
      u    = u.subs(substitutions)
      mom  = rho*u
      E    = E.subs(substitutions)
      psim = psim.subs(substitutions)
      psip = psip.subs(substitutions)

      # create MMS source functions
      rho_src, mom_src, E_src, psim_src, psip_src = createMMSSourceFunctionsRadHydro(
         rho           = rho,
         u             = u,
         E             = E,
         psim          = psim,
         psip          = psip,
         sigma_s_value = sig_s_value,
         sigma_a_value = sig_a_value,
         gamma_value   = gamma_value,
         cv_value      = cv_value,
         alpha_value   = alpha_value,
         display_equations = True)

      # create functions for exact solutions
      rho_f  = lambdify((symbols('x'),symbols('t')), rho,  "numpy")
      u_f    = lambdify((symbols('x'),symbols('t')), u,    "numpy")
      mom_f  = lambdify((symbols('x'),symbols('t')), mom,  "numpy")
      E_f    = lambdify((symbols('x'),symbols('t')), E,    "numpy")
      psim_f = lambdify((symbols('x'),symbols('t')), psim, "numpy")
      psip_f = lambdify((symbols('x'),symbols('t')), psip, "numpy")
      
      # create uniform mesh
      n_elems = 50
      width = 2.0*math.pi
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
      cross_sects = [(ConstantCrossSection(sig_s_value, sig_s_value+sig_a_value),
                      ConstantCrossSection(sig_s_value, sig_s_value+sig_a_value))
                      for i in xrange(mesh.n_elems)]

      # transient options
      t_start  = 0.0
      t_end = 0.1*math.pi

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbosity = 2
      else:
         verbosity = 0

      #slope limiter
      limiter = 'double-minmod'
      
      # run the rad-hydro transient
      rad_new, hydro_new = runNonlinearTransient(
         mesh         = mesh,
         problem_type = 'rad_hydro',
         dt_option    = 'CFL',
         CFL          = 0.5,
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
         rho_f        = rho_f,
         u_f          = u_f,
         E_f          = E_f,
         gamma_value  = gamma_value,
         cv_value     = cv_value,
         check_balance = False)

      # plot
      if __name__ == '__main__':

         # compute exact hydro solution
         hydro_exact = computeAnalyticHydroSolution(mesh, t=t_end,
            rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)

         # plot hydro solution
         plotHydroSolutions(\
            mesh, hydro_new, x_exact=mesh.getCellCenters(), exact=hydro_exact)

         # plot Er solution against exact Er
         Er_exact_fn = 1./GC.SPD_OF_LGT*(psim + psip)
         Fr_exact_fn = (psip - psim)*RU.mu["+"]
         Er_exact = []
         Fr_exact = []
         x = mesh.getCellCenters()
         for xi in x:
             substitutions = {'x':xi, 't':t_end}
             Er_exact.append(Er_exact_fn.subs(substitutions))
             Fr_exact.append(Fr_exact_fn.subs(substitutions))
         plotRadErg(mesh, rad_new.E, rad_new.F, exact_Er=Er_exact, exact_Fr =
               Fr_exact)

# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

