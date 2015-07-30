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
from plotUtilities import plotHydroSolutions, plotTemperatures, plotRadErg
from utilityFunctions import computeRadiationVector, computeAnalyticHydroSolution
from crossXInterface import ConstantCrossSection
from transient import runNonlinearTransient
from hydroBC import HydroBC
import globalConstants as GC

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
      rho =exp(x+t)*0.0+5
      u   =(exp(-x)*sin(t) - 1)*0.0+3
      E   = 0.0*exp(-2*alpha*t)*sin(pi*x)+50
      
      # create solution for radiation field
      c = GC.SPD_OF_LGT
      rad_scale = 1
      #psim = rad_scale*2*t*sin(pi*(1-x))*0.+50*c
      #psip = rad_scale*t*sin(pi*x)*0.+20*c
      psim = 50.*c + 0.0*(t*x)
      psip = 50.*c + 0.0*(t*x)
      
      # numeric values
      alpha_value = 0.01
      cv_value    = 1.0
      gamma_value = 1.4
      sig_s = 1.0
      sig_a = 1.0
      
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
      hydro_IC = computeAnalyticHydroSolution(mesh,t=0.0,
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
      t_end = 0.01
#      t_end = 0.002

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbose = True
      
      # run the rad-hydro transient
      rad_new, hydro_new = runNonlinearTransient(
         mesh         = mesh,
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
         mom_src      = mom_src,
         E_src        = E_src,
         psim_src     = psim_src,
         psip_src     = psip_src,
         verbose      = verbose)


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

