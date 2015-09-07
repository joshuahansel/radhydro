## @package unittests.testCreateMMSSourceFunctions
#  Contains unittest class to test the MMS source creator functions

# add source directory to module search path
import sys
sys.path.append('../src')

# symbolic math packages
from sympy import symbols, exp, sin, pi

# numpy
import numpy as np

# unit test package
import unittest

# local packages
from plotUtilities import plotFunction
from createMMSSourceFunctions import createMMSSourceFunctionsRadHydro

## Derived unittest class to test the MMS source creator functions
#
class TestCreateMMSSourceFunctions(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_CreateMMSSourceFunctionsRadHydro(self):
      
      # declare symbolic variables
      x, t, alpha = symbols('x t alpha')
      
      # create solution for thermodynamic state and flow field
      rho = exp(x+t)
      u   = exp(-x)*sin(t) - 1
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
      rho_f, u_f, E_f, psim_f, psip_f = createMMSSourceFunctionsRadHydro(
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

      # if run standalone, then plot the MMS functions
      if __name__ == '__main__':

         # create list of x points and the time value at which to evaluate
         xpoints = np.linspace(0.0,1.0,100)
         t_value = 0.1
         
         # plot
         plotFunction(rho_f,  xpoints, t_value, '$Q_\\rho$', 'MMS Source, $\\rho$')
         plotFunction(u_f,    xpoints, t_value, '$Q_u$',     'MMS Source, $u$')
         plotFunction(E_f,    xpoints, t_value, '$Q_E$',     'MMS Source, $E$')
         plotFunction(psim_f, xpoints, t_value, '$Q_-$',     'MMS Source, $\Psi^-$')
         plotFunction(psip_f, xpoints, t_value, '$Q_+$',     'MMS Source, $\Psi^+$')

      # assert that the correct value is produced
      test_value = E_f(0.1,0.6)
      actual_value = 17.263054849815813
      self.assertAlmostEqual(test_value, actual_value, 12)

# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

