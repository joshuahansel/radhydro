## @package unittests.testIntegrationUtilities
#  Tests integration utilities

# add source directory to module search path
import sys
sys.path.append('../src')

from mesh import Mesh
from integrationUtilities import computeL1ErrorLD
import unittest

## Derived unittest class to test integrator
class TestIntegrationUtilities(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_computeL1Error1D(self):
      # create mesh
      left = 2.5
      width = 1.5
      mesh = Mesh(3, width, x_start=left)

      # create "numerical solution" data
      numerical_solution = [(1.0,2.0),(1.6,2.2),(2.5,2.7)]

      # specify "exact solution" function
      def exact(x):
         return x**2 + 3.0

      # compute exact integral of difference
      exact_integral = 0.0
      for i in xrange(mesh.n_elems):
         # express local numerical solution as linear function y(x) = m*x + b
         el = mesh.getElement(i)
         xL = el.xl
         xR = el.xr
         yL = numerical_solution[i][0]
         yR = numerical_solution[i][1]
         dx = xR - xL
         dy = yR - yL
         m = dy / dx
         b = yL - xL*m

         # compute local integral of difference
         local_integral = (xR**3 - xL**3)/3.0 - 0.5*m*(xR**2 - xL**2)\
            + (3.0-b)*(xR - xL)
 
         # add to global integral of difference
         exact_integral += local_integral

      # compute numerical integral of difference
      numerical_integral = computeL1ErrorLD(mesh, numerical_solution, exact)

      # assert that numerical and exact integrals are approximately equal
      n_decimal_places = 14
      self.assertAlmostEqual(numerical_integral,exact_integral,n_decimal_places)


# run main function from unittest module
if __name__ == '__main__':
   unittest.main()
