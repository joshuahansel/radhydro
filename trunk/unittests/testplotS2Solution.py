## @package testplotS2Solution
#  Tests the S-2 plot utilities.

# add source directory to module search path
import sys
sys.path.append('../src')

from plotUtilities import plotS2Solution
from mesh import Mesh
import unittest

## Derived unittest class to test S-2 plot utilities
class TestPlotS2Solution(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def testPlot(self):
      n_elems = 5
      mesh = Mesh(n_elems,1.0)
      psi_minus = [(i,i+1.5) for i in xrange(n_elems)]
      psi_plus = [(2*i+1.0,2*i+1.5) for i in xrange(n_elems)]
      plotS2Solution(mesh, psi_minus, psi_plus)

# run main function from unittest module
if __name__ == '__main__':
   unittest.main()
