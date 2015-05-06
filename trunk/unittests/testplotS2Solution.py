# add parent directory to module search path
import sys
sys.path.append('..')

from src.plotS2Solution import plotS2Solution
from src.Mesh import Mesh
import unittest

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
