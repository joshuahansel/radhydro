## @package unittests.testRadHydroShock
#  Contains unittest class to test a radiation-hydrodyamics shock problem.

# add source directory to module search path
import sys
sys.path.append('../src')

# unit test package
import unittest

# local packages
from mesh import Mesh
from hydroState import HydroState
from radiation import Radiation
from plotUtilities import plotHydroSolutions, plotTemperatures
from utilityFunctions import computeRadiationVector, computeAnalyticHydroSolution
from crossXInterface import ConstantCrossSection
from transient import runNonlinearTransient
from hydroBC import HydroBC
from radBC   import RadBC
import globalConstants as GC

## Derived unittest class to test the radiation-hydrodynamics shock problem
#
class TestRadHydroShock(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_RadHydroShock(self):

      # test case
      test_case = "mach2" # mach1.2 mach2 mach50
      
      # create uniform mesh
      n_elems = 100
      width = 0.02
      x_start = -0.01
      mesh_center = x_start + 0.5*width
      mesh = Mesh(n_elems, width, x_start=x_start)

      # gamma constant
      gam = 5.0/3.0
 
      # material 1 and 2 properties: Table 6.1
      sig_a1 = 390.71164263502122
      sig_s1 = 8.5314410158161809E+2-sig_a1
      c_v1   = 1.2348000000000001e-01
      sig_a2 = sig_a1
      sig_s2 = sig_s1
      c_v2   = c_v1

      if test_case == "mach1.2": # Mach 1.2 problem: Table 6.2

         # material 1 IC
         rho1   = 1.0
         E1     = 2.2226400000000000e-02
         u1     = 1.4055888445772469e-01
         e1     = E1/rho1 - 0.5*u1*u1
         Erad_left = 1.372E-06

         # material 2 IC
         rho2   = 1.2973213452231311
         E2     = 2.6753570531538713e-002
         u2     = 1.0834546504247138e-001
         e2     = E2/rho2 - 0.5*u2*u2
         Erad_right = 2.7955320762182542e-06

         # final time
         t_end = 0.5

         # temperature plot filename
         test_filename = "radshock_mach1.2.pdf"

         # temperature plot exact solution filename
         exact_solution_filename = "mach1.2_exact_solution.csv"

      elif test_case == "mach2": # Mach 2 problem: Table 6.3

         # material 1 IC
         rho1   = 1.0
         E1     = 3.9788000000000004e-002
         u1     = 2.3426480742954117e-001
         e1     = E1/rho1 - 0.5*u1*u1
         Erad_left = 1.372E-06

         # material 2 IC
         rho2   = 2.2860748989303659e+000
         E2     = 7.0649692950433357e-002
         u2     = 1.0247468599526272e-001
         e2     = E2/rho2 - 0.5*u2*u2
         Erad_right = 2.5560936967521927e-005

         # final time
         #t_end = 0.5
         t_end = 0.001

         # temperature plot output filename
         test_filename = "radshock_mach2.pdf"

         # temperature plot exact solution filename
         exact_solution_filename = "mach2_exact_solution.csv"

      elif test_case == "mach50": # Mach 50 problem: Table 6.4

         raise NotImplementedError("Mach 50 test requires negativity monitoring," \
            + "which is not yet implemented.")

         # material 1 IC
         rho1   = 1.0
         E1     = 1.7162348000000001e+001
         u1     = 5.8566201857385289e+000
         e1     = E1/rho1 - 0.5*u1*u1
         Erad_left = 1.372E-06

         # material 2 IC
         rho2   = 6.5189217901173153e+000
         E2     = 9.5144308747326214e+000
         u2     = 8.9840319830453630e-001
         e2     = E2/rho2 - 0.5*u2*u2
         Erad_right = 7.3372623010289956e+001

         # final time
         t_end = 1.5

         # temperature plot filename
         test_filename = "radshock_mach50.pdf"

         # temperature plot exact solution filename
         exact_solution_filename = "mach50_exact_solution.csv"

      else:
         raise NotImplementedError("Invalid test case")
         
      # compute radiation BC; assumes BC is independent of time
      c = GC.SPD_OF_LGT
      # NOTE: What is the justification for this? Does Jarrod assume Fr = 0?
      psi_left  = 0.5*c*Erad_left  
      psi_right = 0.5*c*Erad_right

      #Create BC object
      rad_BC = RadBC(mesh, "dirichlet", psi_left=psi_left, psi_right=psi_right)
      
      # construct cross sections and hydro IC
      cross_sects = list()
      hydro_IC = list()
      psi_IC = list()

      for i in range(mesh.n_elems):  
      
         if mesh.getElement(i).x_cent < mesh_center: # material 1
            cross_sects.append( (ConstantCrossSection(sig_s1, sig_s1+sig_a1),
                                 ConstantCrossSection(sig_s1, sig_s1+sig_a1)) )
            hydro_IC.append(
               HydroState(u=u1,rho=rho1,e=e1,spec_heat=c_v1,gamma=gam))

            psi_IC += [psi_left for dof in range(4)]

         else: # material 2
            cross_sects.append((ConstantCrossSection(sig_s2, sig_a2+sig_s2),
                                ConstantCrossSection(sig_s2, sig_a2+sig_s2)))
            hydro_IC.append(
               HydroState(u=u2,rho=rho2,e=e2,spec_heat=c_v2,gamma=gam))

            psi_IC += [psi_right for dof in range(4)]

      rad_IC = Radiation(psi_IC)

      # create hydro BC
      hydro_BC = HydroBC(bc_type='reflective', mesh=mesh)
  
      # transient options
      t_start  = 0.0

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbosity = 2
      else:
         verbosity = 0
      
      # run the rad-hydro transient
      rad_new, hydro_new = runNonlinearTransient(
         mesh         = mesh,
         problem_type = 'rad_hydro',
         dt_option    = 'CFL',
         CFL          = 0.3,
         use_2_cycles = False,
         time_stepper = 'BDF2',
         t_start      = t_start,
         t_end        = t_end,
         rad_BC       = rad_BC,
         cross_sects  = cross_sects,
         rad_IC       = rad_IC,
         hydro_IC     = hydro_IC,
         hydro_BC     = hydro_BC,
         verbosity    = verbosity,
         slope_limiter = 'vanleer',
         check_balance=True)

      # plot
      if __name__ == '__main__':

         # compute exact hydro solution
         hydro_exact = None

         # plot hydro solution
         plotHydroSolutions(mesh, hydro_new, exact=hydro_exact)

         # plot material and radiation temperatures
         plotTemperatures(mesh, rad_new.E, hydro_states=hydro_new, print_values=False,
            save=True, filename=test_filename,
            exact_solution_filename=exact_solution_filename)

# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

