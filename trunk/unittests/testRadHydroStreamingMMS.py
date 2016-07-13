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
from math import sqrt

# unit test package
import unittest

# local packages
from createMMSSourceFunctions import createMMSSourceFunctionsRadHydro
from mesh import Mesh
from hydroState import HydroState
from radiation import Radiation
from plotUtilities import plotHydroSolutions, plotTemperatures, plotRadErg, plotS2Erg
from utilityFunctions import computeRadiationVector, computeAnalyticHydroSolution,\
   computeHydroL2Error, computeHydroConvergenceRates, printHydroConvergenceTable,\
   computeAnalyticRadSolution
from crossXInterface import ConstantCrossSection
from transient import runNonlinearTransient
from hydroBC import HydroBC
from radBC import RadBC
import globalConstants as GC
import radUtilities as RU
import pickle

class TestRadHydroMMS(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_RadHydroMMS(self):
      
      # declare symbolic variables
      x, t, A, B, C,  c, cv, gamma, mu, alpha = \
         symbols('x t A B C c cv gamma mu alpha')
      
      #These constants, as well as cross sections and C_v, rho_ref
      #will set the material 
      #and radiation to be small relative to kinetic energy
      Ctilde = 100.
      P = 0.1

      #Arbitrary mach number well below the sound speed.  The choice of gamma and
      #the cv value, as well as C and P constrain all other material reference
      #parameters, but we are free to choose the material velocity below the sound
      #speed to ensure no shocks are formed
      M = 0.9
      cfl_value = 0.6  #but 2 time steps, so really like a CFL of 0.3
      width = 2.0*math.pi

      # numeric values
      gamma_value = 5./3.
      cv_value = 0.14472799784454

      # number of refinement cycles
      n_cycles = 5

      # number of elements in first cycle
      n_elems = 40

      # transient options
      t_start  = 0.0
      t_end = 2.0*math.pi/Ctilde

      # numeric values
      A_value = 1.0
      B_value = 1.0
      C_value = 1.0
      alpha_value = 0.5
      gamma_value = 5.0/3.0
      sig_s_value = 0.0
      sig_a_value = 1.0
    
      # numeric values
      a_inf = GC.SPD_OF_LGT/Ctilde
      #T_inf = a_inf**2/(gamma_value*(gamma_value - 1.)*cv_value)
      #rho_inf = GC.RAD_CONSTANT*T_inf**4/(P*a_inf**2)
      rho_inf = 1.0
      T_inf = pow(rho_inf*P*a_inf**2/GC.RAD_CONSTANT,0.25)  #to set T_inf based on rho_inf,
      cv_value = a_inf**2/(T_inf*gamma_value*(gamma_value-1.)) # to set c_v, if rho specified
      p_inf = rho_inf*a_inf*a_inf
      Er_inf = GC.RAD_CONSTANT*T_inf**4

      # MMS solutions
      rho = rho_inf*A*(sin(B*x-C*t)+2.)
      u   = M*a_inf*1./(A*(sin(B*x-C*t)+2.))
      p   = p_inf*A*alpha*(sin(B*x-C*t)+2.)
      Er  = alpha*(sin(B*x-Ctilde*C*t)+2.)*Er_inf
      Fr  = alpha*(sin(B*x-Ctilde*C*t)+2.)*c*Er_inf
      #Er = 0.5*(sin(2*pi*x - 10.*t) + 2.)/c
      #Fr = 0.5*(sin(2*pi*x - 10.*t) + 2.)
      e = p/(rho*(gamma_value-1.))
      E = 0.5*rho*u*u + rho*e

      print "The dimensionalization parameters are: "
      print "   a_inf : ", a_inf
      print "   T_inf : ", T_inf
      print " rho_inf : ", rho_inf
      print "   p_inf : ", p_inf

      # derived solutions
      T = e/cv_value
      E = rho*(u*u/2 + e)
      psip = (Er*c + Fr/mu)/2
      psim = (Er*c - Fr/mu)/2


      # create list of substitutions
      substitutions = dict()
      substitutions['A']     = A_value
      substitutions['B']     = B_value
      substitutions['C']     = C_value
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

      dt = []
      dx = []
      err = []

      for cycle in range(n_cycles):
      
          # create uniform mesh
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

          # compute the initial time step size according to CFL conditon (actually
          # half). We will then decrease this time step by the same factor as DX each
          # cycle
          if cycle == 0:

              sound_speed = [sqrt(i.p * i.gamma / i.rho) + abs(i.u) for i in hydro_IC]
              dt_vals = [cfl_value*(mesh.elements[i].dx)/sound_speed[i]
                 for i in xrange(len(hydro_IC))]
              dt_value = min(dt_vals)
              print "dt_value for hydro: ", dt_value

              #Make sure not taking too large of step for radiation time scale
              period = 2.*math.pi/Ctilde
              dt_value = min(dt_value, period/8.)
            
              print "initial dt_value", dt_value

              #Adjust the end time to be an exact increment of dt_values
              print "t_end: ", t_end

          print "This cycle's dt value: ", dt_value
          dt.append(dt_value)
          dx.append(mesh.getElement(0).dx)

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
             dt_option    = 'constant',
             dt_constant  = dt_value,
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

          # compute exact hydro solution
          hydro_exact = computeAnalyticHydroSolution(mesh, t=t_end,
             rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)

          rad_exact = computeAnalyticRadSolution(mesh, t_end,psim=psim_f,psip=psip_f)
 
          #Compute error
          err.append(computeHydroL2Error(hydro_new, hydro_exact, rad_new, rad_exact ))

          n_elems  *= 2
          dt_value *= 0.5

          # compute convergence rates
          rates_dx = computeHydroConvergenceRates(dx,err)
          rates_dt = computeHydroConvergenceRates(dt,err)

          # print convergence table
          if n_cycles > 1:
             printHydroConvergenceTable(dx,err,rates=rates_dx,
                dx_desc='dx',err_desc='$L_2$')
             printHydroConvergenceTable(dt,err,rates=rates_dt,
                dx_desc='dt',err_desc='$L_2$')

      # plot
      if __name__ == '__main__':

         # compute exact hydro solution
         hydro_exact = computeAnalyticHydroSolution(mesh, t=t_end,
            rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)


         # plot hydro solution
         plotHydroSolutions(mesh, hydro_new, x_exact=mesh.getCellCenters(),exact=hydro_exact)

         #plot exact and our E_r
         Er_exact_fn = 1./GC.SPD_OF_LGT*(psim + psip)
         Fr_exact_fn = (psip - psim)*RU.mu["+"]
         Er_exact = []
         Fr_exact = []
         psip_exact = []
         psim_exact = []
         x = mesh.getCellCenters()
         for xi in x:
             
             substitutions = {'x':xi, 't':t_end}
             Er_exact.append(Er_exact_fn.subs(substitutions))
             Fr_exact.append(Fr_exact_fn.subs(substitutions))
             psip_exact.append(psip_f(xi,t_end))
             psim_exact.append(psim_f(xi,t_end))

         plotRadErg(mesh, rad_new.E, Fr_edge=rad_new.F, exact_Er=Er_exact, exact_Fr =
               Fr_exact)

         plotS2Erg(mesh, rad_new.psim, rad_new.psip, exact_psim=psim_exact,
                 exact_psip=psip_exact)

         plotTemperatures(mesh, rad_new.E, hydro_states=hydro_new, print_values=True)

         #Make a pickle to save the error tables
         from sys import argv
         pickname = "results/testRadHydroStreamingMMSC100.pickle"
         if len(argv) > 2:
            if argv[1] == "-o":
               pickname = argv[2].strip()

         #Create dictionary of all the data
         big_dic = {"dx": dx}
         big_dic["dt"] =  dt
         big_dic["Errors"] = err
         pickle.dump( big_dic, open( pickname, "w") )

# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

