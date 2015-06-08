## @package testTRTOnly
#  Tests the two material TRT problem. Currently with BE time disc.
# this is just for debugging. This will all be in an executioner class
#

import sys
sys.path.append('../src')

from random import random
import numpy as np
import unittest

from mesh import Mesh
from crossXInterface import ConstantCrossSection
from radiationSolveSS import radiationSolveSS
from musclHancock import HydroState
import globalConstants as GC
from TRTUtilities import convSpecHeatErgsEvToJksKev, \
                         computeEquivIntensity, \
                         computeRadTemp
from newtonStateHandler import NewtonStateHandler                         
from radUtilities import *
from copy import deepcopy
from plotUtilities import printTupled, plotTemperatures
from utilityFunctions import computeL2RelDiff
from radiation import Radiation
from radiationTimeStepper import takeRadiationStep
from nonlinearSolve import nonlinearSolve
from transientSource import computeRadiationSource


## Derived unittest class to test source builder
#
class TestTRTOnly(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_TRTBE(self):

      # number of decimal places to test
      n_decimal_places = 13

      # create mesh
      n_elems = 200
      mesh = Mesh(n_elems,1.)

      # initialize temperature
      T_init = 0.05
      T_l    = 0.5
      T_r    = 0.05

      gam = 1.4
 
      # material 1 properties and IC
      sig_s1 = 0.0
      sig_a1 = 0.2  #macroscopic, not micro so no factor of rho here, cm^-1
      c_v1   = convSpecHeatErgsEvToJksKev(1.E+12) #specific heat capacity in ergs/(ev-g)
      rho1   = 0.01  #g/cc
      e1     = T_init*c_v1

      # material 2 properties and IC
      sig_s2 = 0.0
      sig_a2 = 2000.
      c_v2   = c_v1
      rho2   = 10.
      e2     = T_init *c_v2

      #Construct hydro states and cross sections
      cross_sects = []
      hydro_states = []
      for i in range(mesh.n_elems):  
      
         if mesh.getElement(i).x_cent < 0.5:
            cross_sects.append( (ConstantCrossSection(sig_s1, sig_s1+sig_a1),
                                 ConstantCrossSection(sig_s1, sig_s1+sig_a1)) )
            hydro_states.append( (
               HydroState(u=0,rho=rho1,int_energy=e1,gamma=gam,spec_heat=c_v1),
               HydroState(u=0,rho=rho1,int_energy=e1,gamma=gam,spec_heat=c_v1)) )
         else:
            cross_sects.append((ConstantCrossSection(sig_s2, sig_a2+sig_s2),
                                ConstantCrossSection(sig_s2, sig_a2+sig_s2)))
            hydro_states.append( (
               HydroState(u=0,rho=rho2,int_energy=e2,spec_heat=c_v2, gamma=gam), 
               HydroState(u=0,rho=rho2,spec_heat=c_v2,int_energy=e2, gamma=gam)) )


      #keep copy of original cross sections
      cx_orig = deepcopy(cross_sects)

      # time step size and c*dt
      dt = 0.001
      t = 0.
      t_end = 0.04
      c_dt = GC.SPD_OF_LGT*dt
  
      # create the steady-state source
      n = 4*mesh.n_elems
      Q = np.zeros(n)
  
      #initialize radiation
      psi_left  = computeEquivIntensity(T_l)
      psi_right = computeEquivIntensity(T_r)
      rad_old   = Radiation([psi_right for i in range(n)]) #equilibrium solution
      rad_older = deepcopy(rad_old)

      #initiialize  hydro old
      hydro_old = deepcopy(hydro_states)

      # time-stepper
      time_stepper = "BE"
      beta = {"CN":0.5, "BDF2":2./3., "BE":1.}

      # transient loop
      transient_incomplete = True # boolean flag signalling end of transient

      while transient_incomplete:
  
          # adjust time step size if necessary
          if t + dt >= t_end:
             dt = t_end - t
             t = t_end
             transient_incomplete = False # signal end of transient
          else:
             t += dt

          c_dt = GC.SPD_OF_LGT*dt

          # construct newton state handler
          newton_handler = NewtonStateHandler(mesh,
                               time_stepper=time_stepper,
                               cx_new = cx_orig,
                               hydro_guess=hydro_states)

          # nonlinear iteration tolerance
          tol = 1.E-9

          #initialize copies
          hydro_older = deepcopy(hydro_old)

          # perform nonlinear iterations:
          converged = False
          k = 0
          while not converged:

#               # increment iteration counter
#               k += 1
#
#               # newton_handler returns a deepcopy, not a name copy
#               hydro_prev = newton_handler.getNewHydroStates()
#
#               # get the modified scattering cross sections
#               cx_mod_prev = newton_handler.getEffectiveOpacities(dt)
#
#               planckian_new = newton_handler.evalPlanckianImplicit(dt=dt, hydro_star= hydro_old)
#
#               # evaluate transient source, including linearized planckian
#               Q_tr = computeRadiationSource(
#                   mesh          = mesh,
#                   time_stepper   = time_stepper,
#                   problem_type   = 'rad_mat',
#                   planckian_new  = planckian_new,
#                   dt            = dt,
#                   psi_left      = psi_left,
#                   psi_right     = psi_right,
#                   cx_older      = cx_orig,
#                   cx_old        = cx_orig,
#                   cx_new        = cx_mod_prev,
#                   rad_old       = rad_old ,
#                   rad_older     = rad_older,
#                   hydro_star    = hydro_old,
#                   hydro_older   = hydro_older,
#                   hydro_old     = hydro_old)
# 
#               # solve the transient system
#               alpha = 1./(GC.SPD_OF_LGT*dt)
#               rad_new = radiationSolveSS(mesh, cx_mod_prev, Q_tr,
#                  bc_psi_left = psi_left,
#                  bc_psi_right = psi_right,
#                  diag_add_term = alpha, implicit_scale = beta[time_stepper] )
#
#               # update internal energy
#               newton_handler.updateIntEnergy(rad_new.E, dt, hydro_star = hydro_old)
#
#               # check nonlinear convergence
#               hydro_new = newton_handler.getNewHydroStates()
#               rel_diff = computeL2RelDiff(hydro_new, hydro_prev, aux_func=lambda x: x.e)
#               print("  Iteration %d: Difference = %7.3e" % (k,rel_diff))
#               if rel_diff < tol:
#                  print("  Nonlinear iteration converged")
#                  break
#
#               # store new to prev
#               hydro_prev = hydro_new
              # increment iteration counter
              k += 1

              hydro_prev = newton_handler.getNewHydroStates()

              # get the modified scattering cross sections
              cross_sects = newton_handler.getEffectiveOpacities(dt)

              #evaluate new plackian term
              planckian_new = newton_handler.evalPlanckianImplicit(dt=dt, hydro_star
                      = hydro_old)

              # evaluate transient source, including linearized planckian
              Q_tr = computeRadiationSource(
                  mesh          = mesh,
                  time_stepper   = time_stepper,
                  problem_type   = 'rad_mat',
                  planckian_new  = planckian_new,
                  dt            = dt,
                  psi_left      = psi_left,
                  psi_right     = psi_right,
                  cx_older      = cx_orig,
                  cx_old        = cx_orig,
                  cx_new        = cross_sects,
                  rad_old       = rad_old ,
                  rad_older     = rad_older,
                  hydro_star    = hydro_old,
                  hydro_older   = hydro_older,
                  hydro_old     = hydro_old)

              # solve the transient system
              alpha = 1./(GC.SPD_OF_LGT*dt)
              rad_new = radiationSolveSS(mesh, cross_sects, Q_tr,
                 bc_psi_left = psi_left,
                 bc_psi_right = psi_right,
                 diag_add_term = alpha, implicit_scale = beta[time_stepper] )


              #update internal energies 
              newton_handler.updateIntEnergy(rad_new.E,dt,hydro_star = hydro_old)

              # check nonlinear convergence
              hydro_new = newton_handler.getNewHydroStates()
              rel_diff = computeL2RelDiff(hydro_new, hydro_prev, aux_func=lambda x: x.e)
              print("Iteration %d: Difference = %7.3e" % (k,rel_diff))
              if rel_diff < tol:
                 print("Nonlinear iteration converged")
                 break

              #store new to prev
              hydro_prev = hydro_new
                  
          # print each time step if run standalone
          if __name__ == '__main__':
             print("t = %0.3f -> %0.3f:"
                % (t-dt,t) )
  
          print "Difference in time steps: ", computeL2RelDiff(hydro_old, hydro_new,
                  aux_func=lambda x: x.e)

          # save oldest solutions
          hydro_older = deepcopy(hydro_old)
          rad_older = deepcopy(rad_old)
  
          # save old solutions
          hydro_old = deepcopy(hydro_new)
          rad_old   = deepcopy(rad_new)

          # store hydro
          hydro_states = hydro_new


      # plot solutions if run standalone
      if __name__ == "__main__":
          plotTemperatures(mesh, rad_new.E, hydro_states=hydro_new, print_values=True)

  
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

