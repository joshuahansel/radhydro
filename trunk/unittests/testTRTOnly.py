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
from crossXInterface import CrossXInterface
from radiationSolveSS import radiationSolveSS
from musclHancock import HydroState
from transientSource import * 
from radiationTimeStepper import RadiationTimeStepper
import globalConstants as GC
from TRTUtilities import convSpecHeatErgsEvToJksKev, \
                         computeEquivIntensity, \
                         computeRadTemp
from newtonStateHandler import NewtonStateHandler                         
from radUtilities import *
from copy import deepcopy
from plotUtilities import printTupled, plotTemperatures
from utilityFunctions import computeL2RelDiff


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

      # Set up the two material problem.
      # build constant cross sections
      gam = 1.4
      sig_s1 = 0.0
      sig_a1 = 0.2  #macroscopic, not micro so no factor of rho here, cm^-1
      c_v1   = convSpecHeatErgsEvToJksKev(1.E+12) #specific heat capacity in ergs/(ev-g)
      rho1   = 0.01  #g/cc
      e1     = T_init*c_v1

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
            cross_sects.append( (CrossXInterface(sig_s1, sig_s1+sig_a1),
               CrossXInterface(sig_s1, sig_s1+sig_a1)) )
            hydro_states.append( (HydroState(u=0,rho=rho1,int_energy=e1,
                gamma=gam,spec_heat=c_v1),
                HydroState(u=0,rho=rho1,int_energy=e1,gamma=gam,spec_heat=c_v1)) )
         else:
            cross_sects.append((CrossXInterface(sig_s2, sig_a2+sig_s2),
               CrossXInterface(sig_s2, sig_a2+sig_s2)))
            hydro_states.append( (HydroState(u=0,rho=rho2,int_energy=e2,spec_heat=c_v2, gamma=gam), 
                HydroState(u=0,rho=rho2,spec_heat=c_v2,int_energy=e2, gamma=gam)) )


      #keep copy of original cross sections
      cx_orig = deepcopy(cross_sects)

      # time step size and c*dt
      dt = 0.001
      t = 0.
      t_end = 0.001
      c_dt = GC.SPD_OF_LGT*dt
  
      # create the steady-state source
      n = 4*mesh.n_elems
      Q = np.zeros(n)
  
      #initialize radiation
      psi_left  = computeEquivIntensity(T_l)
      psi_right = computeEquivIntensity(T_r)
      psi_old   = [psi_right for i in range(n)] #equilibrium solution

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

          # construct newton state handler
          newton_handler = NewtonStateHandler(mesh,
                               time_stepper=time_stepper,
                               cx_new = cross_sects,
                               hydro_new=hydro_states)

          tol = 1.E-9

          #perform nonlinear iterations:
          while True:

              #store old temperature
              hydro_prev = newton_handler.getNewHydroStates()

              # get the modified scattering cross sections
              cross_sects = newton_handler.getEffectiveOpacities(dt)

              # take radiation step, currently hardcoded here
              transient_source = TransientSource(mesh, time_stepper, problem_type='trt',
                      newton_handler=newton_handler)
                  
              # evaluate transient source, including linearized planckian
              Q_tr = transient_source.evaluate(
                  dt            = dt,
                  bc_flux_left  = psi_left,
                  bc_flux_right = psi_right,
                  cx_older      = cx_orig,
                  cx_old        = cx_orig,
                  cx_new        = cross_sects,
                  psi_old       = psi_old ,
                  hydro_star = hydro_old)

              # solve the transient system
              alpha = 1./(GC.SPD_OF_LGT*dt)
              psi = radiationSolveSS(mesh, cross_sects, Q_tr,
                 bc_psi_left = psi_left,
                 bc_psi_right = psi_right,
                 diag_add_term = alpha, implicit_scale = beta[time_stepper] )

              # extract angular fluxes from solution vector
              psim, psip = extractAngularFluxes(psi, mesh)

              # compute new scalar energy density
              E = computeEnergyDensity(psim, psip)

              #update internal energies 
              newton_handler.updateIntEnergy(E,dt,hydro_star = hydro_old)

              #check convergence
              hydro_new = newton_handler.getNewHydroStates()
              rel_diff = computeL2RelDiff(hydro_new, hydro_prev, aux_func=lambda x: x.e)
              print "Current difference is: ", rel_diff

              #store new to prev
              hydro_prev = hydro_new

              if rel_diff < tol:
                 break
                  
          # print each time step if run standalone
          if __name__ == '__main__':
             print("t = %0.3f -> %0.3f:"
                % (t-dt,t) )
  
          # save oldest solutions
          psi_older = deepcopy(psi_old)
  
          # save old solutions
          psi_old = deepcopy(psi)

          print "Difference in time steps: ", computeL2RelDiff(hydro_old, hydro_new,
                  aux_func=lambda x: x.e)

          #store hydro
          hydro_states = hydro_new
          hydro_old = deepcopy(hydro_states)


      # plot solutions if run standalone
      if __name__ == "__main__":
          plotTemperatures(mesh, E, hydro_states=hydro_states)

      print "Radiation temperature"
      #printTupled(computeRadTemp(psim, psip))

      # assert that solution has converged
      n_decimal_places = 12
      self.assertAlmostEqual(L1_norm_diff,0.0,n_decimal_places)

  
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()
