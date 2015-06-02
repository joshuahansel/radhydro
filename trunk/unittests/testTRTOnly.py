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
from plotUtilities import printTupled


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
      n_elems = 5
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
            cross_sects.append( (CrossXInterface(sig_a1, sig_s1),
               CrossXInterface(sig_a1, sig_s1)) )
            hydro_states.append( (HydroState(u=0,rho=rho1,int_energy=e1,
                gamma=gam,spec_heat=c_v1),
                HydroState(u=0,rho=rho1,int_energy=e1,gamma=gam,spec_heat=c_v1)) )
         else:
            cross_sects.append((CrossXInterface(sig_a2, sig_s2),
               CrossXInterface(sig_a2, sig_s2)))
            hydro_states.append( (HydroState(u=0,rho=rho2,int_energy=e2,spec_heat=c_v2, gamma=gam), 
                HydroState(u=0,rho=rho2,spec_heat=c_v2,int_energy=e2, gamma=gam)) )


      #keep copy of original cross sections
      cx_orig = deepcopy(cross_sects)

      # time step size and c*dt
      dt = 0.001
      t = 0.
      t_end = 1.0
      c_dt = GC.SPD_OF_LGT*dt
  
      psi_left  = 1.0
      psi_right = 0.25
  
      # create the steady-state source
      n = 4*mesh.n_elems
      Q = np.zeros(n)
  
      #initialize radiation
      psi_left = computeEquivIntensity(T_l)
      psi_right = computeEquivIntensity(T_r)
      psi_old   = [psi_right for i in range(n)] #equilibrium solution

      #initiialize  hydro old
      hydro_old = deepcopy(hydro_states)

      #print out temperature to check
      psi_m, psi_p = extractAngularFluxes(psi_old,mesh)

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
                               hydro_states_implicit=hydro_states)

          # take radiation step, currently hardcoded here
          transient_source = TransientSource(mesh, time_stepper,problem_type='trt',
                  newton_handler=newton_handler)
              
          # get the modified cross scattering cross sections
          cross_sects = newton_handler.getEffectiveOpacities(cx_orig,dt)
          printTupled(cross_sects)

          # evaluate transient source, including linearized planckian
          Q_tr = transient_source.evaluate(
              dt            = dt,
              bc_flux_left  = psi_left,
              bc_flux_right = psi_right,
              cx_older      = cx_orig,
              cx_old        = cx_orig,
              cx_new        = cross_sects,
              psi_old       = psi_old ,
          hydro_states_star = hydro_old)

          # solve the transient system
          alpha = 1./(GC.SPD_OF_LGT*dt)
          psi = radiationSolveSS(mesh, cross_sects, Q_tr,
             bc_psi_left = psi_left,
             bc_psi_right = psi_right,
             diag_add_term = alpha, implicit_scale = beta[time_stepper] )

          # extract angular fluxes from solution vector
          psim, psip = extractAngularFluxes(psi, mesh)

          # compute scalar flux
          phi = computeScalarFlux(psip, psim)

          # compute difference of transient and steady-state scalar flux
          phi_diff = [tuple(map(operator.sub, phi[i], phi_ss[i]))
             for i in xrange(len(phi))]

          # compute discrete L1 norm of difference
          L1_norm_diff = computeDiscreteL1Norm(phi_diff)

          # print each time step if run standalone
          if __name__ == '__main__':
             print("t = %0.3f -> %0.3f: L1 norm of diff with steady-state: %7.3e"
                % (t-dt,t,L1_norm_diff))
  
          # save oldest solutions
          psi_older = deepcopy(psi_old)
  
          # save old solutions
          psi_old = deepcopy(psi)
  
      # plot solutions if run standalone
      if __name__ == "__main__":
         plotScalarFlux(mesh, psim, psip, scalar_flux_exact=phi_ss,
            exact_data_continuous=False)

      # assert that solution has converged
      n_decimal_places = 12
      self.assertAlmostEqual(L1_norm_diff,0.0,n_decimal_places)

  
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()
