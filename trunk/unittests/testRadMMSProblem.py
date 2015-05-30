## @package testRadMMSProblem
#

import sys
sys.path.append('../src')

import matplotlib as plt
from pylab import *
import numpy as np
from copy import deepcopy
import unittest
import operator # for adding tuples to each other elementwise
from math import pi, sin, cos

from mesh import Mesh
from crossXInterface import CrossXInterface
from radiationSolveSS import radiationSolveSS
from plotUtilities import makeContinuousXPoints, plotAngularFlux, plotScalarFlux
from utilityFunctions import computeDiscreteL1Norm, getIndex
from radiationTimeStepper import RadiationTimeStepper
from radUtilities import mu, computeScalarFlux
import globalConstants as GC

## Derived unittest class to run a transient radiation MMS problem
#
class TestRadTransient(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_RadTransient(self):

       # create uniform mesh
       mesh = Mesh(50, 1.0)
   
       # compute uniform cross sections
       sig_s = 1.0
       sig_a = 2.0
       cross_sects = [(CrossXInterface(sig_a, sig_s), CrossXInterface(sig_a, sig_s))
                     for i in xrange(mesh.n_elems)]
   
       # transient options
       dt      = 0.1         # time step size
       t_start = 0.0         # begin time
       t_end   = 10.0        # end time
       time_stepper = 'CN' # time-stepper
   
       # boundary fluxes
       psi_left  = 0.0
       psi_right = 0.0
   
       # speed of light
       c = GC.SPD_OF_LGT

       # MMS source functions
       # pointwise source for minus direction
       def QMMSm(t,x,sigs,sigt):
          return 2.0/c*sin(pi*(1.0-x)) - 2.0*mu["-"]*pi*t*cos(pi*(1.0-x))\
             + 2.0*sigt*t*sin(pi*(1.0-x)) - sigs/2.0*(t*sin(pi*x)\
             + 2.0*t*sin(pi*(1.0-x)))

       # pointwise source for plus direction
       def QMMSp(t,x,sigs,sigt):
          return 1.0/c*sin(pi*x) + mu["+"]*pi*t*cos(pi*x)\
             + sigt*t*sin(pi*x) - sigs/2.0*(t*sin(pi*x)\
             + 2.0*t*sin(pi*(1.0-x)))

       # all solution values
       def QMMS(t,mesh,cx):
          Q = np.zeros(mesh.n_elems * 4)
          for i in xrange(mesh.n_elems):
             # get global indices
             iLm = getIndex(i,"L","-") # dof i,L,-
             iLp = getIndex(i,"L","+") # dof i,L,+
             iRm = getIndex(i,"R","-") # dof i,R,-
             iRp = getIndex(i,"R","+") # dof i,R,+

             # get x points of element
             el = mesh.getElement(i)
             xL = el.xl
             xR = el.xr

             # get cross section values
             sigsL = cx[i][0].sig_s
             sigtL = cx[i][0].sig_t
             sigsR = cx[i][1].sig_s
             sigtR = cx[i][1].sig_t

             # compute source
             Q[iLm] = QMMSm(t,xL,sigsL,sigtL)
             Q[iLp] = QMMSp(t,xL,sigsL,sigtL)
             Q[iRm] = QMMSm(t,xR,sigsR,sigtR)
             Q[iRp] = QMMSp(t,xR,sigsR,sigtR)

          return Q

       # zero IC
       psip_old   = [(0.0,0.0) for i in range(mesh.n_elems)]
       psim_old   = deepcopy(psip_old)
       psip_older = deepcopy(psip_old)
       psim_older = deepcopy(psip_old)
       E_old      = deepcopy(psip_old)
       E_older    = deepcopy(psip_old)
       Q_older    = QMMS(0.0,mesh,cross_sects)
       Q_old      = QMMS(0.0,mesh,cross_sects)
   
       # create time-stepper
       radiation_time_stepper = RadiationTimeStepper(mesh, time_stepper)

       # transient loop
       t = t_start
       transient_incomplete = True # boolean flag signalling end of transient
       while transient_incomplete:
   
           # adjust time step size if necessary
           if t + dt >= t_end:
              dt = t_end - t
              t = t_end
              transient_incomplete = False # signal end of transient
           else:
              t += dt

           # compute new source
           Q_new = QMMS(t,mesh,cross_sects)

           # take radiation step
           psim, psip, E, F = radiation_time_stepper.takeStep(
              dt            = dt,
              bc_flux_left  = psi_left,
              bc_flux_right = psi_right,
              cx_older      = cross_sects,
              cx_old        = cross_sects,
              cx_new        = cross_sects,
              psim_older    = psim_older,
              psip_older    = psip_older,
              psim_old      = psim_old,
              psip_old      = psip_old,
              E_older       = E_older,
              E_old         = E_old,
              Q_older       = Q_older,
              Q_old         = Q_old,
              Q_new         = Q_new)

           # compute scalar flux
           phi = computeScalarFlux(psip, psim)

           # print each time step if run standalone
           if __name__ == '__main__':
              print("t = %0.3f -> %0.3f" % (t-dt,t))
   
           # save oldest solutions
           psip_older = deepcopy(psip_old)
           psim_older = deepcopy(psim_old)
           E_older    = deepcopy(E_old)
           Q_older    = deepcopy(Q_old)
   
           # save old solutions
           psip_old = deepcopy(psip)
           psim_old = deepcopy(psim)
           E_old    = deepcopy(E)
           Q_old    = deepcopy(Q_new)

       # get continuous x-points
       xlist = makeContinuousXPoints(mesh)
   
       # compute exact scalar flux solution
       def exactPhi(x,t):
          return t*sin(pi*x) + 2.0*t*sin(pi*(1.0-x))
       phi_exact = [exactPhi(x,t_end) for x in xlist]

       # plot solutions if run standalone
       if __name__ == "__main__":
          plotScalarFlux(mesh, psim, psip, scalar_flux_exact=phi_exact)

       # assert that solution has converged
       #n_decimal_places = 12
       #self.assertAlmostEqual(L1_norm_diff,0.0,n_decimal_places)
       
    
# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

