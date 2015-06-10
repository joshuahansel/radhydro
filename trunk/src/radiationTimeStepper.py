## @package src.radiationTimeStepper
#  Contains function for taking radiation time steps
#

from transientSource import computeRadiationSource
from globalConstants import SPD_OF_LGT as c
from radiationSolveSS import radiationSolveSS

## The parameter \f$\beta\f$ for each time-stepper
#
beta = {"CN":0.5, "BDF2":2./3., "BE":1.}

## Takes radiation time step
#
#  TODO: add input parameter desc.
#
#  @param[in] mesh          mesh object
#  @param[in] time_stepper  string identifier for the chosen time-stepper,
#                           e.g., 'CN'
#
def takeRadiationStep(mesh, time_stepper, problem_type, dt,
   cx_new, psi_left, psi_right, add_ext_source=False, **kwargs):

   # evaluate transient source
   Q_tr = computeRadiationSource(
      mesh           = mesh,
      time_stepper   = time_stepper,
      problem_type   = problem_type,
      add_ext_source = add_ext_source,
      dt             = dt,
      psi_left       = psi_left,
      psi_right      = psi_right,
      **kwargs)

   # compute diagonal modifier
   alpha = 1./(c*dt)

   # solve transient system
   rad = radiationSolveSS(
      mesh           = mesh,
      cross_x        = cx_new,
      Q              = Q_tr,
      bc_psi_left    = psi_left,
      bc_psi_right   = psi_right,
      diag_add_term  = alpha,
      implicit_scale = beta[time_stepper] )

   return rad
