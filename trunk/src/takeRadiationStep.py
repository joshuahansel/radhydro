## @package src.takeRadiationStep
#  Contains function to take a single radiation time step.

from transientSource import computeRadiationSource,\
                            computeRadiationExtraneousSource
from radiationSolveSS import radiationSolveSS
from globalConstants import SPD_OF_LGT as c


## The parameter \f$\beta\f$ for each time-stepper
#
beta = {"CN":0.5, "BDF2":2./3., "BE":1.}


## Takes single radiation time step
#
#  @param[in] mesh          mesh object
#  @param[in] time_stepper  string identifier for the chosen time-stepper,
#                           e.g., 'CN'
#
def takeRadiationStep(mesh, time_stepper, problem_type, dt,
   cx_new, rad_BC,
   Qpsi_older, Qpsi_old, Qpsi_new, **kwargs):

   # assert that the appropriate sources were provided
   assert Qpsi_new.size != 0, 'New source must be provided'
   if time_stepper != 'BE':
      assert Qpsi_old.size != 0, 'Old source must be provided for CN or BDF2'
   if time_stepper == 'BDF2':
      assert Qpsi_older.size != 0, 'Older source must be provided for BDF2'

   # evaluate transient source
   Q_tr = computeRadiationSource(
      mesh           = mesh,
      time_stepper   = time_stepper,
      problem_type   = problem_type,
      dt             = dt,
      rad_BC         = rad_BC,
      Qpsi_older     = Qpsi_older,
      Qpsi_old       = Qpsi_old,
      Qpsi_new       = Qpsi_new,
      **kwargs)

   # compute diagonal modifier
   alpha = 1./(c*dt)

   # solve transient system
   rad_new = radiationSolveSS(
      mesh           = mesh,
      cross_x        = cx_new,
      Q              = Q_tr,
      rad_BC         = rad_BC,
      diag_add_term  = alpha,
      implicit_scale = beta[time_stepper] )

   return rad_new

