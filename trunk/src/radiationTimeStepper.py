## @package src.radiationTimeStepper
#  Contains class for taking radiation time steps
#

from transientSource import TransientSource
import globalConstants as GC
from radiationSolveSS import radiationSolveSS

## The parameter \f$\beta\f$ for each time-stepper
#
beta = {"CN":0.5, "BDF2":2./3., "BE":1.}

## Takes radiation time step
#
class RadiationTimeStepper:

   ## Constructor
   #
   #  @param[in] mesh          mesh object
   #  @param[in] time_stepper  string identifier for the chosen time-stepper,
   #                           e.g., 'CN'
   #
   def __init__(self, mesh, time_stepper):
      self.mesh         = mesh
      self.time_stepper = time_stepper
      self.transient_source = TransientSource(mesh, time_stepper)

   ## Takes time step
   #
   def takeStep(self, **kwargs):

      # evaluate transient source
      Q_tr = self.transient_source.evaluate(**kwargs)

      # solve the transient system
      alpha = 1./(GC.SPD_OF_LGT*kwargs['dt'])
      psi = radiationSolveSS(self.mesh, kwargs['cx_new'], Q_tr,
         bc_psi_left = kwargs['bc_flux_left'],
         bc_psi_right = kwargs['bc_flux_right'],
         diag_add_term = alpha, implicit_scale = beta[self.time_stepper] )

      return psi
