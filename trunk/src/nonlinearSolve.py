## @package src.nonlinearSolve
#  Provides functions for performing nonlinear solves
#
#  TODO: generalize to accept all problem types
#

from newtonStateHandler import NewtonStateHandler
from radiationTimeStepper import takeRadiationStep
from utilityFunctions import computeL2RelDiff

## Performs nonlinear solve
#
#  @return new hydro and rad solutions
#
def nonlinearSolve(mesh, time_stepper, problem_type, dt, psi_left, psi_right,
   cx_old, hydro_old, hydro_guess, rad_old,rad_older=None,cx_older=None,hydro_older=None,tol=1.0e-9):

   if problem_type != 'rad_mat':
      raise NotImplementedError("Nonlinear solve function currently only solves\
         thermal radiative transfer problems")

   # construct newton state handler
   newton_handler = NewtonStateHandler(mesh,
                       time_stepper=time_stepper,
                       cx_new=cx_old,
                       hydro_guess=hydro_guess)

   # initialize convergence flag and iteration counter
   converged = False
   k = 0

   # perform nonlinear iterations:
   while not converged:

       # increment iteration counter
       k += 1

       # newton_handler returns a deepcopy, not a name copy
       hydro_prev = newton_handler.getNewHydroStates()

       #Update the QE term TODO this should be called 
       # inside the class. Probably need a flag set in newton handler
       # to ensure it is called before self.QE is ever used
       newton_handler.updateMatCouplingQE(cx_old=cx_old,
                                          cx_older=cx_older,
                                          hydro_old=hydro_old,
                                          hydro_older=hydro_older,
                                          rad_old = rad_old,
                                          rad_older=rad_older)

       # get the modified scattering cross sections
       cx_mod_prev = newton_handler.getEffectiveOpacities(dt)

       planckian_new = newton_handler.evalPlanckianImplicit(dt=dt, hydro_star= hydro_old)

       # evaluate transient source, including linearized planckian
       rad_new = takeRadiationStep(
           mesh          = mesh,
           time_stepper  = time_stepper,
           problem_type  = problem_type,
           planckian_new = planckian_new,
           dt            = dt,
           psi_left      = psi_left,
           psi_right     = psi_right,
           cx_old        = cx_old,
           cx_older      = cx_older,
           cx_new        = cx_mod_prev,
           rad_old       = rad_old,
           rad_older     = rad_older,
           hydro_star    = hydro_old,
           hydro_old     = hydro_old,
           hydro_older   = hydro_older)

       # update internal energy
       newton_handler.updateIntEnergy(rad_new.E, dt, hydro_star = hydro_old)

       # check nonlinear convergence
       hydro_new = newton_handler.getNewHydroStates()
       rel_diff = computeL2RelDiff(hydro_new, hydro_prev, aux_func=lambda x: x.e)
       print("  Iteration %d: Difference = %7.3e" % (k,rel_diff))
       if rel_diff < tol:
          print("  Nonlinear iteration converged")
          break

       # store new to prev
       hydro_prev = hydro_new

   # return new hydro and radiation
   return hydro_new, rad_new

