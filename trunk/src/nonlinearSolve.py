## @package src.nonlinearSolve
#  Provides functions for performing nonlinear solves
#

from copy import deepcopy

from takeRadiationStep import takeRadiationStep
from utilityFunctions import computeL2RelDiff, computeEffectiveOpacities,\
   updateCrossSections, computeHydroInternalEnergies
from hydroSource import updateVelocity, updateInternalEnergy, QEHandler
from radSlopesHandler import computeTotalEnergySlopes


## Performs nonlinear solve
#
#  @return new hydro and rad solutions
#
def nonlinearSolve(mesh, time_stepper, problem_type, dt, psi_left, psi_right,
   cx_old, hydro_old, hydro_star, rad_old, slopes_old, e_rad_old,
   Qpsi_new, Qmom_new, Qerg_new, Qpsi_old, Qmom_old, Qerg_old, Qpsi_older,
   Qmom_older, Qerg_older,
   rad_older=None, cx_older=None, hydro_older=None, slopes_older=None,
   e_rad_older=None, tol=1.0e-9, verbose=False):

   # assert that that older arguments were passed if using BDF2
   if time_stepper == 'BDF2':
      assert(rad_older      != None)
      assert(hydro_older    != None)
      assert(cx_older       != None)
      assert(slopes_older   != None)
      assert(e_rad_older.size != 0)

   # initialize iterates to the old quantities
   hydro_new  = deepcopy(hydro_star)
   hydro_prev = deepcopy(hydro_star)
   rad_prev   = deepcopy(rad_old)
   cx_prev    = deepcopy(cx_old)

   #Guess that e_rad previous is erad_old
   e_rad_prev = deepcopy(e_rad_old)

   # initialize convergence flag and iteration counter
   converged = False
   k = 0

   # perform nonlinear iterations:
   while not converged:

       # increment iteration counter
       k += 1

       # update velocity
       if problem_type != 'rad_mat':
          updateVelocity(
             mesh         = mesh,
             time_stepper = time_stepper,
             dt           = dt,
             hydro_star   = hydro_star,
             hydro_new    = hydro_new,
             cx_older     = cx_older,
             cx_old       = cx_old,
             cx_prev      = cx_prev,
             rad_older    = rad_older,
             rad_old      = rad_old,
             rad_prev     = rad_prev,
             hydro_older  = hydro_older,
             hydro_old    = hydro_old,
             hydro_prev   = hydro_prev,
             Qmom_new     = Qmom_new,
             Qmom_old     = Qmom_old,
             Qmom_older   = Qmom_older)

       # Compute E_slopes for use by E_star state
       # For now it is hardcoded with E_star edge internal energies
       e_star = []
       for i in xrange(len(hydro_star)):
          e_star_i = computeHydroInternalEnergies(i,hydro_star[i],slopes_old)
          e_star.append(e_star_i)

       # Compute E_slopes
       E_slopes_old = computeTotalEnergySlopes(hydro_star, slopes_old, e_star)
       print "THESE DO NOT STORE THE RIGHT SLOPES, THEY NEED TO BE RECOMPUTED BASED ON e_rad_old's and odler"
       E_slopes_star = deepcopy(E_slopes_old)
       E_slopes_older = deepcopy(E_slopes_old)

       # compute QE
       src_handler = QEHandler(mesh, time_stepper)
       QE = src_handler.computeTerm(
          cx_prev     = cx_prev,
          cx_old      = cx_old,
          cx_older    = cx_older,
          rad_prev    = rad_prev,
          rad_old     = rad_old,
          rad_older   = rad_older,
          hydro_prev  = hydro_prev,
          hydro_old   = hydro_old,
          hydro_older = hydro_older,
          slopes_old  = slopes_old,
          slopes_older = slopes_older,
          E_slopes_star = E_slopes_star,
          E_slopes_old = E_slopes_old,
          E_slopes_older = E_slopes_older,
          e_rad_old = e_rad_old,
          e_rad_older = e_rad_older,
          Qerg_new     = Qerg_new,
          Qerg_old     = Qerg_old,
          Qerg_older   = Qerg_older)

       # get the modified scattering cross sections
       cx_mod_prev = computeEffectiveOpacities(
          time_stepper = time_stepper,
          dt           = dt,
          cx_prev      = cx_prev,
          hydro_prev   = hydro_prev,
          slopes_old   = slopes_old,
          e_rad_prev = e_rad_prev)

       # perform radiation solve
       rad_new = takeRadiationStep(
           mesh          = mesh,
           time_stepper  = time_stepper,
           problem_type  = problem_type,
           dt            = dt,
           psi_left      = psi_left,
           psi_right     = psi_right,
           cx_new        = cx_mod_prev,
           cx_prev       = cx_prev,
           cx_old        = cx_old,
           cx_older      = cx_older,
           rad_prev      = rad_prev,
           rad_old       = rad_old,
           rad_older     = rad_older,
           hydro_new     = hydro_new,
           hydro_prev    = hydro_prev,
           hydro_star    = hydro_star,
           hydro_old     = hydro_old,
           hydro_older   = hydro_older,
           QE            = QE,
           slopes_old    = slopes_old,
           slopes_older  = slopes_older,
           E_slopes_star = E_slopes_star,
           E_slopes_old  = E_slopes_old,
           E_slopes_older = E_slopes_older,
           e_rad_prev = e_rad_prev, #Current guess for $C)vT^*$
           e_rad_old = e_rad_old,
           e_rad_older = e_rad_older,
           Qpsi_new      = Qpsi_new,
           Qpsi_old      = Qpsi_old,
           Qpsi_older    = Qpsi_older)

       # update internal energy
       e_rad_new = updateInternalEnergy(
          time_stepper = time_stepper,
          dt           = dt,
          QE           = QE,
          cx_prev      = cx_prev,
          rad_new      = rad_new,
          hydro_new    = hydro_new,
          hydro_prev   = hydro_prev,
          hydro_star   = hydro_star,
          slopes_old   = slopes_old,
          E_slopes_old = E_slopes_old,
          e_rad_prev   = e_rad_prev)

       # check nonlinear convergence
       # TODO: compute diff of rad solution as well to add to convergence criteria
       rel_diff = computeL2RelDiff(hydro_new, hydro_prev, aux_func=lambda x: x.e)
       if verbose:
          print("      Iteration %d: Difference = %7.3e" % (k,rel_diff))
       if rel_diff < tol:
          if verbose:
             print("      Nonlinear iteration converged")
          break

       # reset previous iteration quantities if needed
       hydro_prev = deepcopy(hydro_new)
       rad_prev   = deepcopy(rad_new)
       e_rad_prev = deepcopy(e_rad_new)
       updateCrossSections(cx_prev,hydro_new,slopes_old,e_rad_new)      

   # return new hydro and radiation
   return hydro_new, rad_new, cx_prev, e_rad_new


