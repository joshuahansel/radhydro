## @package src.transient
#  Contains functions to run transients.

from copy import deepcopy
import numpy as np
from math import sqrt

from nonlinearSolve import nonlinearSolve
from utilityFunctions import computeL2RelDiff, computeAnalyticHydroSolution
from transientSource import computeRadiationExtraneousSource
from hydroSource import computeMomentumExtraneousSource,\
   computeEnergyExtraneousSource
from takeRadiationStep import takeRadiationStep
from hydroSlopes import HydroSlopes
from musclHancock import hydroPredictor, hydroCorrectorSimon, hydroCorrectorJosh
from balanceChecker import BalanceChecker
from plotUtilities import plotHydroSolutions

## Runs transient for a radiation-only problem.
#
#  @param[in] psim_src  extraneous source function for \f$\Psi^-\f$
#  @param[in] psip_src  extraneous source function for \f$\Psi^+\f$
#
def runLinearTransient(mesh, time_stepper,
   psi_left, psi_right, cross_sects, rad_IC, psim_src, psip_src,
   dt_option='constant', dt_constant=None, t_start=0.0, t_end=1.0, verbosity=2):

   # check input arguments
   if dt_option == 'constant':
      assert dt_constant is not None, "If time step size option is chosen to \
         be 'constant', then a time step size must be provided."

   # initialize time and solutions
   t = t_start
   rad_old = rad_IC
   Qpsi_old = computeRadiationExtraneousSource(psim_src, psip_src, mesh, t_start)
   rad_older  = None
   Qpsi_older = None
   
   # transient loop
   time_index = 0
   transient_incomplete = True # boolean flag signalling end of transient
   while transient_incomplete:

       # increment time index
       time_index += 1

       # if first step, then can't use BDF2
       if time_index == 1 and time_stepper == 'BDF2':
          time_stepper_this_step = 'BE'
       else:
          time_stepper_this_step = time_stepper

       # get time step size
       if dt_option == 'constant':
          dt = dt_constant
       else:
          raise NotImplementedError('Invalid time step size option')
  
       # adjust time step size if it would overshoot the end of the transient
       if t + dt >= t_end:
          dt = t_end - t
          t = t_end
          transient_incomplete = False # signal end of transient
       else:
          t += dt

       # print each time step
       if verbosity > 0:
          print("Time step %d: t = %f -> %f:" % (time_index,t-dt,t))

       # compute new extraneous source
       Qpsi_new = computeRadiationExtraneousSource(psim_src, psip_src, mesh, t)
  
       # take radiation step
       #
       # NOTE: In this case, cross sections are assumed to be constant
       #       with respect to time because cross sections are generally
       #       functions of material properties, and there is no coupling
       #       to material physics in a radiation-only problem.
       #
       rad_new = takeRadiationStep(
          mesh          = mesh,
          time_stepper  = time_stepper_this_step,
          problem_type  = 'rad_only',
          dt            = dt,
          psi_left      = psi_left,
          psi_right     = psi_right,
          cx_older      = cross_sects,
          cx_old        = cross_sects,
          cx_new        = cross_sects,
          rad_older     = rad_older,
          rad_old       = rad_old,
          Qpsi_older    = Qpsi_older,
          Qpsi_old      = Qpsi_old,
          Qpsi_new      = Qpsi_new)

       # save older solutions
       Qpsi_older = deepcopy(Qpsi_old)
       rad_older  = deepcopy(rad_old)

       # save old solutions
       Qpsi_old   = deepcopy(Qpsi_new)
       rad_old    = deepcopy(rad_new)

   # return final solution
   return rad_new


## Runs transient for a nonlinear radiation-material problem.
#
#  @param[in] add_ext_src  flag to signal that extraneous sources are to be added
#  @param[in] psim_src  extraneous source function for \f$\Psi^-\f$
#  @param[in] psip_src  extraneous source function for \f$\Psi^+\f$
#  @param[in] mom_src   extraneous source function for the conservation
#                       of momentum equation
#  @param[in] E_src     extraneous source function for the conservation
#                       of total energy equation
#
def runNonlinearTransient(mesh, problem_type,
   psi_left, psi_right, cross_sects, rad_IC, hydro_IC, hydro_BC,
   psim_src=None, psip_src=None, mom_src=None, E_src=None,
   time_stepper='BE', dt_option='constant', dt_constant=None, CFL=0.5,
   slope_limiter="vanleer", t_start=0.0, t_end=1.0, use_2_cycles=False,
   rho_f=None,u_f=None,E_f=None,gamma_value=None,cv_value=None,
   verbosity=2, check_balance=False):

   # check input arguments
   if dt_option == 'constant':
      assert dt_constant != None, "If time step size option is chosen to \
         be 'constant', then a time step size must be provided."

   # initialize old quantities
   t_old = t_start
   cx_old = deepcopy(cross_sects)
   rad_old = deepcopy(rad_IC)
   hydro_old = deepcopy(hydro_IC)
   Qpsi_old, Qmom_old, Qerg_old = computeExtraneousSources(
      psim_src, psip_src, mom_src, E_src, mesh, t_start)

   #Just guess e_rad old is hydro initial conditionsstuff
   e_rad_old = np.array([(i.e, i.e) for i in hydro_old])
   

   # set older quantities to nothing; these shouldn't exist yet
   cx_older       = None
   rad_older      = None
   hydro_older    = None
   slopes_older   = None
   e_rad_older = None
   Qpsi_older     = None
   Qmom_older     = None
   Qerg_older     = None
   
   # transient loop
   time_index = 0
   transient_incomplete = True # boolean flag signalling end of transient
   while transient_incomplete:

       # increment time index
       time_index += 1

       # if first step, then can't use BDF2
       if time_index == 1 and time_stepper == 'BDF2':
          time_stepper_this_step = 'BE'
       else:
          time_stepper_this_step = time_stepper

       # get time step size
       if dt_option == 'constant':
          # constant time step size
          dt = dt_constant
       elif dt_option == 'CFL':
          # compute time step size according to CFL condition
          sound_speed = [sqrt(i.p * i.gamma / i.rho) + abs(i.u) for i in hydro_old]
          dt_vals = [CFL*(mesh.elements[i].dx)/sound_speed[i]
             for i in xrange(len(hydro_old))]
          dt = min(dt_vals)

          # if using 2 cycles, then twice the time step size may be taken
          if use_2_cycles:
             dt *= 2.0
       else:
          raise NotImplementedError('Invalid time step size option')
  
       # adjust time step size if it would overshoot the end of the transient
       if t_old + dt >= t_end:
          dt = t_end - t_old
          t_new = t_end
          transient_incomplete = False # signal end of transient
       else:
          t_new = t_old + dt

       # print each time step
       print("Time step %d: t = %f -> %f:" % (time_index,t_old,t_new))
  
       # take time step
       if problem_type == 'rad_mat':

          #Force balance checker vars to zeros
          hydro_F_left = {"rho":0.,"erg":0.,"mom":0.}
          hydro_F_right = {"rho":0.,"erg":0.,"mom":0.}

          if time_stepper == 'TRBDF2':

              # take a half time step with CN
              hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
              Qpsi_new, Qmom_new, Qerg_new, src_totals_cycle1 =\
                 takeTimeStepRadiationMaterial(
                 mesh         = mesh,
                 time_stepper = 'CN',
                 dt           = 0.5*dt,
                 psi_left     = psi_left,
                 psi_right    = psi_right,
                 hydro_BC     = hydro_BC,
                 cx_old       = cx_old,
                 hydro_old    = hydro_old,
                 rad_old      = rad_old,
                 e_rad_old = e_rad_old,
                 psim_src     = psim_src,
                 psip_src     = psip_src,
                 mom_src      = mom_src,
                 E_src        = E_src,
                 t_old        = t_old,
                 Qpsi_old     = Qpsi_old,
                 Qmom_old     = Qmom_old,
                 slope_limiter= slope_limiter,
                 Qerg_old     = Qerg_old)

              # take a half time step with BDF2
              hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
              Qpsi_new, Qmom_new, Qerg_new, src_totals_cycle2 =\
                 takeTimeStepRadiationMaterial(
                 mesh         = mesh,
                 time_stepper = 'BDF2',
                 dt           = dt,
                 psi_left     = psi_left,
                 psi_right    = psi_right,
                 hydro_BC     = hydro_BC,
                 cx_old       = cx_new,
                 cx_older     = deepcopy(cx_old),
                 hydro_old    = hydro_new,
                 hydro_older  = deepcopy(hydro_old),
                 rad_old      = rad_new,
                 rad_older    = deepcopy(rad_old),
                 slopes_older = deepcopy(slopes_old),
                 e_rad_old = e_rad_new,
                 e_rad_older = e_rad_old,
                 slope_limiter= slope_limiter,
                 psim_src     = psim_src,
                 psip_src     = psip_src,
                 mom_src      = mom_src,
                 E_src        = E_src,
                 t_old        = t_old,
                 Qpsi_old     = Qpsi_old,
                 Qmom_old     = Qmom_old,
                 Qerg_old     = Qerg_old,
                 Qpsi_older   = deepcopy(Qpsi_old),
                 Qmom_older   = deepcopy(Qmom_old),
                 Qerg_older   = deepcopy(Qerg_old))

              # add up source totals for each cycle to total for whole time step
              src_totals = dict()
              for key in src_totals_cycle1:
                 src_totals[key] = src_totals_cycle1[key] + src_totals_cycle2[key]
             
          else: # assume it's a single step method

              # take time step without MUSCL-Hancock
              hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
              Qpsi_new, Qmom_new, Qerg_new, src_totals =\
                 takeTimeStepRadiationMaterial(
                 mesh         = mesh,
                 time_stepper = time_stepper_this_step,
                 dt           = dt,
                 psi_left     = psi_left,
                 psi_right    = psi_right,
                 hydro_BC     = hydro_BC,
                 cx_old       = cx_old,
                 cx_older     = cx_older,
                 hydro_old    = hydro_old,
                 hydro_older  = hydro_older,
                 rad_old      = rad_old,
                 rad_older    = rad_older,
                 slopes_older = slopes_older,
                 e_rad_old = e_rad_old,
                 e_rad_older = e_rad_older,
                 slope_limiter = slope_limiter,
                 psim_src     = psim_src,
                 psip_src     = psip_src,
                 mom_src      = mom_src,
                 E_src        = E_src,
                 t_old        = t_old,
                 Qpsi_old     = Qpsi_old,
                 Qmom_old     = Qmom_old,
                 Qerg_old     = Qerg_old,
                 Qpsi_older   = Qpsi_older,
                 Qmom_older   = Qmom_older,
                 Qerg_older   = Qerg_older)

       else: # problem_type == 'rad_hydro'

          # if user chose to use the 2-cycle scheme
          if use_2_cycles:

             print("  Cycle 1:")

             # take time step with MUSCL-Hancock
             hydro_half, rad_half, cx_half, slopes_old, e_rad_half,\
             Qpsi_half, Qmom_half, Qerg_half, hydro_F_left, hydro_F_right,\
             src_totals_cycle1 =\
                takeTimeStepMUSCLHancock(
                mesh           = mesh,
                dt             = 0.5*dt, 
                psi_left       = psi_left,
                psi_right      = psi_right,
                hydro_BC       = hydro_BC,
                slope_limiter  = slope_limiter,
                cx_old         = cx_old,
                cx_older       = cx_older,
                hydro_old      = hydro_old,
                hydro_older    = hydro_older,
                rad_old        = rad_old,
                rad_older      = rad_older,
                slopes_older   = slopes_older,
                e_rad_old   = e_rad_old,
                e_rad_older = e_rad_older,
                time_stepper_predictor='CN',
                time_stepper_corrector='CN',
                psim_src     = psim_src,
                psip_src     = psip_src,
                mom_src      = mom_src,
                E_src        = E_src,
                t_old        = t_old,
                Qpsi_old     = Qpsi_old,
                Qmom_old     = Qmom_old,
                Qerg_old     = Qerg_old,
                Qpsi_older   = Qpsi_older,
                Qmom_older   = Qmom_older,
                Qerg_older   = Qerg_older,
                verbosity    = verbosity,
                rho_f = rho_f, u_f = u_f, E_f = E_f,
                gamma_value = gamma_value,
                cv_value=cv_value
            )

             print("  Cycle 2:")

             # take time step with MUSCL-Hancock
             hydro_new, rad_new, cx_new, slopes_half, e_rad_new,\
             Qpsi_new, Qmom_new, Qerg_new, hydro_F_left, hydro_F_right,\
             src_totals_cycle2 =\
                takeTimeStepMUSCLHancock(
                mesh           = mesh,
                dt             = 0.5*dt, 
                psi_left       = psi_left,
                psi_right      = psi_right,
                hydro_BC       = hydro_BC,
                slope_limiter  = slope_limiter,
                cx_old         = cx_half,
                cx_older       = cx_old,
                hydro_old      = hydro_half,
                hydro_older    = hydro_old,
                rad_old        = rad_half,
                rad_older      = rad_old,
                slopes_older   = slopes_old,
                e_rad_old   = e_rad_half,
                e_rad_older = e_rad_old,
                time_stepper_predictor='CN',
                time_stepper_corrector='BDF2',
                psim_src     = psim_src,
                psip_src     = psip_src,
                mom_src      = mom_src,
                E_src        = E_src,
                t_old        = t_old + 0.5*dt,
                Qpsi_old     = Qpsi_half,
                Qmom_old     = Qmom_half,
                Qerg_old     = Qerg_half,
                Qpsi_older   = Qpsi_old,
                Qmom_older   = Qmom_old,
                Qerg_older   = Qerg_old,
                verbosity    = verbosity,
                rho_f = rho_f, u_f = u_f, E_f = E_f,
                gamma_value = gamma_value,
                cv_value=cv_value)

             # add up source totals for each cycle to total for whole time step
             src_totals = dict()
             for key in src_totals_cycle1:
                src_totals[key] = src_totals_cycle1[key] + src_totals_cycle2[key]
             
          else: # use only 1 cycle

             # for first step, can't use BDF2; use CN instead
             if time_index == 1:
                time_stepper_corrector = 'CN'
             else:
                time_stepper_corrector = 'BDF2'

             # take time step with MUSCL-Hancock
             hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
             Qpsi_new, Qmom_new, Qerg_new, hydro_F_left, hydro_F_right,\
             src_totals =\
                takeTimeStepMUSCLHancock(
                mesh           = mesh,
                dt             = dt, 
                psi_left       = psi_left,
                psi_right      = psi_right,
                hydro_BC       = hydro_BC,
                slope_limiter  = slope_limiter,
                cx_old         = cx_old,
                cx_older       = cx_older,
                hydro_old      = hydro_old,
                hydro_older    = hydro_older,
                rad_old        = rad_old,
                rad_older      = rad_older,
                slopes_older   = slopes_older,
                e_rad_old   = e_rad_old,
                e_rad_older = e_rad_older,
                time_stepper_predictor='CN',
                time_stepper_corrector=time_stepper_corrector,
                psim_src     = psim_src,
                psip_src     = psip_src,
                mom_src      = mom_src,
                E_src        = E_src,
                t_old        = t_old,
                Qpsi_old     = Qpsi_old,
                Qmom_old     = Qmom_old,
                Qerg_old     = Qerg_old,
                Qpsi_older   = Qpsi_older,
                Qmom_older   = Qmom_older,
                Qerg_older   = Qerg_older,
                verbosity    = verbosity,
                rho_f = rho_f, u_f = u_f, E_f = E_f,
                gamma_value = gamma_value,
                cv_value=cv_value)

       # compute balance
       if check_balance:
          bal = BalanceChecker(mesh, problem_type, time_stepper, dt)
          bal.computeBalance(psi_left, psi_right, hydro_old,
                 hydro_new, rad_old, rad_new, hydro_F_right=hydro_F_right,
                 hydro_F_left=hydro_F_left, src_totals=src_totals, 
                 cx_new=cx_new,write=True)

       # save older solutions
       cx_older  = deepcopy(cx_old)
       rad_older = deepcopy(rad_old)
       hydro_older = deepcopy(hydro_old)
       slopes_older = deepcopy(slopes_old)
       e_rad_older = deepcopy(e_rad_old)
       Qpsi_older = deepcopy(Qpsi_old)
       Qmom_older = deepcopy(Qmom_old)
       Qerg_older = deepcopy(Qerg_old)

       # save old solutions
       t_old = t_new
       cx_old  = deepcopy(cx_new)
       rad_old = deepcopy(rad_new)
       hydro_old = deepcopy(hydro_new)
       e_rad_old = deepcopy(e_rad_new)
       Qpsi_old = deepcopy(Qpsi_new)
       Qmom_old = deepcopy(Qmom_new)
       Qerg_old = deepcopy(Qerg_new)

   # return final solutions
   return rad_new, hydro_new


## Takes time step without any MUSCL-Hancock.
#
#  This should only be called if the problem type is 'rad_mat'.
#
def takeTimeStepRadiationMaterial(mesh, time_stepper, dt, psi_left, psi_right,
   cx_old=None, cx_older=None, hydro_old=None, hydro_older=None, rad_old=None, rad_older=None,
   hydro_BC=None, slopes_older=None, e_rad_old=None, e_rad_older=None,
   psim_src=None, psip_src=None, mom_src=None, E_src=None, t_old=None, Qpsi_old=None, Qmom_old=None, Qerg_old=None,
   Qpsi_older=None, Qmom_older=None, Qerg_older=None, slope_limiter=None):

       # compute new extraneous sources
       Qpsi_new, Qmom_new, Qerg_new = computeExtraneousSources(
          psim_src, psip_src, mom_src, E_src, mesh, t_old+dt)

       # update hydro BC
       hydro_BC.update(states=hydro_old, t=t_old)

       # compute slopes
       slopes_old = HydroSlopes(hydro_old, bc=hydro_BC, limiter=slope_limiter)

       # if there is no material motion, then the homogeneous hydro solution
       # should be equal to the old hydro solution
       hydro_star = deepcopy(hydro_old)

       # perform nonlinear solve
       hydro_new, rad_new, cx_new, e_rad_new = nonlinearSolve(
          mesh         = mesh,
          time_stepper = time_stepper,
          problem_type = 'rad_mat',
          dt           = dt,
          psi_left     = psi_left,
          psi_right    = psi_right,
          cx_old       = cx_old,
          cx_older     = cx_older,
          hydro_old    = hydro_old,
          hydro_older  = hydro_older,
          hydro_star   = hydro_star,
          rad_old      = rad_old,
          rad_older    = rad_older,
          slopes_old   = slopes_old,
          slopes_older = slopes_older,
          e_rad_old    = e_rad_old,
          e_rad_older  = e_rad_older,
          Qpsi_new     = Qpsi_new,
          Qmom_new     = Qmom_new,
          Qerg_new     = Qerg_new,
          Qpsi_old     = Qpsi_old,
          Qmom_old     = Qmom_old,
          Qerg_old     = Qerg_old,
          Qpsi_older   = Qpsi_older,
          Qmom_older   = Qmom_older,
          Qerg_older   = Qerg_older)

       # add up sources for entire time step for balance checker
       src_totals = {}
       vol = mesh.getElement(0).dx
       src_totals["mom"] = sum([vol*dt*i for i in Qmom_new])
       src_totals["erg"] = sum([vol*dt*0.5*(i[0]+i[1]) for i in Qerg_new])

       return hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
          Qpsi_new, Qmom_new, Qerg_new, src_totals


## Takes time step with MUSCL-Hancock.
#
#  This should only be called if the problem type is 'rad_hydro'.
#
def takeTimeStepMUSCLHancock(mesh, dt, psi_left, psi_right,
   cx_old, cx_older, hydro_old, hydro_older, rad_old, rad_older,
   hydro_BC, slope_limiter, slopes_older, e_rad_old, e_rad_older,
   psim_src, psip_src, mom_src, E_src, t_old,
   Qpsi_old, Qmom_old, Qerg_old, Qpsi_older, Qmom_older, Qerg_older,
   time_stepper_predictor='CN', time_stepper_corrector='BDF2',verbosity=2,
   rho_f=None,u_f=None,E_f=None,gamma_value=None,cv_value=None):

   debug_mode = False

   # assert that BDF2 was not chosen for the predictor time-stepper
   assert time_stepper_predictor != 'BDF2', 'BDF2 cannot be used in\
      the predictor step.'

   if debug_mode:

      hydro_exact = computeAnalyticHydroSolution(mesh, t=t_old,
            rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)

      plotHydroSolutions(mesh, hydro_old, x_exact=mesh.getCellCenters(),
          exact=hydro_exact)
    
   if verbosity > 1:
      print "    Predictor step:"

   # Force everything to BE
   print "FORCING BE EVERYWHERE DEBUG"
   time_stepper_predictor = 'BE'
   time_stepper_corrector = 'BE'

   # update hydro BC
   hydro_BC.update(states=hydro_old, t=t_old)

   # compute slopes
   slopes_old = HydroSlopes(hydro_old, bc=hydro_BC, limiter=slope_limiter)

   # perform predictor step of MUSCL-Hancock
   hydro_star = hydroPredictor(mesh, hydro_old, slopes_old, dt)


   if debug_mode:
    #  print "hydro_old:"
   #   for i in hydro_old:
   #      print i

      print "hydro_star predictor:"
  #    for i in hydro_star:
 #        print i
      plotHydroSolutions(mesh, hydro_star, x_exact=mesh.getCellCenters(),
                exact=None)

   # compute new extraneous sources
   Qpsi_half, Qmom_half, Qerg_half = computeExtraneousSources(
      psim_src, psip_src, mom_src, E_src, mesh, t_old+0.5*dt)

   # perform nonlinear solve
   hydro_half, rad_half, cx_half, e_rad_half = nonlinearSolve(
      mesh         = mesh,
      time_stepper = time_stepper_predictor,
      problem_type = 'rad_hydro',
      dt           = 0.5*dt,
      psi_left     = psi_left,
      psi_right    = psi_right,
      cx_old       = cx_old,
      hydro_old    = hydro_old,
      hydro_star   = hydro_star,
      rad_old      = rad_old,
      slopes_old   = slopes_old,
      e_rad_old    = e_rad_old,
      Qpsi_new     = Qpsi_half,
      Qmom_new     = Qmom_half,
      Qerg_new     = Qerg_half,
      Qpsi_old     = Qpsi_old,
      Qmom_old     = Qmom_old,
      Qerg_old     = Qerg_old,
      Qpsi_older   = Qpsi_older, # this is a dummy argument
      Qmom_older   = Qmom_older, # this is a dummy argument
      Qerg_older   = Qerg_older, # this is a dummy argument
      verbosity    = verbosity)



   if debug_mode:
      print "hydro_half:"
#      for i in hydro_half:
#         print i
      hydro_exact = computeAnalyticHydroSolution(mesh, t=t_old+0.5*dt,
            rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)

      plotHydroSolutions(mesh, hydro_half, x_exact=mesh.getCellCenters(),
        exact=hydro_exact)

   if verbosity > 1:
      print "    Corrector step:"

   # update hydro BC
   hydro_BC.update(states=hydro_half, t=t_old+0.5*dt, edge_value=True)
   #hydro_BC.update(states=hydro_half, t=t_old+0.5*dt, edge_value=False)

   # perform corrector step of MUSCL-Hancock
   hydro_star, hydro_F_left, hydro_F_right = hydroCorrectorSimon(
   #hydro_star, hydro_F_left, hydro_F_right = hydroCorrectorJosh(
      mesh, hydro_old, hydro_half, slopes_old, dt, bc=hydro_BC)

   if debug_mode:
      print "hydro_star corrector:"
 #     for i in hydro_star:
 #        print i

      plotHydroSolutions(mesh, hydro_star, x_exact=mesh.getCellCenters(),
                exact=None)

   # compute new extraneous sources
   Qpsi_new, Qmom_new, Qerg_new = computeExtraneousSources(
      psim_src, psip_src, mom_src, E_src, mesh, t_old+dt)

   # perform nonlinear solve
   hydro_new, rad_new, cx_new, e_rad_new = nonlinearSolve(
      mesh         = mesh,
      time_stepper = time_stepper_corrector,
      problem_type = 'rad_hydro',
      dt           = dt,
      psi_left     = psi_left,
      psi_right    = psi_right,
      cx_old       = cx_old,
      cx_older     = cx_older,
      hydro_old    = hydro_old,
      hydro_older  = hydro_older,
      hydro_star   = hydro_star,
      rad_old      = rad_old,
      rad_older    = rad_older,
      slopes_old   = slopes_old,
      slopes_older = slopes_older,
      e_rad_old = e_rad_old,
      e_rad_older = e_rad_older,
      Qpsi_new     = Qpsi_new,
      Qmom_new     = Qmom_new,
      Qerg_new     = Qerg_new,
      Qpsi_old     = Qpsi_old,
      Qmom_old     = Qmom_old,
      Qerg_old     = Qerg_old,
      Qpsi_older   = Qpsi_older,
      Qmom_older   = Qmom_older,
      Qerg_older   = Qerg_older,
      verbosity    = verbosity)

   if debug_mode:
       hydro_exact = computeAnalyticHydroSolution(mesh, t=t_old+dt,
            rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)

    
       plotHydroSolutions(mesh, hydro_new, x_exact=mesh.getCellCenters(),
            exact=hydro_exact)

   # add up sources for entire time step for balance checker
   src_totals = {}
   vol = mesh.getElement(0).dx
   src_totals["mom"] = sum([vol*dt*i for i in Qmom_new])
   src_totals["erg"] = sum([vol*dt*0.5*(i[0]+i[1]) for i in Qerg_new])

   if verbosity > 1:
      print ""

   if debug_mode:
      print "hydro_new:"
      for i in hydro_new:
         print i

   return hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
      Qpsi_new, Qmom_new, Qerg_new, hydro_F_left, hydro_F_right,\
      src_totals


## Computes all extraneous sources at time \f$t\f$
#
#  @param[in] psim_src
#  @param[in] psip_src
#  @param[in] mom_src
#  @param[in] E_src
#  @param[in] mesh
#  @param[in] t
#
#  @return extraneous source vectors evaluated at \f$t\f$:
#    \f$Q^{ext,\pm}\f$, \f$Q^{ext,\rho u}\f$, \f$Q^{ext,E}\f$
#
def computeExtraneousSources(psim_src, psip_src, mom_src, E_src, mesh, t):

   # compute radiation extraneous source
   if psim_src != None and psip_src != None:
      Qpsi = computeRadiationExtraneousSource(psim_src, psip_src, mesh, t)
   else:
      Qpsi = np.zeros(mesh.n_elems*4)

   # compute momentum extraneous source
   if mom_src != None:
      Qmom = computeMomentumExtraneousSource(mom_src, mesh, t)
   else:
      Qmom = np.zeros(mesh.n_elems)

   # compute energy extraneous source
   if E_src != None:
      Qerg = computeEnergyExtraneousSource(E_src, mesh, t)
   else:
      Qerg = [(0.0,0.0) for i in xrange(mesh.n_elems)]

   return Qpsi, Qmom, Qerg


