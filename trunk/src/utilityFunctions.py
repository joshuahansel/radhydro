## @package src.utilityFunctions
#  Contains helper functions that do not belong in any particular class.

from math import log, sqrt
import numpy as np
import globalConstants as GC
from crossXInterface import CrossXInterface
from hydroState import HydroState
from radiation import Radiation
from scipy.integrate import quad

QUAD_REL_TOL = 1.0E-10

#-----------------------------------------------------------------------------------
## Converge f_L and f_R to f_a and f_x
def EdgToMom(f_l, f_r):
    
    # Based on f(x) = f_a + 2/h*f_x(x - x_i)
    f_a = 0.5*(f_l + f_r)
    f_x = 0.5*(f_r - f_l)

    return f_a, f_x


#-----------------------------------------------------------------------------------
## Converge f_x and f_a to f_l and f_r
def MomToEdg(f_a, f_x):
    
    # Based on f(x) = f_a + 2/h*f_x(x - x_i)
    return (f_a - f_x), (f_a + f_x)

#-----------------------------------------------------------------------------------
## Index function for 1-D LD S-2 DOF handler. This gives the global index in array
#
#  @param[in] i     cell index, from 1 to n-1, where n is number of cells
#  @param[in] side  string, either "L" or "R", corresponding to left or right dof
#  @param[in] dir   string, either "-" or "+", corresponding to - or + direction
#
#  @return    global dof getIndex
#
def getIndex(i, side, dir):

    side_shift = {"L" : 0, "R" : 2}
    dir_shift  = {"-" : 0, "+" : 1}
    return 4*i + side_shift[side] + dir_shift[dir]

#-----------------------------------------------------------------------------------
## Local index function for S-2 DOF. This gives the local index in array of for the 
#  different DOF, essentially no offset

def getLocalIndex(side, dir):

    return getIndex(0, side, dir)

#-----------------------------------------------------------------------------------
## Computes convergence rates
#
#  @param[in] dx        list of mesh sizes or time step sizes for each cycle
#  @param[in] err       list of errors for each cycle
#
#  @return  list of convergence rates. The size of this list will be the number
#           of cycles minus one.
#
def computeConvergenceRates(dx,err):

   # determine number of refinement cycles from length of lists
   n_cycles = len(dx)

   # initialize list of rates
   rates = list()

   # loop over cycles
   for cycle in xrange(n_cycles):
      # compute convergence rate
      if cycle > 0:
         rates.append(log(err[cycle]/err[cycle-1]) / log(dx[cycle]/dx[cycle-1]))

   return rates

#-----------------------------------------------------------------------------------
## Computes convergence rates for hydro quantities.
#
#  @param[in] dx   list of mesh sizes or time step sizes for each cycle
#  @param[in] err  list of dictionaries of errors for each cycle and quantity
#
#  @return  list of dictionaries of convergence rates for each quantity.
#     The size of this list will be the number of cycles minus one.
#
def computeHydroConvergenceRates(dx,err):

   # determine number of refinement cycles from length of lists
   n_cycles = len(dx)

   # initialize list of rates
   rates = list()

   # loop over cycles
   for cycle in xrange(n_cycles):
      # compute convergence rate
      if cycle > 0:
         # loop over keys of dictionary to compute rate for each quantity
         rate_dict = dict()
         for key in err[cycle]:
            rate_dict[key] = log(err[cycle][key]/err[cycle-1][key]) / \
               log(dx[cycle]/dx[cycle-1])
         # add rate dictionary to list
         rates.append(rate_dict)

   return rates

#-----------------------------------------------------------------------------------
## Compute convergence rates for radiation variables
#
#  @param[in] dx   list of mesh sizes or time step sizes for each cycle
#  @param[in] err  list of dictionaries of errors for each cycle and quantity
#
#  @return  list of dictionaries of convergence rates for each quantity.
#     The size of this list will be the number of cycles minus one.
#
def computeRadiationConvergenceRates(dx,err):

   # determine number of refinement cycles from length of lists
   n_cycles = len(dx)

   # initialize list of rates
   rates = list()

   # loop over cycles
   for cycle in xrange(n_cycles):
      # compute convergence rate
      if cycle > 0:
         # loop over keys of dictionary to compute rate for each quantity
         rate_dict = dict()
         for key in err[cycle]:
            rate_dict[key] = log(err[cycle][key]/err[cycle-1][key]) / \
               log(dx[cycle]/dx[cycle-1])
         # add rate dictionary to list
         rates.append(rate_dict)

   return rates
#-----------------------------------------------------------------------------------
## Prints convergence table and convergence rates
#
#  @param[in] dx        list of mesh sizes or time step sizes for each cycle
#  @param[in] err       list of errors for each cycle
#  @param[in] rates     list of convergence rates. The size of this list will
#                       be the number of cycles minus one. If this argument
#                       is not provided, rates are computed in this function.
#  @param[in] dx_desc   string description for the size quantity, e.g.,
#                       'dx' or 'dt'
#  @param[in] err_desc  string description for the error, e.g., 'L1' or 'L2'
#
def printConvergenceTable(dx,err,rates=None,dx_desc='size',err_desc='err',
   quantity_desc=''):

   # compute rates if they were not provided
   if rates is None:
      rates = computeConvergenceRates(dx,err)

   # determine number of refinement cycles from length of lists
   n_cycles = len(dx)

   # create title
   if quantity_desc == '':
      title = '\nConvergence:'
   else:
      title = '\n' + quantity_desc + ' Convergence:'

   # print header
   print(title)
   print('-------------------------------')
   print('%11s %11s    Rate' % (dx_desc,err_desc))
   print('-------------------------------')

   # loop over cycles
   for cycle in xrange(n_cycles):
      # compute convergence rate
      if cycle > 0:
         rate_string = '%7.3f' % rates[cycle-1]
      else:
         rate_string = '-'

      # print line to convergence table
      print('%11.3e %11.3e %7s' % (dx[cycle],err[cycle],rate_string))
   print('-------------------------------')
   print('\n')


## Prints convergence table and convergence rates for each hydro quantity
#
#  @param[in] dx        list of mesh sizes or time step sizes for each cycle
#  @param[in] err       list of dictionaries of errors for each cycle and
#                       each quantity
#  @param[in] rates     list of dictionaries of convergence rates for each
#                       quantity. The size of this list will
#                       be the number of cycles minus one. If this argument
#                       is not provided, rates are computed in this function.
#  @param[in] dx_desc   string description for the size quantity, e.g.,
#                       'dx' or 'dt'
#  @param[in] err_desc  string description for the error, e.g., 'L1' or 'L2'
#
def printHydroConvergenceTable(dx,err,rates=None,dx_desc='size',err_desc='err'):

   # compute rates if they were not provided
   if rates is None:
      rates = computeHydroConvergenceRates(dx,err)

   # loop over each quantity
   for key in err[0]:

      # extract an error list for the quantity
      err_quantity = list()
      for i in xrange(len(err)):
         err_quantity.append(err[i][key])

      # extract a rate list for the quantity
      rates_quantity = list()
      for i in xrange(len(rates)):
         rates_quantity.append(rates[i][key])

      # call convergence table function
      printConvergenceTable(dx,err_quantity,rates_quantity,dx_desc,err_desc,
         quantity_desc=key)


## Function to compute the discrete \f$L^1\f$ norm of an array \f$\mathbf{y}\f$
#  of 2-tuples: \f$\|\mathbf{y}\|_1 = \sum\limits_i |y_{i,L}| + |y_{i,R}|\f$
#
#  @param[in] values  array \f$\mathbf{y}\f$ of 2-tuples:
#                     \f$y_i = (y_{i,L},y_{i,R})\f$
#
#  @return  the discrete \f$L^1\f$ norm \f$\|y\|_1\f$
#
def computeDiscreteL1Norm(values):

   # number of tuples in array
   n = len(values)

   # initialize norm to zero
   norm = 0.0

   # loop over tuples
   for i in xrange(n):
      norm += abs(values[i][0]) + abs(values[i][1])

   return norm

## Function to compute the error for the hydro solution.
#
def computeHydroError(hydro, hydro_exact):
   err = computeL2RelDiff(hydro, hydro_exact, aux_func=lambda x: x.e)
   return err

## Function to compute the L-2 error for the hydro solution
#  for a number of different quantities.   Optionally
#  you can add in the radiation stuff
#
def computeHydroL2Error(hydro, hydro_exact, rad=None, rad_exact=None):


   print "WARNING: there is an assumption of uniform mesh spacing in this computation\n"
   #WARNING: basically the way we compute the error is we compute the exact cell
   #averages and just compare the L2 difference between our computed averages and
   #the exact cell averages (at the end time).  This is not the same as computing a true L2 error
   #over space. 


   # dictionary of quantities to their function
   funcs = {'rho':   lambda state: state.rho,
            'rho u': lambda state: state.getConservativeVariables()[1],
            'E':     lambda state: state.getConservativeVariables()[2],
            'u':     lambda state: state.u,
            'p':     lambda state: state.p,
            'e':     lambda state: state.e}

   # compute error for each entry in dictionary
   err = dict()
   for key in funcs:
      err[key] = computeL2RelDiff(hydro, hydro_exact, aux_func=funcs[key])

   if rad != None:
       err['Er'] = computeL2RelDiff(rad.E, rad_exact.E, aux_func=lambda x: 0.5*(x[0]+x[1]))
       err['Fr'] = computeL2RelDiff(rad.F, rad_exact.F, aux_func=lambda x: 0.5*(x[0]+x[1]))

   return err

## Function to compute the L2 integral based on relative error of the larger 
#  of the two values. Passed in values are 2-tuples. Average of the L and R values is
#  used to computed average
#
#  @param[in] values_1 array \f$\mathbf{y}\f$ of 2-tuples:
#                     \f$y_i = (y_{i,L},y_{i,R})\f$
#  @param[in] values_2 array \f$\mathbf{y}\f$ of 2-tuples:
#                 \f$y_i = (y_{i,L},y_{i,R})\f$
#  @param[in] aux_func optional function to apply to the values in tuple
#
#  @return  the discrete \f$L^2\f$ norm \f$\|y\|_1\f$
#
def computeL2RelDiffTuples(values1, values2, aux_func=None):

   #apply aux_func, if None default to just the value
   if aux_func == None:
       aux_func = lambda x: x
   f = aux_func

   # get averages
   avg1 = np.array([0.5*(f(i[0])+f(i[1])) for i in values1])
   avg2 = np.array([0.5*(f(i[0])+f(i[1])) for i in values2])

   #compute norms
   norm1 = np.linalg.norm(avg1)
   norm2 = np.linalg.norm(avg2)
   norm_diff = np.linalg.norm(avg1 - avg2)

   return norm_diff/(max(norm1,norm2))

## Function to compute the L2 integral based on relative error of the larger 
#  of the two values. Passed in values are arrays.
#
#  @param[in] values_1 array \f$\mathbf{y}\f$
#  @param[in] values_2 array \f$\mathbf{y}\f$
#  @param[in] aux_func optional function to apply to the values
#
#  @return  the discrete \f$L^2\f$ norm \f$\|y\|_2\f$
#
def computeL2RelDiff(values1, values2, aux_func=None):

   #apply aux_func, if None default to just the value
   if aux_func == None:
       aux_func = lambda x: x
   f = aux_func

   # get values
   vals1 = np.array([f(i) for i in values1])
   vals2 = np.array([f(i) for i in values2])

   #compute norm of the first value
   norm1 = np.linalg.norm(vals1)
   norm_diff = np.linalg.norm(vals1 - vals2)

   return norm_diff/(norm1)

## Computes effective scattering fraction \f$\nu^k\f$ in linearization
#
#  @param[in] T      Previous iteration temperature \f$T^k\f$
#  @param[in] sig_a  Previous iteration absorption cross section \f$\sigma_a^k\f$
#  @param[in] rho    New density \f$\rho^{n+1}\f$
#  @param[in] spec_heat   Previous iteration specific heat \f$c_v^k\f$
#  @param[in] dt          time step size \f$\Delta t\f$
#  @param[in] scale       coefficient corresponding to time-stepper \f$\gamma\f$
#
def getNu(T, sig_a, rho, spec_heat, dt, scale):

    ## compute \f$c\Delta t\f$
    c_dt = GC.SPD_OF_LGT*dt

    ## compute \f$\beta^k\f$
    beta = 4.*GC.RAD_CONSTANT * T * T * T / spec_heat

    # Evaluate numerator
    num  = scale*sig_a*c_dt*beta/rho

    return num/(1. + num)

#--------------------------------------------------------------------------------
## Computes effective cross sections for linearization. 
#  
#  The effective re-emission source is included as a scattering cross section
#  given by
#  \f[
#      \tilde{\sigma_s} = \sigma_s + \nu\sigma_a(T^k),
#  \f]
#  where \f$\nu\f$ is the effective scattering ratio, which depends on the
#  temporal discretization.
#
#
def computeEffectiveOpacities(time_stepper, dt, cx_prev, hydro_prev,
    slopes_old, e_rad_prev):

    # get coefficient corresponding to time-stepper
    scales = {"CN":0.5, "BE":1., "BDF2":2./3.}
    scale = scales[time_stepper]

    cx_effective = list()

    # loop over cells:
    for i in xrange(len(cx_prev)):

        cx_i = list()

        # get hydro state and specific heat
        state = hydro_prev[i]
        spec_heat = state.spec_heat

        # get density and temperature at edges
        rho = computeEdgeDensities(i, state, slopes_old)
        T   = computeEdgeTemperatures(state.spec_heat, e_rad_prev[i])

        # loop over edges
        for x in range(2): 

            # get cross sections
            sig_a = cx_prev[i][x].sig_a
            sig_s = cx_prev[i][x].sig_s

            # compute effective scattering ratio
            nu = getNu(T[x], sig_a, rho[x], spec_heat, dt, scale)
        
            #Create new FIXED cross section instance. No need to add scale term
            #here because it will be included in scattering source term
            sig_s_effective = nu*sig_a + sig_s
            cx_i.append( CrossXInterface(sig_s_effective, sig_s+sig_a) )

        cx_effective.append(tuple(cx_i))

    return cx_effective


## Computes edge densities for a cell given hydro state and slopes
#
#  @param[in] i       cell index
#  @param[in] state   average hydro state for cell \f$i\f$
#  @param[in] slopes  HydroSlopes object
#
#  @return \f$(\rho_{i,L},\rho_{i,R})\f$
#
def computeEdgeDensities(i, state, slopes):

   # compute edge velocities
   rhoL = state.rho - 0.5*slopes.rho_slopes[i]
   rhoR = state.rho + 0.5*slopes.rho_slopes[i]
   return (rhoL, rhoR)


## Computes edge velocities for a cell given hydro state and slopes
#
#  @param[in] i       cell index
#  @param[in] state   average hydro state for cell \f$i\f$
#  @param[in] slopes  HydroSlopes object
#
#  @return \f$(u_{i,L},u_{i,R})\f$
#
def computeEdgeVelocities(i, state, slopes):

   # compute edge velocities
   rho = state.rho
   u   = state.u
   mom = rho*u
   rhoL = rho - 0.5*slopes.rho_slopes[i]
   rhoR = rho + 0.5*slopes.rho_slopes[i]
   momL = mom - 0.5*slopes.mom_slopes[i]
   momR = mom + 0.5*slopes.mom_slopes[i]
   return (momL / rhoL, momR / rhoR)


## Computes edge temperatures for a cell given cv and edge internal energies
#
#  @param[in] cv           average hydro state for cell \f$i\f$
#  @param[in] e_values      internal energy at edge values
#
#  @return \f$(T_{i,L},T_{i,R})\f$
#
def computeEdgeTemperatures(cv, e_values):

   # compute internal energies at left and right edges
   eL = e_values[0]
   eR = e_values[1]

   # compute edge temperature using internal energy slope
   return (eL / cv, eR / cv)

## Computes edge internal energies for a cell, given hydro state and total
#  energy slope.
#
#  This is done in a matter so that 0.5*(E_L+E_R)=E_i, preserving
#  total energy conservation
#
#  @param[in] state    average hydro state for cell \f$i\f$
#  @param[in] slopes    hydro slope object
#
#  @return \f$(e_{i,L},e_{i,R})\f$
#
def computeHydroInternalEnergies(i, state, slopes):

   #Use conserved variables to construct e_L and e_R
   rho, mom, erg = state.getConservativeVariables()

   #Need all values at edges
   rho_L = rho - 0.5*slopes.rho_slopes[i]
   rho_R = rho + 0.5*slopes.rho_slopes[i]
   mom_L = mom - 0.5*slopes.mom_slopes[i]
   mom_R = mom + 0.5*slopes.mom_slopes[i]
   erg_L = erg - 0.5*slopes.erg_slopes[i]
   erg_R = erg + 0.5*slopes.erg_slopes[i]

   # compute internal energies at left and right edges
   u_L = mom_L/rho_L
   u_R = mom_R/rho_R

   e_L = erg_L/rho_L - 0.5*u_L**2
   e_R = erg_R/rho_R - 0.5*u_R**2

   # compute edge temperature using internal energy slope
   return (e_L, e_R)

## Updates all cross sections.
#
def updateCrossSections(cx,hydro,slopes,e_rad):

   # loop over cells
   for i in xrange(len(cx)):

      # get edge stuff
      state_avg = hydro[i]
      spec_heat = state_avg.spec_heat
      gamma = state_avg.gamma

      # compute edge quantities
      rho = computeEdgeDensities(i, state_avg, slopes)
      u = computeEdgeVelocities(i, state_avg, slopes)
      e = e_rad[i]

      # loop over edges and create state for each edge
      for edge in [0,1]:

         # create edge state
         state = HydroState(rho=rho[edge],u=u[edge],e=e[edge],
            spec_heat=spec_heat, gamma=gamma)
 
         # update edge cross section
         cx[i][edge].updateCrossX(state)


## Computes a vector in the ordering of radiation dofs, provided functions
#  of (x,t) for both directions.
#
#  The input function handles are functions of (x,t), and the output source
#  is given in the ordering of radiation dofs.
#
#  @param[in] f_minus  function handle for the minus direction
#  @param[in] f_plus   function handle for the plus direction
#  @param[in] mesh     mesh
#  @param[in] t        time at which to evaluate the functions
#
#  @return vector in the ordering of radiation dofs
#
def computeRadiationVector(f_minus, f_plus, mesh, t):

   # initialize vector
   y = np.zeros(mesh.n_elems*4)

   # loop over elements
   for i in xrange(mesh.n_elems):

      # get left and right x points on element
      xL = mesh.getElement(i).xl
      xR = mesh.getElement(i).xr

      #Evalute the plus and minus
      qLp, qRp = evalEdgeSource(f_plus, xL, xR, t)
      qLm, qRm = evalEdgeSource(f_minus, xL, xR, t)

      # get global indices
      y[getIndex(i,"L","-")] = qLm # dof i,L,-
      y[getIndex(i,"L","+")] = qLp # dof i,L,+
      y[getIndex(i,"R","-")] = qRm # dof i,R,-
      y[getIndex(i,"R","+")] = qRp # dof i,R,+

   return y


## Prints a tuple list.
#
def printTupleList(tuple_list):

    for tuple_i in tuple_list:

        for j in tuple_i:

            print j

## Construct analytic rad objects from the analytic functions of (x,t)
#
#  @param[in] mesh  mesh
#  @param[in] t     time value
#  @param[in] psim    analytic function for psi_minus
#  @param[in] psip    analytic function for psi_plus 
#
#  @return rad object that has analytic moments of psi
#
def computeAnalyticRadSolution(mesh,t,psim,psip):

   psi_vec = np.zeros(4*mesh.n_elems)
   for i in xrange(mesh.n_elems):

      # evaluate exact cell-average moments
      x_l = mesh.getElement(i).xl
      x_r = mesh.getElement(i).xr
      psim_lr = evalEdgeSource(psim,x_l,x_r,t)
      psip_lr = evalEdgeSource(psip,x_l,x_r,t)

      # add hydro state for cell
      psi_vec[getIndex(i,"L","-")] = psim_lr[0]
      psi_vec[getIndex(i,"R","-")] = psim_lr[1]
      psi_vec[getIndex(i,"L","+")] = psip_lr[0]
      psi_vec[getIndex(i,"R","+")] = psip_lr[1]

   return Radiation(psi_vec)


## Computes hydro states from analytic functions of (x,t)
#
#  @param[in] mesh  mesh
#  @param[in] t     time value
#  @param[in] rho   analytic function for density, \f$\rho(x,t)\f$
#  @param[in] u     analytic function for velocity, \f$u(x,t)\f$
#  @param[in] E     analytic function for total energy, \f$E(x,t)\f$
#  @param[in] cv    value for specific heat, \f$c_v\f$
#  @param[in] gamma value for gamma constant, \f$\gamma\f$
#
#  @return list of hydro states analytically evaluated at each cell center
#
def computeAnalyticHydroSolution(mesh,t,rho,u,E,cv,gamma):

   hydro = list()
   for i in xrange(mesh.n_elems):

      # get cell center
    #  x_i = mesh.getElement(i).x_cent

      # evaluate functions at cell center
    #  rho_i = rho(x=x_i, t=t)
    #  u_i   =   u(x=x_i, t=t)
    #  E_i   =   E(x=x_i, t=t)
      x_l = mesh.getElement(i).xl
      x_r = mesh.getElement(i).xr
      rho_i = evalAverageSource(rho,x_l,x_r,t)
      u_i = evalAverageSource(u,x_l,x_r,t)
      E_i = evalAverageSource(E,x_l,x_r,t)

      # add hydro state for cell
      hydro.append(HydroState(rho=rho_i, u=u_i, E=E_i,
         spec_heat=cv, gamma=gamma))

   return hydro


## Compute edge function for an element a certain time
#
def evalEdgeSource(func, x_l, x_r,t):

    #wrap with left and right basis function multiplying and at time t
    h = x_r - x_l
    f_L = lambda x: 2./h*(x_r-x)/h*func(x,t)
    f_R = lambda x: 2./h*(x-x_l)/h*func(x,t)
   
   # #These are the moments
    Q_L_m = quad(f_L, x_l, x_r, epsrel=QUAD_REL_TOL)[0]
    Q_R_m = quad(f_R, x_l, x_r, epsrel=QUAD_REL_TOL)[0]

    #Now compute the edge values based on exact moments
    #This is for the lumped expression of sources
    Q_L = Q_L_m
    Q_R = Q_R_m

    return ( Q_L, Q_R )


## Compute average function for an element and certain time
#
def evalAverageSource(func, x_l, x_r, t):

    h = x_r - x_l
    Q_a = 1./h*quad(func, x_l, x_r, args=(t), epsrel=QUAD_REL_TOL)[0]
    return Q_a



