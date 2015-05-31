## @package src.utilityFunctions
#  Contains helper functions that do not belong in any particular class.

from math import log

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
def printConvergenceTable(dx,err,rates=None,dx_desc='size',err_desc='err'):

   # compute rates if they were not provided
   if rates is None:
      rates = computeConvergenceRates(dx,err)

   # determine number of refinement cycles from length of lists
   n_cycles = len(dx)

   # print header
   print('\n%11s %11s    Rate' % (dx_desc,err_desc))
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
   print('\n')

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
