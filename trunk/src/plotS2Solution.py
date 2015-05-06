import matplotlib.pyplot as plt
import numpy as np
from numpy import array

## Function to transform mesh into discontinuous x-points for plotting.
#
#  @param[in] mesh  mesh data
#  @return    array of the discontinuous x-points
#
def makeXPoints(mesh):
   # number of dofs
   n = 2*mesh.n_elems
   # initialize
   x = np.zeros(n)
   # loop over elements
   for i in xrange(mesh.n_elems):
      iL = 2*i
      iR = iL + 1
      x[iL] = mesh.getElement(i).xl # left  end of element
      x[iR] = mesh.getElement(i).xr # right end of element
   # return
   return x

## Function to transform mesh into continuous x-points for plotting.
#
#  @param[in] mesh  mesh data
#  @return    array of the continuous x-points
#
def makeContinuousXPoints(mesh):
   # number of edges
   n = mesh.n_elems + 1
   # initialize
   x = np.zeros(n)
   # loop over elements
   for i in xrange(mesh.n_elems):
      iL = 2*i
      iR = iL + 1
      x[i] = mesh.getElement(i).xl             # left  end of element
   x[n-1] = mesh.getElement(mesh.n_elems-1).xr # right end of element
   # return
   return x

## Function to transform array of tuples into array for plotting.
#
#  @param[in] tuple_sarray  array of tuples
#  @return    array of the discontinuous data
#
def makeYPoints(tuples_array):
   # number of elements
   n_elems = len(tuples_array)
   # number of dofs
   n = 2*n_elems
   # initialize
   y = np.zeros(n)
   # loop over elements
   for i in xrange(n_elems):
      iL = 2*i
      iR = iL + 1
      y[iL] = tuples_array[i][0] # left  value
      y[iR] = tuples_array[i][1] # right value
   # return
   return y

## Function to plot the S-2 solution.
#
#  @param[in] mesh      mesh data
#  @param[in] psi_minus S-2 angular flux solution for the minus direction,
#                       as an array of tuples of left and right values
#  @param[in] psi_plus  S-2 angular flux solution for the plus direction,
#                       as an array of tuples of left and right values
#  @param[in] save      boolean flag for saving the plot as a .pdf file
#
def plotS2Solution(mesh, psi_minus, psi_plus, save=False,
   psi_minus_exact=None, psi_plus_exact=None):

   # create x-points
   x            = makeXPoints(mesh)           # discontinuous x-points
   x_continuous = makeContinuousXPoints(mesh) # continuous    x-points

   # transform arrays of tuples into arrays
   psi_minus_array = makeYPoints(psi_minus)
   psi_plus_array  = makeYPoints(psi_plus)

   # plot
   plt.plot(x, psi_minus_array, 'r-o', label='psi-')
   plt.plot(x, psi_plus_array,  'b-x', label='psi+')
    
   # plot exact solutions if there are any
   if psi_minus_exact is not None:
      plt.plot(x_continuous, psi_minus_exact, 'k--', label='psi-, exact')
   if psi_plus_exact  is not None:
      plt.plot(x_continuous, psi_plus_exact,  'k-',  label='psi+, exact')

   # annotations
   plt.xlabel('x')
   plt.ylabel('Angular Flux')
   plt.legend(loc='best')

   # save if requested
   if save:
      plt.savefig('s2solution.pdf')
   else:
      plt.show()
