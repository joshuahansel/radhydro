## @package src.integrationUtilities
#  Provides functions for integration.

from scipy.integrate import quad # adaptive quadrature function

## Function to compute the \f$L^1\f$ error of a linear discontinous (LD)
#  numerical solution, given an analytic solution \f$f(x)\f$.
#
#  Computes the \f$L^1\f$ error \f$\int\limits_a^b\left|f(x)-u_h(x)\right|dx\f$,
#  where \f$a\f$ and \f$b\f$ are the left and right endpoints of the mesh,
#  respectively.
#
#  @param[in] mesh                mesh data
#  @param[in] numerical_solution  numerical linear discontinuous solution
#                                 \f$u_h(x)\f$ corresponding to the mesh
#                                 data, provided as a mesh.n_elems-sized list
#                                 of tuples of left and right values, i.e.,
#                                 numerical_solution[i]\f$=(U_{i,L},U_{i,R})\f$.
#  @param[in] f                   analytic soultion function \f$f(x)\f$ with which
#                                 to integrate the difference.
#  @return  the integral \f$\int\limits_a^b\left|f(x)-u_h(x)\right|dx\f$, where
#           \f$a\f$ and \f$b\f$ are the left and right endpoints of the mesh,
#           respectively.
#
def computeL1ErrorLD(mesh, numerical_solution, f):

   # initialize integral to zero
   exact_integral = 0.0

   # loop over elements and add local integrals to global integral
   for i in xrange(mesh.n_elems):
      # express local numerical solution as linear function y(x) = m*x + b
      el = mesh.getElement(i)
      xL = el.xl
      xR = el.xr
      yL = numerical_solution[i][0]
      yR = numerical_solution[i][1]
      dx = xR - xL
      dy = yR - yL
      m = dy / dx
      b = yL - xL*m

      # local solution function
      def y(x):
         return m*x + b

      # local difference function
      def difference(x):
         return abs(f(x) - y(x))
   
      # compute local integral of difference
      local_integral = quad(difference, xL, xR)[0]
    
      # add to global integral of difference
      exact_integral += local_integral

   return exact_integral
