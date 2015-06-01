## @package src.newtonStateHanlder
#  This file contains a class to handle computing linearized source
#  functions and temperature updates during a non-linear TRT solve.
#
#  This class is eventually passed to the PlanckianSource builder. This is
#  implemented as a class to simplify some of the temperature updating procedures.
#
#  This may be changed to functions only package or to inherit from 
#
# for the general problem 
# \f[
#   \frac{\partial Y}{c\partial t} = A[ Y(t)]
# \f]
# where \f$A\f$ is a operator that is a function of \f$Y,t\f$. NOTE: we keep the
# \f$1/c\Delta t\f$ term on the left hand terms. Each derived class is thus
# responsible for writing functions that evaluate \f$A^n,\;A^{n-1},\f$ and
# \f$Y^{n+1}\f$, which are functions evalOld, evalOlder, and evalImplicit,
# respectively.
#
# In implementation, \f$Y^n/c\Delta t\f$
# is also on the right hand side of the equation as a source term. This is
# the only term for which its derived class overrides computeTerm
# rather than implement the eval*** functions, since the term is the same for all
# stepping algorithms
from musclHancock import HydroState
from transientSource import TransientSourceTerm

#===================================================================================
## Main class to handle newton solve and temperature updates, etc.
#
class NewtonStateHandler(TransientSourceTerm):

    #--------------------------------------------------------------------------------
    ## Constructor
    #
    def __init__(self,mesh,time_stepper='BE'):

        TransientSourceTerm.__init__(self,mesh,time_stepper)
        self.mesh = mesh
        self.time_stepper = time_stepper

    #--------------------------------------------------------------------------------
    ## Evaluate nu in linearization
    def evalNu(self):

        return 0.0

    #--------------------------------------------------------------------------------
    ## Evaluate 


