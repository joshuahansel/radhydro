## @package src.createMMSSourceFunctions
#  Contains functions to create MMS source functions

# The following import allows for printing LaTeX formulas in display.
# This script is best run with the ipython QTConsole, which can be
# downloaded using the synaptic package manager as "ipython-qtconsole".
# This interpreter is invoked as "ipython qtconsole", and then this
# script may be executed in the interpreter with "execfile('filename.py')".
from IPython.display import display

# symbolic math packages
from sympy import symbols, sqrt, diff, Eq, init_printing, simplify, sympify
from sympy.utilities.lambdify import lambdify

import numpy as np # numpy
import globalConstants as GC

## Creates MMS source functions of (x,t) for each of the governing equations
#  in the radiation-only system.
#
#  Currently, the provided expressions are assumed to contain only one
#  parameter, called 'alpha'.
#
#  @return (psim_f, psip_f), where 'quantity_f' denotes
#     a function handle for the source function of (x,t) to the equation
#     corresponding to 'quantity'
#
def createMMSSourceFunctionsRadOnly(psim, psip,
   sigma_s_value, sigma_a_value, alpha_value=0.0, display_equations=False):
   
   # declare symbolic variables
   x, t, alpha = symbols('x t alpha')
   sigs, sigt, c = symbols('sigma_s sigma_t c')
   
   # compute other radiation quantities
   phi = psim + psip
   
   # temporal derivatives
   dpsimdt  = diff(psim,t)
   dpsipdt  = diff(psip,t)
   
   # spatial derivatives
   dpsimdx  = diff(psim,x)
   dpsipdx  = diff(psip,x)
   
   # compute sources
   Qpsim = dpsimdt/c - dpsimdx/sqrt(3) + sigt*psim - sigs/2*phi
   Qpsip = dpsipdt/c + dpsipdx/sqrt(3) + sigt*psip - sigs/2*phi
   
   # display equations
   if display_equations:

      # initialize printing to use prettiest format available (e.g., LaTeX, Unicode)
      init_printing()

      # create an equation and then display it
      eq = Eq(symbols('Psi^-'),psim)
      display(eq)
      eq = Eq(symbols('Psi^+'),psip)
      display(eq)
      eq = Eq(symbols('Q_-'),Qpsim)
      display(eq)
      eq = Eq(symbols('Q_+'),Qpsip)
      display(eq)
   
   # substitute for all symbols except x and t
   substitutions = dict()
   substitutions['alpha'] = alpha_value
   substitutions['c']     = GC.SPD_OF_LGT
   substitutions['sigma_s'] = sigma_s_value
   sigma_t_value = sigma_s_value + sigma_a_value
   substitutions['sigma_t'] = sigma_t_value

   # make substitutions
   Qpsim_sub = Qpsim.subs(substitutions)
   Qpsip_sub = Qpsip.subs(substitutions)

   # create MMS source functions
   psim_f = lambdify((symbols('x'),symbols('t')), Qpsim_sub, "numpy")
   psip_f = lambdify((symbols('x'),symbols('t')), Qpsip_sub, "numpy")

   return (psim_f, psip_f)


## Creates MMS source functions of (x,t) for each of the governing equations
#  in the hydro-only system.
#
#  Currently, the provided expressions are assumed to contain only one
#  parameter, called 'alpha'. An ideal gas EOS is assumed.
#
#  @return (rho_f, u_f, E_f, psim_f, psip_f), where 'quantity_f' denotes
#     a function handle for the source function of (x,t) to the equation
#     corresponding to 'quantity'
#
def createMMSSourceFunctionsHydroOnly(rho, u, E,
   gamma_value, cv_value, alpha_value, display_equations=False):
   
   # declare symbolic variables
   x, t, alpha, Qpsim, Qpsip = symbols('x t alpha Qpsim Qpsip')
   gamma, cv, a = symbols('gamma c_v a')
   
   # compute other thermodynamic quantities based on ideal gas EOS
   e = E/rho - u*u/2
   p = rho*e*(gamma - 1)
   rhou = rho*u
   
   # temporal derivatives
   drhodt   = diff(rho,t)
   drhoudt  = diff(rhou,t)
   dEdt     = diff(E,t)
   
   # spatial derivatives
   drhoudx  = diff(rhou,x)
   drhouudx = diff(rhou*u,x)
   dpdx     = diff(p,x)
   dEfluxdx = diff((E+p)*u,x)
   
   # compute sources
   Qrho = drhodt + drhoudx
   Qu = drhoudt + drhouudx + dpdx
   QE = dEdt + dEfluxdx
   Qpsim = 0
   Qpsip = 0
   
   # display equations
   if display_equations:

      # initialize printing to use prettiest format available (e.g., LaTeX, Unicode)
      init_printing()

      # create an equation and then display it
      eq = Eq(symbols('rho'),rho)
      display(eq)
      eq = Eq(symbols('u'),u)
      display(eq)
      eq = Eq(symbols('E'),E)
      display(eq)
      eq = Eq(symbols('e'),e)
      display(eq)
      eq = Eq(symbols('p'),p)
      display(eq)
      eq = Eq(symbols('Q_rho'),Qrho)
      display(eq)
      eq = Eq(symbols('Q_u'),Qu)
      display(eq)
      eq = Eq(symbols('Q_E'),QE)
      display(eq)
   
   # substitute for all symbols except x and t
   substitutions = dict()
   substitutions['alpha'] = alpha_value
   substitutions['a']     = GC.RAD_CONSTANT
   substitutions['c_v']   = cv_value
   substitutions['gamma'] = gamma_value

   # make substitutions
   Qrho_sub  = Qrho.subs(substitutions)
   Qu_sub    = Qu.subs(substitutions)
   QE_sub    = QE.subs(substitutions)
   
   # create MMS source functions
   rho_f  = lambdify((symbols('x'),symbols('t')), Qrho_sub, "numpy")
   u_f    = lambdify((symbols('x'),symbols('t')), Qu_sub,   "numpy")
   E_f    = lambdify((symbols('x'),symbols('t')), QE_sub,   "numpy")
   psim_f = lambdify((symbols('x'),symbols('t')), Qpsim,    "numpy")
   psip_f = lambdify((symbols('x'),symbols('t')), Qpsip,    "numpy")

   return (rho_f, u_f, E_f, psim_f, psip_f)


## Creates MMS source functions of (x,t) for each of the governing equations
#  in the full RH system.
#
#  Currently, the provided expressions are assumed to contain only one
#  parameter, called 'alpha'. An ideal gas EOS is assumed.
#
#  @return (rho_f, u_f, E_f, psim_f, psip_f), where 'quantity_f' denotes
#     a function handle for the source function of (x,t) to the equation
#     corresponding to 'quantity'
#
def createMMSSourceFunctionsRadHydro(rho, u, E, psim, psip,
   sigma_s_value, sigma_a_value, gamma_value, cv_value,
   alpha_value, display_equations=False):
   
   # declare symbolic variables
   x, t, alpha = symbols('x t alpha')
   sigs, siga, sigt = symbols('sigma_s sigma_a sigma_t')
   gamma, cv, c, a = symbols('gamma c_v c a')
   
   # compute other thermodynamic quantities based on ideal gas EOS
   e = E/rho - u*u/2
   p = rho*e*(gamma - 1)
   T = e/cv
   rhou = rho*u
   
   # compute other radiation quantities
   phi = psim + psip
   Er = phi/c
   Fr = (psip - psim)/sqrt(3)
   Fr0 = Fr - sympify('4/3')*Er*u
   Q0 = siga*a*c*T**4 - sigt*u/c*Fr0
   Q1 = sympify('4/3')*sigt*Er*u
   Qm = Q0 - sympify('3/sqrt(3)')*Q1
   Qp = Q0 + sympify('3/sqrt(3)')*Q1
   
   # temporal derivatives
   drhodt   = diff(rho,t)
   drhoudt  = diff(rhou,t)
   dEdt     = diff(E,t)
   dpsimdt  = diff(psim,t)
   dpsipdt  = diff(psip,t)
   
   # spatial derivatives
   drhoudx  = diff(rhou,x)
   drhouudx = diff(rhou*u,x)
   dpdx     = diff(p,x)
   dEfluxdx = diff((E+p)*u,x)
   dpsimdx  = diff(psim,x)
   dpsipdx  = diff(psip,x)
   
   # compute sources
   Qrho = drhodt + drhoudx
   Qu = drhoudt + drhouudx + dpdx - sigt/c*Fr0
   QE = dEdt + dEfluxdx + siga*c*(a*T**4 - Er) - sigt*u/c*Fr0
   Qpsim = dpsimdt/c - dpsimdx/sqrt(3) + sigt*psim - sigs/2*phi - Qm/2
   Qpsip = dpsipdt/c + dpsipdx/sqrt(3) + sigt*psip - sigs/2*phi - Qp/2
   
   # display equations
   if display_equations:

      # initialize printing to use prettiest format available (e.g., LaTeX, Unicode)
      init_printing()

      # create an equation and then display it
      eq = Eq(symbols('rho'),rho)
      display(eq)
      eq = Eq(symbols('u'),u)
      display(eq)
      eq = Eq(symbols('E'),E)
      display(eq)
      eq = Eq(symbols('e'),simplify(e))
      display(eq)
      eq = Eq(symbols('T'),T)
      display(eq)
      eq = Eq(symbols('p'),p)
      display(eq)
      eq = Eq(symbols('Psi^-'),psim)
      display(eq)
      eq = Eq(symbols('Psi^+'),psip)
      display(eq)
      eq = Eq(symbols('E_r'),Er)
      display(eq)
      eq = Eq(symbols('F_r'),Fr)
      display(eq)
      eq = Eq(symbols('Q_rho'),Qrho)
      display(eq)
      eq = Eq(symbols('Q_u'),Qu)
      display(eq)
      eq = Eq(symbols('Q_E'),QE)
      display(eq)
      eq = Eq(symbols('Q_-'),Qpsim)
      display(eq)
      eq = Eq(symbols('Q_+'),Qpsip)
      display(eq)
   
   # substitute for all symbols except x and t
   substitutions = dict()
   substitutions['alpha'] = alpha_value
   substitutions['a']     = GC.RAD_CONSTANT
   substitutions['c']     = GC.SPD_OF_LGT
   substitutions['c_v']   = cv_value
   substitutions['gamma'] = gamma_value
   substitutions['sigma_s'] = sigma_s_value
   substitutions['sigma_a'] = sigma_a_value
   sigma_t_value = sigma_s_value + sigma_a_value
   substitutions['sigma_t'] = sigma_t_value

   # make substitutions
   Qrho_sub  = Qrho.subs(substitutions)
   Qu_sub    = Qu.subs(substitutions)
   QE_sub    = QE.subs(substitutions)
   Qpsim_sub = Qpsim.subs(substitutions)
   Qpsip_sub = Qpsip.subs(substitutions)
   
   # create MMS source functions
   rho_f  = lambdify((symbols('x'),symbols('t')), Qrho_sub,  "numpy")
   u_f    = lambdify((symbols('x'),symbols('t')), Qu_sub,    "numpy")
   E_f    = lambdify((symbols('x'),symbols('t')), QE_sub,    "numpy")
   psim_f = lambdify((symbols('x'),symbols('t')), Qpsim_sub, "numpy")
   psip_f = lambdify((symbols('x'),symbols('t')), Qpsip_sub, "numpy")

   return (rho_f, u_f, E_f, psim_f, psip_f)

