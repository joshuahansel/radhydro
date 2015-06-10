# add source directory to module search path
import sys
sys.path.append('../src')

# The following import allows for printing LaTeX formulas in display.
# This script is best run with the ipython QTConsole, which can be
# downloaded using the synaptic package manager as "ipython-qtconsole".
# This interpreter is invoked as "ipython qtconsole", and then this
# script may be executed in the interpreter with "execfile('filename.py')".
from IPython.display import display

# symbolic math packages
from sympy import symbols, exp, sin, sqrt, pi, diff, Eq, init_printing

# numpy
import numpy as np

from plotUtilities import plotSymbolicFunction

# user options:
display_equations = False

# initialize printing to use prettiest format available (e.g., LaTeX, Unicode)
init_printing()

# declare symbolic variables
x, t, alpha = symbols('x t alpha')
sigs, siga, sigt = symbols('sigma_s sigma_a sigma_t')
gamma, cv, c, a = symbols('gamma c_v c a')

# create solution for thermodynamic state and flow field
rho = exp(-1*alpha*t)*sin(pi*x) + 2
u   = exp(-2*alpha*t)*sin(pi*x)
E   = exp(-3*alpha*t)*sin(pi*x) + 3

# create solution for radiation field
psim = 2*t*sin(pi*(1-x))
psip = t*sin(pi*x)

# compute other thermodynamic quantities based on ideal gas EOS
e = E/rho - u*u/2
p = rho*e*(gamma - 1)
T = e/cv
rhou = rho*u

# compute other radiation quantities
phi = psim + psip
Er = phi/c
Fr = (psip - psim)/sqrt(3)
Fr0 = Fr - 4/3*Er*u
Q0 = siga*a*c*T**4 - sigt*u/c*Fr0
Q1 = 4/3*sigt*Er*u
Qm = Q0 - 3/sqrt(3)*Q1
Qp = Q0 + 3/sqrt(3)*Q1

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
   eq = Eq(symbols('rho'),rho)
   display(eq)
   eq = Eq(symbols('u'),u)
   display(eq)
   eq = Eq(symbols('E'),E)
   display(eq)
   eq = Eq(symbols('e'),e)
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

# substitute for all symbols except x
substitutions = dict()
substitutions['t']     = 0.1
substitutions['alpha'] = 0.01
substitutions['a']     = 1.0
substitutions['c']     = 1.0
substitutions['c_v']   = 1.0
substitutions['gamma'] = 1.4
substitutions['sigma_s'] = 1.0
substitutions['sigma_a'] = 1.0
substitutions['sigma_t'] = 2.0

# create list of x points
xpoints = np.linspace(0.0,1.0,100)

# plot
#plotSymbolicFunction(xpoints, rho, substitutions, '$\\rho$', 'Density')
#plotSymbolicFunction(xpoints, u,   substitutions, '$u$',     'Velocity')
#plotSymbolicFunction(xpoints, E,   substitutions, '$E$',     'Total Energy')
#plotSymbolicFunction(xpoints, p,   substitutions, '$p$',     'Pressure')
#plotSymbolicFunction(xpoints, e,   substitutions, '$e$',     'Internal Energy')
#plotSymbolicFunction(xpoints, T,   substitutions, '$T$',     'Temperature')
plotSymbolicFunction(xpoints, Qrho, substitutions, '$Q_\\rho$', 'MMS Source, $\\rho$')
plotSymbolicFunction(xpoints, Qu, substitutions, '$Q_u$', 'MMS Source, $u$')
plotSymbolicFunction(xpoints, QE,  substitutions, '$Q_E$',   'MMS Source, $E$')
plotSymbolicFunction(xpoints, Qpsim, substitutions, '$Q_-$', 'MMS Source, $\Psi^-$')
plotSymbolicFunction(xpoints, Qpsip, substitutions, '$Q_+$', 'MMS Source, $\Psi^+$')

