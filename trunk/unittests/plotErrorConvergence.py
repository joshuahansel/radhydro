# add source directory to module search path
import sys
sys.path.append('../src')

# symbolic math packages
from math import pi

import pickle

# numpy
import numpy as np
import matplotlib.pyplot as plt
import math

from utilityFunctions import printHydroConvergenceTable


def main():

    data = pickle.load( open( "results/testRadHydroDiffMMS.pickle","r") )
    print [key for key in data]
    print data["Errors"]

    dx = data["dx"]
    dt = data["dt"]
    err= [data["Errors"][i]["e"] for i in range(len(data["Errors"]))]
    err1= [data["Errors"][i]["Fr"] for i in range(len(data["Errors"]))]

    plt.figure(0)
    plt.loglog(dx,err,"+-", label="$e$ L$_2$ Relative Error")

    err[0]*=0.9

    #Add reference lines
    plt.loglog([dx[0],dx[-1]], [err[0], err[0]*(dx[-1]/dx[0])**2.],"--",label="Second Order Ref.")
    plt.loglog([dx[0],dx[-1]], [1.4*err[0], 1.4*err[0]*(dx[-1]/dx[0])**1.],"-.",label="First Order Ref.")
    plt.legend(loc='best')
    plt.xlabel("$\Delta x$")
    plt.ylabel(r"$L_2^{\mathrm{rel}}(e)$", usetex=True)
    plt.savefig("MMS_diffusion_limit_e_convergence.pdf",bbox_inches='tight')


    plt.figure(1)
    plt.loglog(dt,err1,"+-", label=r"$\rho u$ L$_2$ Relative Error")

    err1[0]*=0.9
    f = 0.67

    #Add reference lines
    plt.loglog([dt[0],dt[-1]], [f*err1[0], f*err1[0]*(dt[-1]/dt[0])**2.],"--",label="Second Order Ref.")
    plt.loglog([dt[0],dt[-1]], [1.4*err1[0], 1.4*err1[0]*(dt[-1]/dt[0])**1.],"-.",label="First Order Ref.")
    plt.legend(loc='best')
    plt.xlabel("$\Delta t$")
    plt.ylabel(r"$L_2^{\mathrm{rel}}(\rho u)$", usetex=True)
    plt.savefig("MMS_diffusion_limit_rhou_convergence.pdf",bbox_inches='tight')

if __name__ == "__main__":
    main()

