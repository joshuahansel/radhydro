""" This is the main executioner. To execute the program type 'python main.py'
"""

""" Import files from built in libraries and user-defined classes """

from src.Mesh import Mesh
from src.CrossXInterface import *

def main():

    #Create a mesh, currently hardcoded
    mesh = Mesh(10, 5.)
    print mesh


    #Dear josh: this will Test out inv cubed cross X. Derived classes are a little wierd in python, you
    #you just deal with derived class types, and any bass class function or members
    #it needs it will have access to, as long as the constructors are called
    #correctly. 
    print "\n The fancy cross sections are"
    cross_sects = [InvCubedCrossX(5.0,i+1,i+1,scale_coeff=10.0) for i in
            xrange(len(cross_sects))]
    
    for i in cross_sects:
        print i

    
    #Dear Josh: These are normal cross sections, these are all you need to deal with
    #in the S2 Solver. The above was just me learning how inheritence works in python,
    #but we shouldnt really need to deal with it other than cross X's
    cross_sects = [CrossXInterface(1,3) for i in xrange(mesh.n_elems)]
    print "\nCross sections are: "
    for i in cross_sects:
        print i




    #Call radiation solver



if __name__ == "__main__":
    main()
