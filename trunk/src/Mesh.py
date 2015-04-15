## @package Mesh
#  Contains mesh classes

#================================================================================
"""Very simple 1D, cartesian, fixed grid mesh.  Just contains a list of Element
objects, which are defined below.  
"""
## Class for mesh.
#
#  Just contains a list of element objects.
class Mesh:

    """Define the constructor. Pass in number of elements, width of the entire
    domain, and the starting location of the left edge.  By default it is zeo. 
    
    Note that you cannot overload constructors in python
    like you are used to. If you overload, it will just redefine them"""
    #----------------------------------------------------------------------------
    def __init__(self, n_elements, width, x_start=0.):

        ##The mesh currently has fixed element widths, but let each element have one
        self.n_elems = n_elements
        dx = width/float(n_elements)
        
        #Build elements in a list comprehension
        self.elements = [Element(i,(i+0.5)*dx,dx) for i in xrange(n_elements)]
               
    #----------------------------------------------------------------------------
    def __str__(self):

        ## Define what happens when 'print <Mesh_object>' is called
        print_str = "\n-----------------------------------------------------" \
                    "\n                     Elements" \
                    "\n-----------------------------------------------------\n"

        ## Concatenate element strings together and return as one long string
        for el in self.elements:       #for each "Element" in self.elements

            print_str += el.__str__()  #el is an object, not an index in this loop

        return print_str


    #----------------------------------------------------------------------------
    """Accessor function"""
    def getElement(self, el_id):

        return elements[el_id]


#================================================================================
"""Simple 1D element. Element has x_l, x_r, and x_cent. All elements are of a fixed
width, but we will let each element store its dx in case we want to modify this in
the future."""
class Element:

    #----------------------------------------------------------------------------
    def __init__(self, el_id, x_center, width ):

        self.el_id = el_id              #id for this element
        self.dx = width                 ## @var dx spatial width, doxygen?
        self.x_cent = x_center          #center of cell
        self.xl = x_center - width/2.   #
        self.xr = x_center + width/2.


    #----------------------------------------------------------------------------
    def __str__(self):

        return "ID: %i x_l : %.4f x_r : %.4f\n" % (self.el_id, self.xl, self.xr)

