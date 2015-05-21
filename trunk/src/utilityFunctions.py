## @package src.utilityFunctions
#  Contains helper functions that do not belong in any particular class.

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


