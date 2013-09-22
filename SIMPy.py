import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve


def my_diag(A):
#############################################
# my_diag                                   #
#   Extraction of diag from a sparse matrix #
#############################################

    N = np.size(A, 1)  # get number of diag elements = num colums
    ii = np.arange(0, N)  # create seq of int from 1 to N
    return A[ii, ii]


def solve(A, b, *vartuple):
###############################################################################
# SIMPy                                                                       #
#   Solves system of linear equations A*x=b  using a stationary iterative     #
#   method. Terminates after maximum imax iterations, or when the inf-norm    #
#   of the residual, relative the inf-norm of the initial residual, becomes   #
#   less than tol. The following stationary iterative methods are             #
#   implemented: Jacobi, Gauss-Seidel, SOR                                    #
# Syntax:                                                                     #
#   x,error,iter,flag,G = SIMPy(A, b, method, imax, tol, omega, x, reqG)      #
# Input    :                                                                  #
#   A      :  system matrix                                                   #
#   b      :  right hand side vector (default init. with zero-vector)         #
#   method :  SIM, "jacobi","gs","sor" (delfault method is jacobi)            #
#   imax   :  maximum number of iterations (default is number of equations)   #
#   tol    :  tolerance on residual, relative to initial residual (sqrt(eps)) #
#   omega  :  SOR-relaxation parameter, 0 <= omega < 2 (defualt 1)            #
#   x0     :  initial guess vector                                            #
#   reqG   :  request G, => True or False (default False)                     #
# Output   :                                                                  #
#   x      :  solution to A*x = b (converged)                                 #
#   error  :  history of inf-norm of residual normed w.r.t. initial residual  #
#   iter   :  number of iterations to reach converged solution                #
#   flag   :  convergence flag (0: solution converged, 1: no convergence)     #
#   G      :  iteration matrix (expensive! - don"t request unless needed)     #
# Note     :                                                                  #
#   SIMPy is a python "conversion" of handout material from DTU course 41319  #
###############################################################################

    # Set default input, if arguments are not given
    nvargin = len(vartuple)  # determine number of variable input arguments

    if nvargin < 6:  # reqG undefined
        reqG = False  # default 0 = no
    else:
        reqG = vartuple[5]
    if nvargin < 5:  # x0 undefined
        x = 0*b  # default init. with zero-vector
    else:
        x = vartuple[4]
    if nvargin < 4:  # omega undefined
        omega = 1  # default omega, sor=>gs
    else:
        omega = vartuple[3]
    if nvargin < 3:  # tol undefined
        tol = np.sqrt(np.spacing(1))  # default tol
    else:
        tol = vartuple[2]
    if nvargin < 2:  # imax undefined
        imax = np.size(A, 0)  # default imax
    else:
        imax = vartuple[1]
    if nvargin < 1:  # method undefined
        method = "jacobi"  # delfault method
    else:
        method = vartuple[0]

    # Compute initial residual vector and norm
    returnflag = 0  # initialize flag for early function return
    rhsInorm = np.linalg.norm(b, np.inf)  # Inf-norm of rhs-vectorhon

    if rhsInorm == 0:  # homogene problem, x = b = 0
        x = b  # homogene solution
        error0 = 0  # zero error
        returnflag = 1  # return function
    else:  # inhomogene problem, non-zero rhs vector
        if np.linalg.norm(x, np.inf) == 0:  # zero initial guess
            res = rhsInorm  # norm of residual vector
            error0 = 1  # relative error
        else:  # non-zero initial guess
            r = b - A*x  # initial residual vector
            res = np.linalg.norm(r, np.inf)  # norm of residual vector
            error0 = res/rhsInorm  # relative error for initial guess
            if error0 <= tol:  # initial error less than tolerance
                returnflag = 1  # return function
        res0 = res  # ini. res. - stored for error computation

    # Matrix splitting based on "method" input
    if method.lower() == "jacobi":  # jacobi splitting
        w = 1
        M = sp.spdiags(my_diag(A), 0, np.size(A, 0), np.size(A, 1))
        N = M-A
    elif method.lower() == "gs":  # gauss-seidel splitting
        w = 1
        M = sp.tril(A, 0)
        N = M-A
    elif method.lower() == "sor":  # SOR splitting
        w = omega
        diagV = my_diag(A)  # extract diagonal of sparse matrix
        M = sp.spdiags(diagV, 0, np.size(A, 0), np.size(A, 1)) \
            + w*sp.tril(A, -1)
        N = (1-w)*sp.spdiags(diagV, 0, np.size(A, 0), np.size(A, 1)) \
            - w*sp.triu(A, 1)

    # Compute iteration matrix if requested as output (expensive!)
    G = 0  # set default return value for G
    if reqG:  # iteration matrix requested
        print(np.shape(M))
        print(np.shape(N))
        G = spsolve(M, N, None, False)  # iteration matrix

    # Return function
    if returnflag == 1:
        iter = 0
        flag = 0
        return x, error0, 0, flag, G

    # Iterate till error < tol
    iter = 0  # initialize iteration counter
    error = np.zeros(imax+1)  # vector to hold iteration error history
    error[0] = error0

    while iter < imax and error[iter] > tol:
        iter = iter+1  # update iteration counter
        x = spsolve(M, N*x+w*b)  # update approximation
        r = b - A*x  # residual vector
        res = np.linalg.norm(r, np.inf)  # norm of residual vector
        error[iter] = res/res0  # relative error

    error = error[0:iter+1]  # remove undone iterations from error

    # Check for final convergence
    if (error[iter] > tol):  # no convergence
        flag = 1  # failed convergence flag
    else:  # solution converged
        flag = 0  # convergence flag

    return x, error, iter, flag, G