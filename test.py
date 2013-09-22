from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt
import SIMPy

##################
# Pousuille flow #
##################

N = 10  # problem size
tol = 1e-4  # tolerance for solution
imax = 1000  # max number of iterations
method = "SOR"  # SIM method
omega = 1.2  # factor for SOR
x = np.zeros(N)  # starting guess for solution
z = np.linspace(0, 1, N)  # generate grid

# Assemble tri-diagonal system matrix for CDS operator
data = np.zeros((3, N))
data[0, 0:N-1] = 1  # super diagonal
data[1, :] = -2  # diagonal
data[2, 1:N] = 1  # sub diagonal
offsets = np.array([-1, 0, 1])
A = sp.spdiags(data, offsets, N, N, format="csc")

# Assemble source vector
b = np.zeros(N)
b[1:N] = -8/(N-1) ^ 2

u1 = spsolve(A, b)  # direct solution
u2, _, iter, _, G = SIMPy.solve(A, b,
            "sor", 500, 1e-4, 1, np.ones(N), False)  # iterative solution

## Plotting
plt.plot(u1, z, '--o', linewidth=2, label="direct solution")
plt.plot(u2, z, '--o', linewidth=2, label="iterative solution")
plt.legend()
plt.show()

print("done!")
