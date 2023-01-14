# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy.linalg as lg
from scipy.sparse import diags
import scipy.sparse.linalg
from numpy.linalg import norm
from numpy.linalg import inv

# function to determine the one-dimensional matrix
def matrix_1d(n):
    h = 1/n
    matrix = np.zeros(shape=(n+1,n+1))
    
    # fill diagonal with 4's
    np.fill_diagonal(matrix, 4)
    
    b = np.ones(n)
    # fill upper diagonal with -h-2
    np.fill_diagonal(matrix[1:], (-h-2)*b)
    # fill lower diagonal with h-2
    np.fill_diagonal(matrix[:,1:], (h-2)*b)
    
    # fill corners with 2*(h**2)       
    matrix[0][0] = 2*(h**2)
    matrix[n][n] = 2*(h**2)
    
    # clean rest of first and last row
    for j in range(1,n+1):
        matrix[0][j] = 0
    for k in range(0,n):
        matrix[n][k] = 0
       
    # divide matrix by 2*(h**2)
    matrix = matrix / (2 * (h**2))
    return matrix

# function to determine f in the 1-d case
def functionvalues1(n):
    h = 1/n
    values = np.zeros(n+1)
    
    # boundary
    values[0] = 0
    values[n] = math.sin(1)
    
    # function
    for i in range(1,n):
        values[i] = math.sin(i*h) + math.cos(i*h)
    
    return values

# function to determine u in the 1-d case
def u(n):
    h = 1/n
    u_values = np.zeros(n+1)
    
    # calculate exact solution
    for i in range(n+1):
        u_values[i] = math.sin(i*h)
    
    return u_values

# function to determine an matrix that has 1's on the diagonal
# except in the corners, there it has zero's 
def semi_I(n):
    id = np.zeros(shape=(n+1,n+1))
    
    # 1's everywhere except in the corners
    for i in range(1,n):
        id[i][i] = 1
    return id   

# function that makes a 2-d matrix out of two
# 1-d matrices A and B       
def matrix_2d(A,B):
    n = np.shape(A)[0]
    # take sum of kronecker product of matrices
    matrix = np.kron(A,B) + np.kron(B,A)
    
    # if 0 on diagonal, replace by 1
    for i in range((n)**2):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
    return matrix
        
# function that determines f^h in the 2-d case
def functionvalues2(n):
    h = 1/n
    f = np.zeros(shape=(n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            # boundary
            if i == 0 or j == 0:
                f[i][j] = 0
            # boundary
            elif i == n:
                f[i][j] = math.sin(j*h)
            # boundary
            elif j == n:
                f[i][j] = math.sin(i*h)
            # normal f
            else:
                f[i][j] = ((i*h)**2+(j*h)**2)*math.sin((i*h)*(j*h)) + (i*h + j*h) *  math.cos ((i*h)*(j*h))
    # make matrix into one long vector
    f = f.flatten('F')
    return f
       
# function that determines u_{ex} in the 2-d case 
def u_2d(n):
    h = 1/n
    u = np.zeros(shape=(n+1,n+1))
    # determine u_{ex}
    for i in range(n+1):
        for j in range(n+1):
            u[i][j] = math.sin((i*h)*(j*h))
    # make matrix into one long vector
    u = u.flatten('F')
    return u

# function that determines the LU-decomposition of a given matrix A
def LU(A):
    n = np.shape(A)[0]
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    
    # LU factorization, loping over rows
    for i in range(n):
        # eiminate entries below i with row operations on U
        # reverse the row operations to change L
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i] 
    return L,U

# function that determines the  discrete solution u of a given LU-decomposition
# and an output function f            
def LU_solver(L, U, f):
    n = np.shape(L)[0]
    y = np.zeros(n)
    u = np.zeros(n)
    y[0] = f[0]
    
    # forward solving 
    for i in range(1,n):
        y[i] = f[i]
        for j in range(i):
            y[i] -= L[i][j]*y[j]
    u[-1] = y[-1] / U[-1][-1]
    
    # backward solving
    for k in range(n-2, -1, -1):
        u[k] = y[k]
        for l in range(n-1, k, -1):
            u[k] -= U[k][l]*u[l]
        if U[k][k] != 0:
            u[k] = u[k] / U[k][k]
    return u

# function that implements the Successive Overrelaxation Method 
# with omega = 1.5
# given a matrix A, vector f and a max number of iterations            
def SOR(A, f, max_iterations):
    # counter and omega
    iterations = 0
    omega = 1.5
    # initializing u, residual, norms of residuals
    # error and counter array
    n = np.shape(A)[0] 
    u = np.zeros(n)
    r = np.zeros(n)
    res_norm = np.zeros(max_iterations)
    M = np.zeros(max_iterations)
    E = np.zeros(max_iterations)
    
    # while max iterations not reached
    while iterations < max_iterations:
        # SOR algorithm
        for i in range(n):
            sigma = u[i]
            u[i] = f[i]
            for j in range(i):
                u[i] -= A[i][j] * u[j]
            for k in range(i+1,n):
                u[i] -= A[i][k] * u[k]
            if A[i][i] != 0:
                u[i] = u[i] / A[i][i]
                u[i] = (1-omega)*sigma + omega * u[i]
        # calculate residual
        r = f - np.matmul(A,u)
        # calculate norm of residual and save in array
        res = norm(r)
        res_norm[iterations] = res
        # calculate norm of f
        f_norm = norm(f)
        # determine norm of residual divided by norm of f
        # and save in array
        error = res / f_norm
        E[iterations] = error
        # save counter for plot
        M[iterations] = iterations
        iterations += 1

    return u, E, M, res_norm

# function that implements the GMRES method
# using the SOR(1.5) matrix as a preconditioner
def GMRES_SOR(A, f, tol=(10)**(-10)):
    n = np.shape(A)[0]
    # initialize array s for u's, v's and residuals
    U = []
    V = []
    residuals = []
    # append u_0 (zero vector)
    u_0 = np.zeros(n)
    U.append(u_0)
    # initialize Hessenberg
    H = np.zeros(shape=(n+1,n))
    # initialize e
    e = np.zeros(n+1)
    
    # make preconditioner matrix
    M = A.copy()
    omega = 1.5
    for i in range(n):
        for j in range(i+1,n):
            # 1/omega times diagonal
            M[i][i] = (1/omega) * A[i][i]
            if i < n-1:
                # zero above diagonal
                M[i][j] = 0
    # invert preconditioner
    M_inv = inv(M)

    # calculate r_0 and v_0
    r_0 = np.matmul(M_inv, f - np.matmul(A,u_0))
    v_0 = r_0 / norm(r_0)
    # append v_0 to V and norm of r_0/norm of f to residuals
    V.append(v_0)
    norm_f = norm(f)
    residuals.append(norm(r_0)/norm_f)
    # set e_0 equal to norm of r_0
    e[0] = norm(r_0)

    # apply GMRES to determine V[k+1] and entries of H
    for k in range(n):
        v = np.matmul(A, V[k])
        V.append(v)
        for i in range(k+1):
            H[i][k] = np.matmul(V[k+1],np.transpose(V[i])) 
            V[k+1] -= H[i][k] * V[i]
        H[k+1][k] = norm(V[k+1])
        if (H[k+1][k] != 0 and k != n-1):
            V[k+1] = V[k+1] / H[k+1][k]
        
        # solve least squares of H-e and determine u
        y = scipy.sparse.linalg.lsqr(H, e, atol=1e-64, btol = 1e-64)[0] 
        u = np.transpose(V[k+1])*y 
        # append u to array of u's 
        U.append(u)
        
        # determine residual
        r = f - np.matmul(A,u)
        # determine norm of residual, divide by norm f
        # and add to residuals array
        res = norm(r)
        error = res / norm_f
        residuals.append(error)
        
        # stop if array is smaller than tolerance
        if error < tol:
            return U, residuals
    
    # stop when k reaches n
    return U, residuals

# function that implements the GMRES method
# using the ILU matrix as a preconditioner    
def GMRES_ILU(A, f, tol=(10)**(-10)):
    n = np.shape(A)[0]
    # initialize array s for u's, v's and residuals
    U = []
    V = []
    residuals = []
    # append u_0 (zero vector)
    u_0 = np.zeros(n)
    U.append(u_0)
    # initialize Hessenberg
    H = np.zeros(shape=(n+1,n))
    # initialize e
    e = np.zeros(n+1)
    
    # make preconditioner matrix ILU
    L = np.zeros(shape=(n,n))
    for z in range(n):
        L[z][z] = 1
    
    Up = np.copy(A)
    
    # LU with zeroes where A is zero
    for i in range(n):
        for j in range(i+1, n):
            if A[j][i] != 0:
                L[j][i] = Up[j][i] / Up[i][i]
            for k in range(n):
                if A[j][k] != 0:
                    Up[j][k] = Up[j][k] - L[j][i]*Up[i][k]
    # invert preconditioner matrix
    M_inv = np.matmul(inv(Up),inv(L))

    # calculate r_0 and v_0
    r_0 = np.matmul(M_inv, f - np.matmul(A,u_0))
    v_0 = r_0 / norm(r_0)
    # append v_0 to V and norm of r_0/norm of f to residuals
    V.append(v_0)
    norm_f = norm(f)
    residuals.append(norm(r_0)/norm_f)
    # set e_0 equal to norm of r_0
    e[0] = norm(r_0)
    
    # apply GMRES to determine V[k+1] and entries of H
    for k in range(n):
        v = np.matmul(A, V[k])
        V.append(v)
        for i in range(k+1):
            H[i][k] = np.matmul(V[k+1],np.transpose(V[i])) 
            V[k+1] -= H[i][k] * V[i]
        H[k+1][k] = norm(V[k+1])
        if (H[k+1][k] != 0 and k != n-1):
            V[k+1] = V[k+1] / H[k+1][k]
        
        # solve least squares of H-e and determine u
        y = scipy.sparse.linalg.lsqr(H, e, atol=1e-64, btol = 1e-64)[0] 
        u = np.transpose(V[k+1])*y 
        # append u to array of u's 
        U.append(u)
        
        # determine residual
        r = f - np.matmul(A,u)
        # determine norm of residual, divide by norm f
        # and add to residuals array
        res = norm(r)
        error = res / norm_f
        residuals.append(error)
        
        # stop if array is smaller than tolerance
        if error < tol:
            return U, residuals
    
    # stop when k reaches n
    return U, residuals
            
        
        
        
        