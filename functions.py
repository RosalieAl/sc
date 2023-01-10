# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:32:01 2022

@author: hester
"""

import matplotlib as plt
import math
import numpy as np
import scipy.linalg as lg
from scipy.sparse import diags
from numpy.linalg import norm


def matrix_1d(n):
    h = 1/n
    matrix = np.zeros(shape=(n+1,n+1))
    
    # for i in range(n+1):
    #     for j in range(n+1):
    #         if i == j:
    #           matrix[i][j] = 4
    #         if i == j + 1:
    #             matrix [i][j] = -h - 2
    #         if i == j - 1:
    #             matrix[i][j] = h - 2
     
    np.fill_diagonal(matrix, 4)
    
    b = np.ones(n)
    np.fill_diagonal(matrix[1:], (-h-2)*b)
    np.fill_diagonal(matrix[:,1:], (h-2)*b)
           
    matrix[0][0] = 2*(h**2)
    matrix[n][n] = 2*(h**2)
    
    for j in range(1,n+1):
        matrix[0][j] = 0
    for k in range(0,n):
        matrix[n][k] = 0
       
    matrix = matrix / (2 * (h**2))
    return matrix

def functionvalues1(n):
    h = 1/n
    values = np.zeros(n+1)
    # boundary
    values[0] = 0
    values[n] = math.sin(1)
    for i in range(1,n):
        values[i] = math.sin(i*h) + math.cos(i*h)
    return values

def u(n):
    h = 1/n
    u_values = np.zeros(n+1)
    for i in range(n+1):
        u_values[i] = math.sin(i*h)
    return u_values

def semi_I(n):
    id = np.zeros(shape=(n+1,n+1))
    for i in range(1,n):
        id[i][i] = 1
    return id
        
def matrix_2d(A,B):
    n = np.shape(A)[0]
    matrix = np.kron(A,B) + np.kron(B,A)
    for i in range((n)**2):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
    return matrix

def reordering(A,n):
    B = A.copy()
    j = 0
    for i in range(n+1,(n+1)**2-(n+1),1):
        r = B[i][j]
        B[i][j]=B[i][j+n-1]
        B[i][j+n-1]=r
        r = B[i][i+2]
        B[i][i+2]=B[i][i+n+1]
        B[i][i+n+1]=r
        j+=1
    return B
        
def functionvalues2(n):
    h = 1/n
    f = np.zeros(shape=(n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                f[i][j] = 0
            elif i == n:
                f[i][j] = math.sin(j*h)
            elif j == n:
                f[i][j] = math.sin(i*h)
            else:
                f[i][j] = ((i*h)**2+(j*h)**2)*math.sin((i*h)*(j*h)) + (i*h + j*h) *  math.cos ((i*h)*(j*h))
    f = f.flatten('F')
    return f
        
def u_2d(n):
    h = 1/n
    u = np.zeros(shape=(n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            u[i][j] = math.sin((i*h)*(j*h))
    u = u.flatten('F')
    return u
        
# def LU_scipy(A):
#     P, L, U = lg.lu(A)
#     return L,U
        
#def LU(A):
#    n = np.shape(A)[0]
#    L = np.zeros(shape=(n,n))
#    for z in range(n):

    #        L[z][z] = 1
#    U = np.copy(A)
    
#    for i in range(n):
#        for j in range(i+1, n):
#            L[j][i] = U[j][i] / U[i][i]
#            for k in range(n):
#                U[j][k] = U[j][k] - L[j][i]*U[i][k]
#    return L, U

def LU(A):
    n = np.shape(A)[0]
    
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    
    #Loop over rows
    for i in range(n):
            
        #Eliminate entries below i with row operations 
        #on U and reverse the row operations to 
        #manipulate L
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]
        
    return L,U
            
def LU_solver(L, U, f):
    n = np.shape(L)[0]
    y = np.zeros(n)
    u = np.zeros(n)
    
    y[0] = f[0]
    
    for i in range(1,n):
        y[i] = f[i]
        for j in range(i):
            y[i] -= L[i][j]*y[j]
    
    u[-1] = y[-1] / U[-1][-1]
    
    for k in range(n-2, -1, -1):
        u[k] = y[k]
        for l in range(n-1, k, -1):
            u[k] -= U[k][l]*u[l]
        #for l in range(k, n):
        #    u[k] -= U[k][l]*u[l]
        if U[k][k] != 0:
            u[k] = u[k] / U[k][k]
    return u

# def matrix_diag_ord(n):
#     matrix = np.zeros(shape=((n+1)**2,(n+1)**2))
#         for i in range((n+1)**2):
#             for j in range((n+1)**2):
                    
    
#     return matrix
            
def SOR(A, f, max_iterations):
    iterations = 0
    omega = 1.5
    n = np.shape(A)[0]
    u = np.zeros(n)
    r = np.zeros(n)
    M = np.zeros(max_iterations)
    E = np.zeros(max_iterations)
    
    while iterations < max_iterations:
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
        r = f - np.matmul(A,u)
        res = norm(r)
        f_norm = norm(f)
        error = res / f_norm
        # print(error)
        E[iterations] = error
        # print(E[iterations])
        M[iterations] = iterations
        iterations += 1
    
    # r = f - np.matmul(A,u)
    return u, E, M
            
            
            
        
        
        
        