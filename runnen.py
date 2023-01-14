# -*- coding: utf-8 -*-

import matplotlib as plt
import math
import numpy as np
import scipy.linalg as lg
import scipy as sc
import functions as f
import time
from numpy.linalg import norm
from matplotlib import pyplot as plt


def main():
    
    # define p, h and n
    p = 2
    h = 1/(2**p)
    n = 2**p
    
    # initialize the matrix A^h
    A_h = f.matrix_2d(f.matrix_1d(n),f.semi_I(n))
    print(A_h)
    
    # calculate the amount of nonzero elements of A_h
    nonzerosA = np.count_nonzero(A_h, axis=None, keepdims=False)
    print(nonzerosA)
    
    # define the amount of iterations
    m =  100
    
    # calculate LU-decomposition of A^h
    # calculate the time it takes to factorize
    start_time1 = time.time()
    L, U = f.LU(A_h)
    print("--- %s seconds ---" % (time.time() - start_time1))
    print(plt.pyplot.imshow(L))
    print(plt.pyplot.imshow(A_h))
    
    # calculate the amount of nonzero elements of L and the fill-in ratio
    nonzerosL = np.count_nonzero(L, axis=None, keepdims=False)
    print(nonzerosL)
    print(nonzerosL/nonzerosA)
    
    # determine f^h
    f_h = f.functionvalues2(n)
    
    # calculate solution u using the LU-decomposition
    # calculate the time it takes to solve
    start_time2 = time.time()
    u_h = f.LU_solver(L,U,f_h)
    print("--- %s seconds ---" % (time.time() - start_time2))
    print(u_h)
        
    # calculate solution u using the Successive Overrelaxation Method
    # and calculate the time it takes
    start_time3 = time.time()
    u_h2, E, M, residuals = f.SOR(A_h,f_h,m) 
    print(u_h2)
    print("--- %s seconds ---" % (time.time() - start_time3))
    print(E)
    # finding the amount of iterations it takes SOR to converge
    # so we can get the residual reduction factor for the last five iterations
    i_convergence = 0
    for i in range(m):
        if E[i] < 10**(-10) and E[i-1] >= 10**(-10):
            i_convergence = i
            print(i_convergence)
            break
    # print the residual reduction factor for the last five iterations of SOR
    for j in range(i_convergence-5, i_convergence):
        print(residuals[j]/residuals[j-1])
    
    # implementation assignment 2
    u_ex = f.u_2d(n)
    length = len(u_ex)
    max_distance = 0 
    # calculating the max-norm of the discretization error 
    for i in range(length):
        if abs(u_h[i] - u_ex[i]) > max_distance:
            max_distance = abs(u_h[i] - u_ex[i])
    print(max_distance)

    # SOR: make a logarithmic plot of ||r_m||2 /||fh||2 versus m
    plt.yscale("log")
    plt.scatter(M, E)
    plt.xlabel('$m$')
    plt.ylabel('$log_{10} Error$')
    plt.axis([min(M), max(M), min(E), max(E)])
    plt.show()
    
    # implement the GMRES with SOR as preconditioner
    # and calculate the time it takes
    start_time4 = time.time()
    Ugs, resgs = f.GMRES_SOR(A_h, f_h)
    print("--- %s seconds ---" % (time.time() - start_time4))
    print(Ugs)
    print(resgs)
    # number of iterations
    iterations = len(resgs)
    print(iterations)
    # make iterations array for the plot
    Mgs = np.zeros(iterations)
    for i in range(iterations):
        Mgs[i]=i
    # GMRES SOR: make a logarithmic plot of ||r_m||2 /||fh||2 versus m
    plt.yscale("log")
    plt.scatter(Mgs, resgs)
    plt.xlabel('$m$')
    plt.ylabel('$log_{10} Error$')
    plt.axis([min(Mgs), max(Mgs), min(resgs), max(resgs)])
    plt.show()
    
    # implement the GMRES with ILU as preconditioner
    # and calculate the time it takes
    start_time5 = time.time()
    Ugi, resgi = f.GMRES_ILU(A_h, f_h)
    print("--- %s seconds ---" % (time.time() - start_time5))
    print(Ugi)
    print(resgi)
    # number of iterations
    iterations2 = len(resgi)
    print(iterations2)
    # make iterations array for the plot
    Mgi = np.zeros(iterations2)
    for i in range(iterations2):
        Mgi[i]=i
    # GMRES ILU: make a logarithmic plot of ||r_m||2 /||fh||2 versus m
    plt.yscale("log")
    plt.scatter(Mgi, resgi)
    plt.xlabel('$m$')
    plt.ylabel('$log_{10} Error$')
    plt.axis([min(Mgi), max(Mgi), min(resgi), max(resgi)])
    plt.show()
    return
    
if __name__ == "__main__":
    main()