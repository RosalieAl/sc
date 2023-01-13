# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:15:29 2022

@author: hester
"""

#import matplotlib as plt
import math
import numpy as np
import scipy.linalg as lg
import scipy as sc
import functions as f
import time
from numpy.linalg import norm
from matplotlib import pyplot as plt


def main():
    
    p = 5
    h = 1/(2**p)
    n = 2**p
    A_h = f.matrix_2d(f.matrix_1d(n),f.semi_I(n))
    # m =  100
    # start_time1 = time.time()
    #L, U = f.LU(A_h)
    #print(plt.pyplot.imshow(L))
    #print(plt.pyplot.imshow(A_h))
    #print(A_h)
    #print("--- %s seconds ---" % (time.time() - start_time1))
    #start_time2 = time.time()
    f_h = f.functionvalues2(n)
    #u_h = f.LU_solver(L,U,f_h)
    #print(u_h)
    #print("--- %s seconds ---" % (time.time() - start_time2))
    #start_time1 = time.time()
    #u_h2, E, M, residuals = f.SOR(A_h,f_h,m) 
    #print(u_h2)
    #print("--- %s seconds ---" % (time.time() - start_time1))
    # print(E)
    # i_convergence = 0
    # for i in range(m):
    #     if E[i] < 10**(-10) and E[i-1] >= 10**(-10):
    #         i_convergence = i
    #         print(i_convergence)
    #         break

    # for j in range(i_convergence-5, i_convergence):
    #     print(residuals[j]/residuals[j-1])
    # print(residual)
    
    # res = norm(residual)
    # f_norm = norm(f_h)
    # error = res / f_norm
    # print(error)
    
    # u_ex = f.u_2d(n)
    
    # length = len(u_ex)
    # max_distance = 0 
    
    # for i in range(length):
    #     if abs(u_h[i] - u_ex[i]) > max_distance:
    #         max_distance = abs(u_h[i] - u_ex[i])
    
    #return max_distance, u_h
    # print(max(M))
    # plt.yscale("log")
    # plt.scatter(M, E)
    # plt.xlabel('$m$')
    # plt.ylabel('$log_{10} Error$')
    # plt.axis([min(M), max(M), min(E), max(E)])
    # plt.show()
    
    #vanaf hier GMRES_SOR
    # start_time2 = time.time()
    #U, res = f.GMRES_SOR(A_h, f_h)
    # print("--- %s seconds ---" % (time.time() - start_time2))
    # print(U)
    #print(res)
    # iterations = len(res)
    # print(iterations)
    # M = np.zeros(iterations)
    # for i in range(iterations):
    #     M[i]=i
    # plt.yscale("log")
    # plt.scatter(M, res)
    # plt.xlabel('$m$')
    # plt.ylabel('$log_{10} Error$')
    # plt.axis([min(M), max(M), min(res), max(res)])
    # plt.show()
    
    #vanaf nu GMRES met ILU
    start_time2 = time.time()
    U, res = f.GMRES_ILU(A_h, f_h)
    print("--- %s seconds ---" % (time.time() - start_time2))
    #print(U)
    print(res)
    iterations = len(res)
    print(iterations)
    M = np.zeros(iterations)
    for i in range(iterations):
        M[i]=i
    plt.yscale("log")
    plt.scatter(M, res)
    plt.xlabel('$m$')
    plt.ylabel('$log_{10} Error$')
    plt.axis([min(M), max(M), min(res), max(res)])
    plt.show()
    return
    
if __name__ == "__main__":
    main()