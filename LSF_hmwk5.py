import matplotlib.pyplot as plt
# import scipy.linalg as la
# # from scipy.constants import c
# import math
import scipy.optimize as optimization

def func(x, a, b, c): #I copied this function from scipy's lsq curve fit page https://python4mpia.github.io/fitting_data/least-squares-fitting.html
    return a + b*x + c*x*x

def lsf_quick(xdata, ydata):
    plt.plot(xdata, ydata, '*', markersize= 10, label= "Eigenvalues vs v")
    popt, pcov= optimization.curve_fit(func, xdata, ydata)
    print(popt, pcov)
    plt.plot( xdata, func(xdata, *popt), 'r--', label="Curvefit") #from the documentation for this function
    plt.xlabel("v")
    plt.ylabel("Eigenvalues ")
    plt.legend()
    plt.show()
    return
