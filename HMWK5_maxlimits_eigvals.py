import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
# from scipy.constants import c
from scipy.optimize import fmin
import scipy.integrate as integrate
import math
from EigDiag import EigDiag
from LSF_hmwk5 import lsf_quick
from HMWK4 import morsePE as morsePE_o


# hmwk 4 problem 2- this code currently works!
class Matrix():  # this will make my matrix elements. Every matrix element will be an object of this class
    masse = 0.00054858
    u = 1.00784 / (2 * masse)
    w = 0.01822

    def __init__(self, m, n, L, re):
        self.m = m
        self.n = n
        self.L = L
        self.re = re
        self.reshift = self.re #the way i am handling the shift is not good!, could be incorporated as part of the function definition

    def shiftre(self, amt_to_shift):
        self.reshift= self.re + amt_to_shift
        return

    def wave_m(self, x):  # wavefunction_m
        inside = self.m * math.pi * (x - (self.reshift + self.L / 2)) / self.L
        return math.sqrt(2 / self.L) * math.sin(inside)

    def wave_n(self, x):  # wavefunction_n
        inside = self.n * math.pi * (x - (self.reshift + self.L / 2)) / self.L
        return math.sqrt(2 / self.L) * np.sin(inside)  # changing this line to numpy.sin rather than math.sin
        # return np.sqrt(2./self.L)*np.sin(((float(self.n)*x/self.L)+(float(self.n)/2.))*np.pi) #from noahs -results in diff wavefunctions!

    def tsol(self):  # I already know what <n|T|n> equals
        return ((self.n ** 2) * (math.pow(math.pi, 2)) / (2 * Matrix.u * math.pow(self.L, 2)))

    # def vsol(self, x):  # V(x)
    #     return (0.5 * Matrix.u * math.pow(Matrix.w, 2) * math.pow(x,2))

    # def mwaven_int(self):  # <m|V(x)|n>
    #     fun_i = lambda x: self.wave_m(x) * self.vsol(x) * self.wave_n(x)
    #     return integrate.quad(fun_i, 0, self.L)

    # # i am throwing these in for the plotting part of the homework, they have nothing to do with the matrix elements of this class
    # def energylev(self, v):  # function for finding the exact energy levels
    #     return 1 * Matrix.w * (v + 0.5)


class Morse(Matrix):
    # HMWK 4
    def __init__(self, m, n, L, re, de, a, j):
        super().__init__(m, n, L, re)
        self.de = de
        self.a = a
        self.j = j

    def shiftre(self, amt_to_shift):
        self.reshift= self.re + amt_to_shift
        return

    def morsePE(self, x):  # changed to effective potential V_eff
        """ V(r)"""
        return self.de * np.power((1 - np.exp(-self.a * (x - self.re))), 2)

    def cenPE(self, x):
        return (self.j * (self.j + 1)) / (2 * self.u * np.power(x, 2))

    # experimental function! lets see if this works
    def mwaven_int_EX(self):  # <m|V(x)|n>
        """fun(x)"""
        fun_i = lambda x: self.wave_m(x) * (self.morsePE(x) + self.cenPE(x)) * self.wave_n(x)
        return integrate.quad(fun_i, self.reshift - self.L / 2, self.reshift + self.L / 2)

    # the centrifugal potiential

    def cenPE_int(self):  # J are our rotational energy levels
        fun_i = lambda x: self.wave_m(x) * self.cenPE(x) * self.wave_n(x)
        return integrate.quad(fun_i, self.reshift - self.L / 2, self.reshift + self.L / 2)

    # def exact_eigval(self,v):
    #     we= (self.a/2*math.pi*c)*math.sqrt(2*self.de/self.u)
    #     wexe= math.pow(we,2)/4*self.de
    #     return we*(v+1/2)-wexe*(v+1/2)**2


def main():
    l_val = 5.8  # 1.8 is working okay!
    basis = 50
    bigm = np.zeros((basis, basis))  # initialize the matrix
    de = 4.7 / 27.211  # convert from eV to hartress
    # de = 4.7 #in eV
    a = 1.0  # in bohrs^-1
    re = 1.4  # in bohrs
    j=0
    shiftre=1.4
        # for l_val in np.arange(2,2.8,0.1):
    for m in range(1, basis + 1):  # iterate m through a range of values from 1 to 20
        for n in range(1, basis + 1):  # iterate n through a range of values from 1 to 20
            matrix_elem = Morse(m, n, l_val, re, de, a, j)  # create the Matrix element object
            matrix_elem.shiftre(shiftre)
            if m == n:  # if m == n then H= <n|T|n>
                bigm[m - 1, n - 1] = matrix_elem.tsol() + matrix_elem.mwaven_int_EX()[0]
                # t_mat[m - 1, n - 1] = matrix_elem.tsol()
                # python uses 0-based indexing so we need to subtract m,n by 1
            else:  # if m =/= then H= <m|V|n>, because <m|T|n> = 0 when m =/= n
                bigm[m - 1, n - 1] = matrix_elem.mwaven_int_EX()[0]

    bigm_diag, sortedeig, evecs = EigDiag(bigm)
    # this function will find the eigenvalues, sort them, and put in a diagonal matrix
    # print(f" eigenvalues in diagonal matrix \n {bigm_diag} \n and ")
    print(f"J= {j} : \n eigenvalues in a list {sortedeig[0:13] * 219474.6} in cm^-1 \n")  # print using an fstring
    lsf_quick(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]), sortedeig[0:15] * 219474.6)

    # for homework3, plotting the wavefunctions
    x_PIB = np.linspace(matrix_elem.reshift - l_val / 2, matrix_elem.reshift+ l_val / 2, 100, endpoint=True)  # based off of Issac's HMWK 3 code!
    PIB_basis = np.zeros((len(x_PIB), basis))
    for n in range(1, basis + 1):
        PIB_elem = Matrix(n, n, l_val, re)
        PIB_elem.shiftre(shiftre)
        PIB_basis[:, n - 1] = PIB_elem.wave_n(x_PIB)
    wavevalues = PIB_basis @ evecs

    fig = plt.figure()
    x = np.arange(0.1, 20, 0.1)
    ax1 = plt.subplot(111)

    ax1.plot(x, matrix_elem.cenPE(x) + matrix_elem.morsePE(x), label="Effective Morse Potential")

    ##plotting lines for the De and bounds of my box
    ax1.hlines(de, 0, 20, linestyles="dashed")
    ax1.vlines(matrix_elem.reshift, 0, 20, linestyles= "dotted", colors="m")
    ax1.vlines(matrix_elem.reshift - l_val / 2, 0, 2, colors="r", linestyles="dashed")
    ax1.vlines(matrix_elem.reshift + l_val / 2, 0, 2, colors="r", linestyles="dashed")

    ##plotting wavefunctions
    plt.xlim(0, 11)
    plt.ylim(-0.01, 0.25)
    ax1.set_aspect("auto")
    for n in range(0, 13):
        ax1.plot(x_PIB, wavevalues[:, n] / 100 + sortedeig[n])

    ax1.set_ylabel("Energy (cm^-1)")
    ax1.set_xlabel("R (bohrs)")
    ax1.legend(fontsize= "small", loc= "upper right")
    ax1.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
