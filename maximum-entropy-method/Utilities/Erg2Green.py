# Erg2Green.py
# Yanfei Tang

# The following paper will be useful:
# Xin Wang, Emanuel Gull, et.al PRB 80, 045101 (2009)
# Sebastian Fuch, Emanule Gull, et.al PRB 83, 235113 (2011)
#

import numpy as np
import os
import sys
from scipy import interpolate

def KKintegral(data):
    """
    Do Kramers- Kronig integral to get the real part of Self energy.
            1    {   X2(w')
    X1(w) = -- P | ---------  dw'
            pi   }  w' -  w

    P is cauchy principal value. X1(w) is real part and X2(w) is imaginary part.
    The formular used to compute the integral is:

            1  {   X2(w') - X2(w)         X2(w)      b - w
    X1(w) = -- | ---------------- dw'  +  -----  ln(------)
            pi }       w'- w                pi       w - a

    where a and b is boundary of integral.

    argument:
    --------------------------------------
    data: N by 2 matrix, first column is w, second column is response function.

    """
    #f = interpolate.interp1d(data[:,0], data[:,1])
    #xnew = np.arange(data[0,0], data[-1,0], 0.1)
    #ynew = f(xnew)
    deri = centraldiff(data[:,1], data[:,0])
    w = data[1:-1,0]
    dw = w[1:] - w[:-1]
    dw = np.append(dw, dw[-1])
    X2 = data[1:-1,1]
    X1 = np.zeros(len(w), dtype = "float64")
    for i in range(len(w)):
        sum = 0.0
        for j in range(len(w)):
            if i == j:
                sum += deri[i] * dw[i]
            else:
                sum += (X2[j] - X2[i] )/(w[j] - w[i]) * dw[j]
        X1[i] = sum/np.pi + X2[i]/np.pi * np.log((data[-1,0] - w[i]) / (w[i] - data[0,0] ))
    return w ,X1



def centraldiff(y , x):
    """
    Central differnce:
    f'(x) = f(x + 1/2 h) - f(x - 1/2 h) / h
    A n-lenght of array y will return (n-2)-length arrya
    """
    length = len(y) - 2
    z = np.zeros(length, dtype = "float64")
    for i in range(length):
        z[i] = y[i+2]- y[i]
        dx = x[i+2] - x[i]
        z[i] = z[i]/dx
    return z


def dataForm(fname = "[0,-pi]/[0,-pi]beta[12.5]renorm.datmaxent.dat" , U = 8.0, n = 0.95):
    if os.path.isfile(fname):
        data = np.loadtxt(fname)
        data[:,1] *=  np.pi* (-1.0) * U * U * (1.0 - n/2.0) * n/2.0
        w, realpart = KKintegral(data)
        realpart += U * (n/2.0 - 0.5)
        selferg = realpart + 1j * data[1:-1,1] #* (-1.0) * np.pi * U * U * (1.0 - n/2.0) * n/2.0

    return data[1:-1,0],selferg

def greenFun(fname = "[0,-pi]/[0,-pi]beta[12.5]renorm.datmaxent.dat", U = 8, n=0.95, kx = 0, ky = np.pi, mu = -2.0070):
    w, selferg = dataForm(fname, U, n)
    eps = -2*1.0 * (np.cos(kx) + np.cos(ky))
    G = 1.0/ (w - eps - selferg + mu)
    return w, G





if __name__ == "__main__":
    w1, G1 = greenFun(fname = "[0,-pi]/[0,-pi]beta[12.5]renorm.datmaxent.dat", U = 8, n=0.95, kx = 0, ky = -np.pi, mu = -2.0070)
    w1, G2 = greenFun(fname = "[0,0]/[0,0]beta[12.5]renorm.datmaxent.dat", U = 8, n=0.95, kx = 0, ky = 0, mu = -2.0070)
    w1, G3 = greenFun(fname = "[pi,-pi]/[pi,-pi]beta[12.5]renorm.datmaxent.dat", U = 8, n=0.95, kx = np.pi, ky = -np.pi, mu = -2.0070)
G_tot = G1*2 + G2 + G3