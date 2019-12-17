import sys
import os
import numpy as np
import time
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize
from numpy.linalg import inv

def gaussian(w, mu = 0.0, sig = 4.0):
    """
    A gaussian distribution.
    """
    return 1.0/sig/np.sqrt(2* np.pi) * np.exp(-np.power(w - mu, 2.)/ 2. / np.power(sig, 2.))

def straightline(w):
    """
    A straight line.
    """
    return np.ones(len(w))/(w[-1] - w[0])

class Maxent(object):
    """
    Maximum entropy method of analytical continuation of Imaginary frequency Green's function data.
    Attributes:
    ----------
    wn: ndarray
        Matsubara frequency, e.g. fermionic case wn = (2n+1)/pi/beta.
    G: ndarray, dtype = 'complex'
        imaginary frequency Green's function.
    K: ndarray, dtype = 'complex'
        Kernel.
    w: ndarray
        real frequency.
    aveG: ndarray, dtype = 'complex'
        average of imaginary frequency Green's function.
    stdG: ndarray, dtype = 'complex'
        standard deviation of imaginary frequency Green's function.
    numMfre: int
        number of Matsubara frequency.
    numbins: int
        number of bins, e.g. 20 sets of QMC data.
    numRfre: int
        number of real frequency.
    specF: ndarray
        spectral function.
    defaultM: ndarray
        default model.
    wmin: float
        minimum real frequency.
    wmax: float
        maximum real frequency.
    alpha: float
        tuning parameter to adjust the trade-off between the fit of the data and the nearness of the default model.
    dw: float
        the discrete space of w.
    tol: float
        tolerence of the minimization.
    std: tuple (bool, float)
        whether considering the standard deviation into the chi sqaure computation.
    prob: float
        the probability for this alpha.
    alphamin and alphamax and dalpha: float
        the period of alpha, and the interval of alpha
    alphas: ndarray
        the alpha that are examed.
    allSpecFs: ndarray
        all the spectral functions for different alpha
    allProbs: ndarray
        probabilities for each different alpha
    aveSpecFs: ndarry
        the average of the spectral functions
    mu: float
        parameter in Levenberg-Marquart algorithms.
    maxIteration: int
        Maximum iteration in Lenvenberg-Marquart algorithm.
    """

    def __init__(self, filename = "data", column = 41, numMfre = 50, numRfre = 200, wmin = -15, wmax = 15, defaultModel = "gaussian", tol = 1e-10, std = (False, 1.0), alphamin = -8, alphamax = 0, numAlpha = 100, minimizer = "Bryan", draw = True):
        """
        Contructor
        """
        self.start_time = time.time()
        self.numMfre = numMfre
        self.numRfre = numRfre
        self.wmin = wmin
        self.wmax = wmax
        self.tol = tol
        self.std = std
        self.alphamin = alphamin
        self.alphamax = alphamax
        self.numAlpha = numAlpha
        self.minimizer = minimizer
        self.draw = draw
        self.mu = 0.1
        self.maxIteration = 1e4
        self.allSpecFs, self.allProbs = [], []
        self.readfile(filename, column, numMfre)
        self.calMeanAndStd()
        self.createKernel()
        self.rotation()
        self.compUXiV()
        if defaultModel == 'gaussian':
            self.specF = gaussian(self.w)/self.normalize(gaussian(self.w))
            self.defaultM = self.specF
        elif defaultModel == 'straightline':
            self.specF, self.defaultM = straightline(self.w), straightline(self.w)
        elif defaultModel == 'input':
            data = np.loadtxt(sys.argv[2])
            self.specF, self.defaultM = data[:, 1], data[:, 1]


        else:
            print "Usage: Maxent(filename, column, numRfre, wmin, wmax, defaultModel, tol, std, alphamin, alphamax, numAlpha)"
            print "defaultModel must 'g' or 's'!"
            sys.exit(0)
        print "Use --- %s seconds --- before optimization..." %(time.time() - self.start_time)

    def getAllSpecFs(self):
        """
        Compute all the spectral functions by looping all the alphas.
        """
        self.alphas = np.logspace(self.alphamin, self.alphamax, self.numAlpha)
        # Use uniformed space integral for \int P(alpha)dalpha;
        self.dalpha = self.alphas[1:] - self.alphas[:-1]
        self.dalpha /= 2.0
        self.dalpha = np.insert(self.dalpha, 0, 0.0) + np.append(self.dalpha, 0.0)
        for self.alpha in self.alphas:
            self.getSpecF()
            self.calProb()
            self.allSpecFs.append(self.specF)
            self.allProbs.append(self.prob)
            print "Finish alpha = %s.\n" %(self.alpha)

        self.allProbs = np.array(self.allProbs)
        self.allSpecFs = np.array(self.allSpecFs)
        self.allProbs = self.allProbs/np.sum(self.allProbs*self.dalpha)
        self.aveSpecFs = np.dot(self.allSpecFs.transpose() * self.dalpha, self.allProbs)
        print "Optimization ends. Use --- %s seconds ---" %(time.time() - self.start_time)

    def saveObj(self):
        """
        save the object as .p file
        And plot the figure.
        rFre:
            real frequency
        tSpec:
            true spectral function

        """
        if sys.argv[1][-1] == "G":
            dirname = sys.argv[1][:-1]
        else:
            dirname = sys.argv[1]
        #---------save object pickle-----------------
        #ofile = open(dirname + "object.p", 'w')
        #pickle.dump(self, ofile, -1)
        #ofile.close()
        #---------save dict pickle-------------------
        dict_data = {"wn": self.wn,
                     "w": self.w,
                     "aveG": self.aveG,
                     "stdG": self.stdG,
                     "numMfre": self.numMfre,
                     "numbins": self.numbins,
                     "numRfre": self.numRfre,
                     "defaultM": self.defaultM,
                     "alphas": self.alphas,
                     "allSpecFs": self.allSpecFs,
                     "allProbs": self.allProbs,
                     "aveSpecFs": self.aveSpecFs,
                     }

        if os.path.isfile("./" + dirname + "A"):
            rFre, tSpecF = [], []
            with open(dirname+ "A", "r") as ifile:
                for line in ifile:
                    term = line.split()
                    rFre.append(float(term[0]))
                    tSpecF.append(float(term[1]))
            rFre, tSpecF = np.array(rFre), np.array(tSpecF)
            dict_data["rFre"] = rFre
            dict_data["tSpecF"] = tSpecF
            plt.plot(rFre, tSpecF, "r--", alpha = 0.5, label = "TestFunc")
            ifile.close()

        ofile = open(dirname + ".p", 'w')
        pickle.dump(dict_data, ofile, -1)
        ofile.close()

        result = open(dirname + "maxent.dat", 'w')
        for i in range(len(self.w)):
            result.write(str(self.w[i]) + '\t' + str(self.aveSpecFs[i]) + "\n")
        result.close()

        result = open(dirname + "Palpha.dat", 'w')
        for i in range(len(self.alphas)):
            result.write(str(self.alphas[i]) + '\t' + str(self.allProbs[i]) + "\n")
        result.close()

        if self.draw:
            plt.plot(self.w, self.aveSpecFs, "b->", alpha = 0.8, label = "Maxent")
            plt.xlabel(r"$\omega$")
            plt.ylabel(r"$A(\omega)$")
            plt.legend()
            plt.savefig("./" + dirname + "Comparison.pdf")
            plt.show()



    def readfile(self, filename, column, numMfre):
        """
        read data from file:
        file format:
        #   1   2   3   4   5  ... column
        1  wn reG imG reG imG ...  imG
        2  wn ... ... ... ... ...  imG
        ...
        row
        """
        if column/2 < 2*numMfre:
            print "Shut Down immediately! number of samples must be larger than or equal to 2 times of number of Matsubara frequencies."
            sys.exit(0)
        self.wn, self.G = [], []
        with open(filename, 'r') as inputfile:
            for line in inputfile:
                a, G = line.split(), []
                self.wn.append(float(a[0]))
                for i in np.arange(0, column-1, 2):
                    b = float(a[i+1])
                    c = float(a[i+2])
                    G.append(b + 1j*c)
                self.G.append(G)
        self.wn, self.G = np.array(self.wn), np.array(self.G)
        rows, cols = self.G.shape
        self.wn = self.wn[rows/2-numMfre/2:rows/2+numMfre/2]
        self.G = self.G[rows/2-numMfre/2:rows/2+numMfre/2,:]
        inputfile.close()

    def calMeanAndStd(self):
        """
        From data (G) to calculate the average, standard deviation and covariance matrix.
        """
        self.numMfre, self.numbins = self.G.shape
        self.aveG = self.G.mean(1)
        A = np.array([[self.aveG[i]]*self.numbins for i in range(self.numMfre)]).reshape(self.numMfre, self.numbins) - self.G
        self.stdG = np.sqrt(np.sum(np.conjugate(A) * A, axis = 1)/(self.numbins - 1))
        self.cov = np.zeros([self.numMfre, self.numMfre], dtype = 'complex128')
        for l in range(self.numMfre):
            for k in range(self.numMfre):
                a = 0.0
                for j in range(self.numbins):
                    a += (self.aveG[l] - self.G[l][j] ) * (self.aveG[k] - self.G[k][j]).conjugate()
                self.cov[l][k] = a/(self.numbins-1)



    def createKernel(self):
        """
        The ill-conditioned kernel is:
        #         1
        # K = ---------.
        #     i*wn - w
        It is the roots of difficulty.
        And do singular value decomposition for the kernel.
        """
        self.numMfre = len(self.wn)
        self.w = np.linspace(self.wmin, self.wmax, self.numRfre)
        self.dw = self.w[1:] - self.w[:-1]
        self.dw = np.append(self.dw, self.dw[0])
        self.dw[0] /= 2.0
        self.dw[-1] /= 2.0
        self.K = np.zeros([self.numMfre, self.numRfre], dtype = 'complex128')
        for n in range(self.numMfre):
            for m in range(self.numRfre):
                self.K[n][m] = 1.0/(-self.w[m] + self.wn[n] * 1j)

    def rotation(self):
        """
        Rotate the kernel and data into diagonal representation.
        """
        w, v = np.linalg.eigh(self.cov)
        vt = v.transpose().conjugate()
        self.stdG = np.sqrt(w)
        self.K = np.dot(vt, self.K)
        self.aveG = np.dot(vt, self.aveG)


    def compUXiV(self):
        """
        Do the sigular value decomposition to the kernel matrix K.
        """

        self.U, self.Xi, self.Vt = np.linalg.svd(self.K.transpose().conjugate(), full_matrices = 0)
        self.rank = np.linalg.matrix_rank(self.K.transpose().conjugate())
        self.Xi = np.diag(self.Xi[:self.rank])
        self.Vt = self.Vt[:self.rank, :]
        self.V = self.Vt.transpose().conjugate()
        self.U = self.U[:, :self.rank]
        self.Ut = self.U.transpose().conjugate()


        self.M = np.dot(np.dot(self.Xi, self.Vt), np.diag(1.0/self.stdG/self.stdG))
        self.M = np.dot(self.M, self.V)
        self.M = np.dot(self.M, self.Xi)

    def chiSquare(self, specF):
        """
                   1
        \chi^2 = ----- |(aveG - K * A)/\sigma|^2
                   N
        """
        delta = self.aveG - np.dot(self.K, specF * self.dw)
        return np.real( np.sum( np.conjugate(delta) * delta/self.stdG/self.stdG )/self.numMfre )

    def restoreG(self, specF):
        """
        From the spectral function to find out the tilted G: G = K * A
        """
        return np.dot(self.K, specF * self.dw)

    def normalize(self, specF):
        """
        Find out the normalization facotr \int A(w) * dw
        """
        return np.sum( specF * self.dw )

    def objective(self, specF):
        """
        Q = 1/2 \chi^2 - \alpha * S
        considering the standard deviation or not.
        """
        delta = self.aveG - np.dot(self.K, specF * self.dw)
        if self.std[0]:
            return np.real(np.sum( np.conjugate(delta) * delta/self.stdG/self.stdG ))/2.0 + \
                   self.alpha * np.sum((specF * np.log(np.abs((specF)/self.defaultM)) - specF +  self.defaultM) * self.dw)
        else:
            return np.real(np.sum( np.conjugate(delta) * delta/self.std[1]/self.std[1] ))/2.0 + \
                   self.alpha * np.sum((specF * np.log(np.abs((specF)/(self.defaultM))) - specF +  self.defaultM) * self.dw)

    def getSpecF(self):
        """
        using SLSQP aka sequential least square quadratic programing or Bryan's method (R. K. Bryan, Eur. Biophys. J., 18 (1990) 165) to minimize the objective function to get the spectral function depending on alpha.
        """
        if self.minimizer == "SLSQP":
            cons = ({'type':'eq', 'fun': lambda x: np.sum(x*self.dw) - 1},
                    {'type':'ineq', 'fun': lambda x: x})
            res = minimize(self.objective, self.specF, args=(),  method='SLSQP', constraints = cons, tol = self.tol, options={'disp': True, 'maxiter':2000},)
            if not res.success:
                err = open("Error.txt", "a")
                err.write(res.message + "\n")
                err.write("This occurs at " + str(self.alpha) + "\n")
                err.close()
                self.specF = res.x

        elif self.minimizer == "Bryan":
            iteration = 0
            n = min(self.numMfre, self.numRfre)
            btemp = np.zeros(self.rank)
            self.specF = self.defaultM
            Qold = self.objective(self.defaultM)

            while True:
                iteration += 1
                T = np.dot(np.dot(self.Ut, np.diag(self.specF)), self.U)
                deri = -1.0/self.stdG/self.stdG * (self.aveG - self.restoreG(self.specF))
                g = np.dot(np.dot(self.Xi, self.Vt), deri)
                LHS = (self.alpha + self.mu) * np.diag(np.ones(self.rank)) + np.dot(self.M, T)
                RHS = -self.alpha * btemp - g
                deltab = np.dot(inv(LHS), RHS)
                criteria = np.dot(deltab, np.dot(T, deltab))

                if criteria < 0.2*sum(self.defaultM):
                    if iteration > self.maxIteration:
                        print "Exceeds maximum iteration in Levenberg-Marquart algorithms, exits. Make tolerance smaller."
                        break

                    btemp = btemp + deltab
                    al = np.dot(self.U, btemp)
                    self.specF = np.real(self.defaultM * np.exp(al))
                    Qnew = self.objective(self.specF)
                    if abs(Qnew - Qold)/Qold < self.tol:
                        print "{0} iterations in Levenberg-Marquart algorithms. Function evaluted: {1}, it exits properly.".format(iteration, Qnew)
                        break
                    Qold = Qnew
                    continue

                else:
                    self.mu *= 2
                    self.specF = self.defaultM
                    Qold = self.objective(self.defaultM)
                    btemp = np.zeros(self.rank)
                    print "parameter \mu is too small in the Levenberg-Marquart algorithms."
                    print "\mu is now adjusted to %s" %(self.mu)



    def calProb(self):
        """
        Compute the probablity for this paticular alpha. This probablity is not normalized.
        """
        if self.std[0]:
            cov = np.diag(self.stdG * self.stdG)
        else:
            cov = np.diag(np.ones(self.numMfre) * np.ones(self.numMfre) * self.std[1] * self.std[1])
        mat_a = np.dot(self.K.transpose().conjugate(), inv(cov))
        mat_a = np.dot(mat_a, self.K)
        vec_a = np.sqrt(np.abs(self.specF))
        imax, jmax = mat_a.shape
        mat_b = np.zeros((imax, jmax)) + 1j * 0
        for i in range(0, imax):
            for j in range(0, jmax):
                mat_b[i][j] = vec_a[i] * mat_a[i][j] * vec_a[j]
        S = np.linalg.eigvalsh(mat_b)
        expo = np.exp(-self.objective(self.specF))
        prod = np.prod(self.alpha/(self.alpha+S))


        self.prob = np.sqrt( prod ) * expo/self.alpha

        if np.isnan(self.prob):
            self.prob = 0.0




if __name__ == "__main__":
    """
    filename: the file that stores imaginary-frenquency Green's function data
    column: number of column in the file, an odd number.
    numMfre: number of Matsubara frequency used. (this number should be less than the rows in the file)
    numRfre: the number of grid for spectral function A(w).
    wmin: minimum real frequency
    wmax: maximun real frequency
    defaultModel: this parameter can be 'gaussian', 'straightline' or 'input'.
    tol: tolerance for minimization. 1e-12 for SLSQP; 1e-5 for Bryan
    std: whether or not use the standard deviation.
    alphamin: value of minimun alpha in log space.
    alphamax: value of maximum alpha in log space
    numAlpha: number of alphas.
    minimizer: "SLSQP" or "Bryan". 
    draw: whether or not draw the Maxent result graph.
    """
    Model = Maxent(filename = sys.argv[1], column = 201, numMfre = 50, numRfre = 201, wmin = -15, wmax = 15, defaultModel = 'gaussian', tol = 1e-5, std = (True, 1.0), alphamin = -1, alphamax = 2, numAlpha = 10, minimizer = "Bryan", draw = True)
    Model.getAllSpecFs()
Model.saveObj()