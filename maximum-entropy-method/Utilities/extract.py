# extract.py
# Author @ Yanfei Tang

import linecache
import numpy as np
import os

class extract(object):
    """
    This object is used to extract the data from DCA monte carlo code.
    """

    def __init__(self, filename = '7_G(K,w,b,b)', numSamples = 1000):
        """
        """
        self.keys = ['[0,0]', '[0,-pi]','[pi,0-]','[pi,-pi]']
        self.numSamples = numSamples
        self.readAllFiles(filename)
        self.writeFiles()


    def readFile(self, filename = "./i0/7_G(K,w,b,b)"):
        """
        read the 7_G(K,w,b,b) JSON file.
        """
        flag = True
        i = 38
        wn, G = [], {}

        for k in self.keys:
            G[k] = []
        while flag:

            line = linecache.getline(filename, i)
            flag = (line != "\n")
            i += 1
            if flag:
                a = line.split()
                wn.append(float(a[0]))
                for v,k in list(enumerate(self.keys)):
                    G[k].append(float(a[2*v+1]) + 1j * float(a[2*v+2]))
            else:
                break
        self.n = len(wn)
        return wn, G

    def readAllFiles(self,filename='7_G(K,w,b,b)'):
        """
        Read 1000 samples from directory i10 to i1009.
        """
        path = os.getcwd()
        filenames = []
        for i in range(0,100):
            dirname = '/i' + str(i+10) + '/'
            filenames.append(path+dirname+filename)
        self.G = {}
        for k in self.keys:
            self.G[k] = []
        for file in filenames:
            wn, iG = self.readFile(file)
            for k in self.keys:
                self.G[k].append(iG[k])
        self.wn = []
        for n in range(self.n):
            self.wn.append((2*(n-self.n/2) + 1)*np.pi/5.0)
        self.G['tot'] = []
        for i in range(self.numSamples):
            iG = []
            for j in range(self.n):
                num = 0
                for k in self.keys:
                    num += self.G[k][i][j]
                iG.append(num/len(self.keys))
            self.G['tot'].append(iG)

    def writeFiles(self):
        """
        Write 1000 samples.
        """
        for k,v in self.G.iteritems():
            file = open(k, 'w')
            for i in range(self.n):
                file.write(str(self.wn[i]) + '\t')
                for j in range(self.numSamples):
                    file.write(str(np.real(v[j][i])) + '\t' + str(np.imag(v[j][i])) + '\t')
                file.write('\n')
            file.close()



if __name__ == "__main__":
    data = extract('7_G(K,w,b,b)', 100)