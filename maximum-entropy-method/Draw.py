import sys
import numpy as np
import Maxent
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground= "w", linewidth=3)])
    return at

ifile = open(sys.argv[1])
dataset = pickle.load(ifile)

w = dataset["w"]
allSpecFs = dataset["allSpecFs"]
aveSpecFs = dataset["aveSpecFs"]
alphas = dataset["alphas"]
allProbs = dataset["allProbs"]
#rFre = dataset["rFre"]
#tSpecF = dataset["tSpecF"]

#plt.plot(rFre, tSpecF, 'r--' ,alpha = 0.5,label = "TestFunc")
plt.plot(w, aveSpecFs, 'b->', alpha = 0.8, label = "Maxent")
plt.xlabel(r"$\omega$", fontsize = 16)
plt.ylabel(r"$A(\omega)$", fontsize = 16)
#plt.xlim([-6, 8])
plt.legend(loc = 2)

a = plt.axes([0.55, 0.55, 0.3, 0.3])
plt.semilogx(alphas, allProbs, 'b-')
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$P(\alpha|G)$")
plt.setp(a, yticks = [np.amin(allProbs)*1.2, np.amax(allProbs)*0.8])

#plt.tight_layout()
plt.show()


#----------------------------------------------

fig = plt.figure()
ax = fig.gca(projection='3d')

W, Alphas  = np.meshgrid(w, alphas)


surf = ax.plot_surface(W, Alphas, allSpecFs, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

#ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\alpha$")
ax.set_zlim(-0.1, np.amax(allSpecFs))
plt.setp(ax, zticks=[])

fig.colorbar(surf, shrink=0.5, aspect=5, orientation="vertical")

plt.show()