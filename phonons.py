import numpy as np
import matplotlib.pyplot as plt
import os

import scienceplots
plt.style.use(['science','no-latex'])

from ase.atom import Atom
from chgnet.model import CHGNet, StructOptimizer, CHGNetCalculator
from chgnet.model.dynamics import TrajectoryObserver
from pymatgen.core import Structure
from pymatgen.io.ase import MSONAtoms, AseAtomsAdaptor

from dyn import *

im_root = "./resources/images/"
arr_root = "./resources/arrays/"
cif_root = "./resources/cifs"

dConv = 15.63


def main():
    for file in os.listdir(f"{cif_root}"):
        Dyn = np.load(f"{cif_root}/{file}")
        vdosPlot(Dyn, width = 100, title = file[:-4])
        plot.show()
        #plt.savefig(f"{im_root}/{file[:-4]}.png", dpi = 500)
    return


def vdosPlot(Dyn: np.ndarray = np.load("resources/arrays/betaCristobalite.npy"), width: int = 50, title = ""):
    #Init
    c,d = vdos(Dyn, exclude = 0)
    d = d[d != 0]

    x = np.linspace(0, np.max(d), width)
    y = np.histogram(d, bins = np.linspace(0, np.max(d), width + 1))[0]

    fig,ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)


    ax.set_xlabel(r"$\nu$ [THz]")
    ax.xaxis.set_tick_params(which = "minor", bottom = False)
    ax.set_ylabel(r"VDOS [A.U.]")
    ax.set_yticklabels([])
    ax.yaxis.set_tick_params(which = "minor", bottom = False)

    plt.show()
    plt.savefig(f"{im_root}/{title}.png")
    return ax


def vdos(Dyn: np.ndarray, exclude: str = False, save: str = False):
    D,C = np.linalg.eigh(Dyn)
    D = np.sqrt(D)

    c = C * dConv
    d = D * dConv

    if bool(exclude):
        d = np.nan_to_num(d, exclude)

    if bool(save):
        np.save(save, Dyn)


    return c,d



if __name__ == "__main__":
    main()
