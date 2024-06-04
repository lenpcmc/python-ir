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

def main():
    for file in os.listdir(f"{cif_root}"):
        fig, ax = plt.subplots()
        fig, ax = vdosPlot(fname = f"{file}", numAtoms = 1000, width = 100)
        plt.savefig(f"{im_root}/{file[:-4]}.png", dpi = 500)
    return


def vdosPlot(fname: str = "betaCristobalite.cif", numAtoms = 20, width = 20):
    #Init
    c,d = vdos(fname = fname, numAtoms = numAtoms)

    x = np.linspace(0, np.max(d), width)
    y = np.histogram(d, bins = np.linspace(0, np.max(d), width + 1))[0]

    fig,ax = plt.subplots()
    ax.plot(x, y, 'o')

    ax.set_xlabel(r"$\nu$ [THz]")

    ax.set_ylabel(r"VDOS [A.U.]")
    ax.set_yticklabels([])
    ax.yaxis.set_tick_params(which = "minor", bottom = False)

    return fig, ax


def vdos(fname: str, numAtoms: int = 20, exclude: bool = True, save: bool = True):
    atoms = buildNumber(f"{cif_root}/{fname}", numAtoms)
    Dyn = dynamical(atoms)
    D,C = np.linalg.eigh(Dyn)
    D = np.sqrt(D)

    dConv = dynamicalConversion(1)
    c = C * dConv
    d = D * dConv

    if (exclude):
        d = np.nan_to_num(d)

    if (save):
        np.save(f"{arr_root}/{fname[:-4]}.npy", Dyn)

    return c,d


def dynamicalConversion(vals: float) -> float:
    # 98.225 (eV / A^2 amu) per (J / kg m) / 2pi
    # return vals / 15.63
    return ( vals * (9.822517e13 / 1.e12) )/(6.28)


if __name__ == "__main__":
    main()
