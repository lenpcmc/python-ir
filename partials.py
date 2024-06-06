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

import scipy.optimize as so

from dyn import *

im_root = "./resources/images/"
arr_root = "./resources/arrays/"
cif_root = "./resources/cifs"

dConv = 15.63


def main():
    structure, atoms = buildArray("resources/cifs/betaCristobalite.cif", [2,2,2])
    x = fitMagCharge(structure, atoms)
    mags = atoms.get_magnetic_moments()
    masses = atoms.get_masses()
    for i,s in enumerate(np.unique(atoms.symbols)):
        locs = atoms.symbols == s
        print(2. * mags[locs] * masses[locs] / x.x[i])

    return x


def fitMagCharge(structure: Structure, atoms: MSONAtoms):
    mags = atoms.get_magnetic_moments()
    masses = atoms.get_masses()
    species = np.unique(atoms.symbols)
    oxidations = atoms.arrays["oxi_states"]

    scaleFactor: np.ndarray = np.zeros(len(species))
    locs: dict = { s: atoms.symbols == s for s in species }
    for i,s in enumerate(locs):
        #if (i == len(locs)): break
        scaleFactor[i] = 2 * np.average(mags[locs[s]] * masses[locs[s]] / oxidations[locs[s]])
    #scaleFactor[len(locs)] = np.sum()

    print(f"{scaleFactor = }")
    print(f"{getEnergy(scaleFactor, atoms, locs) = }")
    x = so.minimize(getEnergy, args=(atoms, locs), x0=scaleFactor)
    print(x)
    return x


def getEnergy(scaleFactor: np.ndarray, atoms: MSONAtoms, locs: dict) -> float:
    # Init
    mags = atoms.get_magnetic_moments()
    masses = atoms.get_masses()
    oxidations = atoms.arrays["oxi_states"]
    dij: np.ndarray = atoms.get_all_distances(mic=True)
    iR: float = 1./ dij
    np.fill_diagonal(dij, 0)
    #print(dij)

    q = np.zeros(mags.size)
    #print(q)
    spec = list(locs)[-1]
    for i,s in enumerate(list(locs)):
        if( s == spec ): break
        q += 2. * locs[s] * mags * masses / scaleFactor[i]
    q += locs[spec] * ( - np.sum(q) / np.sum(locs[spec]) )
    q = np.min( (np.abs(q), np.abs(oxidations)), axis = 0)
    q[q * oxidations < 0] *= -1
    print(f"{q = }")
    print(f"{oxidations = }")
    print(f"{np.abs(q) > np.abs(oxidations) = }")
    
#q = 2. * np.array([ locs[s] * mags * masses / scaleFactor[i] for i,s in enumerate(locs) ]).flatten()
    #print(q)

    # Calc
    #print(f"{q = }")
    #print(f"{q.T = }")
    q = q.reshape(1,12)
    energies = iR * (q.T @ q) / 2
    #print(f"{q @ q.T = }")
    np.fill_diagonal(energies, 0)
    print(f"{energies = }")
    print(f"{np.sum(energies) = }")
    return np.sum(energies)



if __name__ == "__main__":
    main()
