import numpy as np
import matplotlib.pyplot as plt

from ase.atom import Atom
from chgnet.model import CHGNet, StructOptimizer, CHGNetCalculator
from pymatgen.core import Structure
from pymatgen.io.ase import MSONAtoms, AseAtomsAdaptor

from build import *

def main():
    atoms = buildArray("betaCristobalite.cif", [2,2,2])
    Dyn = dynamical(atoms)
    plt.imshow(Dyn)
    plt.show()
    return Dyn


def dynamical(atoms :MSONAtoms, h: float = 1e-5, verbose: int = None) -> np.ndarray:
    dynamicVector = np.array([ 3 * [a.mass] for a in atoms ]).reshape(1, 3*len(atoms))
    dynamicVector = 1./np.sqrt(dynamicVector)
    dynamicScale = dynamicVector.T @ dynamicVector
    return dynamicScale * hessian(atoms, h=h, verbose=verbose)


def hessian(atoms: MSONAtoms, h: float = 1e-5, verbose: int = None) -> np.ndarray:
    # Init
    H: list[float] = list()
    verbose: int = len(atoms) + 1 if verbose == None else verbose

    # Iterate 
    for i,_ in enumerate(atoms):
        if i % verbose == 0:
            print(i)
        for pos in range(3):
            hRow = hessRow(atoms, i, pos, h)
            H.append(hRow)

    H: np.ndarray = np.array(H)
    return -1 * (H + H.T) / 2

def hessian(atoms: MSONAtoms, h: float = 1e-5, verbose: int = None) -> np.ndarray:
    H: np.ndarray = np.array([ [ hessRow(atoms, i, pos, h) for pos in range(3) ] for i,_ in enumerate(atoms) ])
    sidel = 3 * len(atoms)
    H = H.reshape(sidel, sidel)
    return -1 * (H + H.T) / 2


def hessRow(atoms: MSONAtoms, i: int, pos: int, h: float = 1e-5) -> np.ndarray:
    # Init
    atoms.calc = CHGNetCalculator() if atoms.calc == None else atoms.calc
    a = atoms[i]

    # Positive Direction
    a.position[pos] += h
    fp = atoms.get_forces()

    # Negative Direction
    a.position[pos] -= 2*h
    fm = atoms.get_forces()

    # Derivative
    a.position[pos] += h
    D = (fp - fm) / (2*h)
    return D.flatten()



if __name__ == "__main__":
    main()
