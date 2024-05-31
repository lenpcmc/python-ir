import numpy as np
import matplotlib.pyplot as plt

from ase.atom import Atom
from chgnet.model import CHGNet, StructOptimizer, CHGNetCalculator
from chgnet.model.dynamics import TrajectoryObserver
from pymatgen.core import Structure
from pymatgen.io.ase import MSONAtoms, AseAtomsAdaptor

from relax import relaxStruct

def main():
    chgnet = CHGNet.load()
    structure = Structure.from_file("betaCristobalite.cif")
    #structure = Structure.from_file("../chgnet/examples/mp-18767-LiMnO2.cif")
    atoms = relaxStruct(structure)["trajectory"].atoms
    Dyn = dynamical(atoms)
    plt.imshow(Dyn)
    plt.show()
    return Dyn


def dynamical(atoms :MSONAtoms, h=1e-5) -> np.ndarray:
    dynamicVector = np.array([ 3 * [a.mass] for a in atoms ]).reshape(1, 3*len(atoms))
    dynamicScale = dynamicVector.T @ dynamicVector
    return dynamicScale * hessian(atoms)


def hessian(atoms :MSONAtoms, h=1e-5) -> np.ndarray:
    # Init
    H = list()

    # Iterate 
    for i,_ in enumerate(atoms):
        for pos in range(3):
            hRow = hessRow(atoms, i, pos, h)
            H.append(hRow)

    return - np.array(H)


def hessRow(atoms :MSONAtoms, i :int, pos :int, h) -> np.ndarray:
    # Init
    myatoms = atoms.copy()
    myatoms.calc = atoms.calc
    a = myatoms[i]

    # Positive Direction
    a.position[pos] += h
    fp = myatoms.get_forces()

    # Negative Direction
    a.position[pos] -= 2*h
    fm = myatoms.get_forces()

    # Derivative
    D = (fp - fm) / (2*h)
    return D.flatten()



if __name__ == "__main__":
    main()
