import numpy as np
import matplotlib.pyplot as plt

from ase.atom import Atom
from chgnet.model import CHGNet, StructOptimizer, CHGNetCalculator
from chgnet.model.dynamics import TrajectoryObserver
from pymatgen.core import Structure
from pymatgen.io.ase import MSONAtoms, AseAtomsAdaptor

from dyn import *

def main():
    atoms = buildNumber("betaCristobalite.cif", 1000)
    Dyn = dynamical(atoms, verbose=100)
    #plt.imshow(Dyn)
    #plt.show()
    
    D,C = np.linalg.eigh(Dyn)
    print(np.sqrt(D))
    print(C)
    #print(f"{C=} \n{D=}")
    #print(np.array([D]))
    D = np.identity(len(D)) @ np.sqrt(np.array([D]))
    #print(f"{C=} \n{D=}")
    #print(C @ D @ C.T)

    return Dyn




if __name__ == "__main__":
    main()
