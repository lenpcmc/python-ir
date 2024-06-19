import numpy as np
import matplotlib.pyplot as plt

from chgnet.model import CHGNet, StructOptimizer
from ase.atoms import Atoms
from ase.visualize import view
from pymatgen.core import Structure

def main():
    atoms = buildArray("betaCristobalite.cif", [20,20,20])
    view(atoms)
    return


def buildNumber(filename :str, numAtoms :int = 1000) -> Atoms:
    chgnet = CHGNet.load()
    structure = Structure.from_file(filename)
    relaxedStruct = relaxStruct(structure)["trajectory"].atoms

    repeatNumber = int(np.cbrt(numAtoms/len(relaxedStruct))) + 1
    repeatArray = 3 * [repeatNumber]
    repeatStruct = relaxedStruct * repeatArray
    return repeatStruct


def buildArray(filename :str, repeat :np.typing.ArrayLike = [1,1,1]) -> Atoms:
    chgnet = CHGNet.load()
    structure = Structure.from_file(filename)
    relaxed = relaxStruct(structure)
    relaxedAtoms = relaxed["trajectory"].atoms
    relaxedStruct = relaxed["final_structure"]
    repeatAtoms = relaxedAtoms * repeat
    repeatStruct = relaxedStruct.make_supercell(repeat)
    return repeatStruct, repeatAtoms


def relaxStruct(structure :Structure) -> dict:
    relaxer = StructOptimizer()
    result = relaxer.relax(structure)
    #print(f"Relaxed structure {result['final_structure']}")
    #print(f"Relaxed total energy {result['trajectory'].energies[-1]} eV")
    return result


if __name__ == "__main__":
    main()
