import numpy as np
import matplotlib.pyplot as plt

from chgnet.model import CHGNet, StructOptimizer
from pymatgen.core import Structure

def main():
    chgnet = CHGNet.load()
    structure = Structure.from_file("../chgnet/examples/mp-18767-LiMnO2.cif")
    x = relaxStruct(structure)
    return x


def relaxStruct(structure):
    relaxer = StructOptimizer()
    result = relaxer.relax(structure)
    #print(f"Relaxed structure {result['final_structure']}")
    #print(f"Relaxed total energy {result['trajectory'].energies[-1]} eV")
    return result


if __name__ == "__main__":
    main()
