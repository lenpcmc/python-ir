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

from vdos import *
from dyn import *
from pqeq import *
from rruffIR import rruffIR

im_root = "./resources/images/"
arr_root = "./resources/arrays/"
cif_root = "./resources/cifs/"

rreff = "resources/rruff/processed/data/Wollastonite__R040008-1__Infrared__Infrared_Data_Processed__1001.txt"

pi = np.pi
ep0 = 8.8542e-12
c = 299_792_458.

muConv = 1.602e-19 * 1e-10

def main():
    spectrum, rs = rruffIR(f"{rreff}")
    #rs = rs[3 * len(spectrum) // 7:]
    #spectrum = spectrum[3 * len(spectrum) // 7:]
    print(spectrum)
    nrs = (rs - np.min(rs)) / (np.max(rs) - np.min(rs))
    spectrum *= 0.03

    atoms = buildNumber(f"{cif_root}/wollastonite.cif", 1000)
    #atoms.get_charges = lambda: atoms.arrays["oxi_states"]
    #atoms.get_charges = lambda: q; q = pqeq(atoms)
    atoms.get_charges = lambda: pqeq(atoms)
    print(f"{atoms.get_charges()}")
    #spectrum = np.linspace(0, 120, 4000)
    Dyn = np.load(f"{arr_root}/wollastonite.npy")
    ds = dipoleSpectrum(atoms, Dyn, spectrum, y = 0.05)
    nds = (ds - np.min(ds)) / (np.max(ds) - np.min(ds))

    fig, (dax,rax) = plt.subplots(2)
    spectrum = list(spectrum * 1/0.03)
    spectrum.reverse()
    dax.plot(spectrum, nds, label = "Prediction")
    rax.plot(spectrum, nrs, label = "Literature")
    dax.legend()
    rax.legend()
    dax.set_title("Wollastonite IR")
    #plt.show()
    plt.savefig("wollastoniteIRP.png", dpi=500)
    return


def dipoleSpectrum(atoms: MSONAtoms, Dyn: np.ndarray, spectrum: float, h: float = 1e-5, y: float = 1e-3) -> np.ndarray:
    vibrations, d = vdos(Dyn)
    density = d[ np.nan_to_num(d, 0) != 0 ]
    ddm = list()
    for i,v in enumerate(vibrations):
        print(i)
        ddm.append(dipolePartial(atoms, v, h))
    ddm = np.array(ddm)
    #ddm = np.array([ dipolePartial(atoms, v, h) for v in vibrations ])
    ddm = ddm * 1.602e-19
    spectrum = spectrum.reshape(( *spectrum.shape, 1 ))
    return np.sum( dipoleAbsorption(spectrum, density, ddm, y), axis = 1 )


def dipoleAbsorption(w: float, k: float, ddm: float, y: float = 1e-3) -> float:
    return (pi/(3*ep0*c)) * np.sum(ddm**2) * (k / ((k - w)**2 + y**2))


def dipole(atoms: MSONAtoms) -> np.ndarray:
    chargeVector = atoms.get_charges()
    chargeTensor = np.array([ chargeVector for _ in range(3) ]).T
    return np.sum([ chargeTensor * atoms.positions ], axis = 1)


def dipolePartial(atoms: MSONAtoms, v: np.ndarray, h: float = 1e-5) -> np.ndarray:
    pshape = atoms.positions.shape
    ptensor = atoms.positions.flatten()

    atoms.positions = (ptensor + v * h).reshape(pshape)
    dp = dipole(atoms)

    atoms.positions = (ptensor - v * h).reshape(pshape)
    dn = dipole(atoms)

    atoms.positions = ptensor.reshape(pshape)
    D = (dp - dn) / (2*h)
    return D


if __name__ == "__main__":
    main()
    pass
