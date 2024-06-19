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
#from pqeq import *
from q2 import *
from rruffIR import rruffIR

im_root = "./resources/images/"
arr_root = "./resources/arrays/"
cif_root = "./resources/cifs/"

rreff = "resources/rruff/processed/data/Danburite__R040013-1__Infrared__Infrared_Data_Processed__249.txt"
rreff2 = "resources/rruff/processed/data/Danburite__R050602-1__Infrared__Infrared_Data_Processed__640.txt"

pi = np.pi
ep0 = 8.8542e-12
c = 299_792_458.

muConv = 1.602e-19 * 1e-10

def main():
    spectrum, rs = rruffIR(f"{rreff2}")
    spectrum, _ = rruffIR(f"{rreff2}")
    #rs = rs[3 * len(spectrum) // 7:]
    #spectrum = spectrum[3 * len(spectrum) // 7:]
    print(spectrum)
    nrs = (rs - np.min(rs)) / (np.max(rs) - np.min(rs))
    spectrum *= 0.03

    atoms = buildNumber(f"{cif_root}/danburite.cif", 1000)
    #q,mu = pqeq(atoms)
    atoms.get_charges = lambda: atoms.arrays["oxi_states"]
    #atoms.get_charges = lambda: pqeq(atoms)
    #spectrum = np.linspace(0, 120, 4000)
    Dyn = np.load(f"{arr_root}/danburite.npy")
    #Dyn = dynamical(atoms)
    ds = dipoleSpectrum(atoms, Dyn, spectrum, y = 0.3)
    nds = (ds - np.min(ds)) / (np.max(ds) - np.min(ds))

    fig, (dax,rax) = plt.subplots(2)
    spectrum = list(spectrum * 1/0.03)
    #spectrum.reverse()
    dax.plot(spectrum, nds, label = "Prediction")
    rax.plot(spectrum, nrs, label = "Literature")
    dax.legend()
    rax.legend()
    dax.set_title("Danburite IR")
    #plt.show()
    plt.savefig("DanburiteIRP.png", dpi=500)
    return


def dipoleSpectrum(atoms: MSONAtoms, Dyn: np.ndarray, spectrum: float, h: float = 1e-5, y: float = 1e-3) -> np.ndarray:
    c,d = vdos(Dyn)
    density = d[ np.nan_to_num(d, 0) != 0 ]
    vibrations = c[ np.nan_to_num(d, 0) != 0 ]
    ddm = np.array([ dipolePartial(atoms, v, h) for v in vibrations ]) * 1.602e-19
    print(f"{ddm = }")
    spectrum = spectrum.reshape(( *spectrum.shape, 1 ))
    return np.sum( dipoleAbsorption(spectrum, density, ddm, y), axis = 1 )


def dipoleAbsorption(w: float, k: float, ddm: float, y: float = 1e-3) -> float:
    ddm2 = np.sum(ddm**2, axis = (1,2))
    ddm2 = ddm2.reshape( (1,) + ddm2.shape )
    return (pi/(3*ep0*c)) * ddm2 * (k / ((k - w)**2 + y**2))


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
