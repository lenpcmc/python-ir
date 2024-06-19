import numpy as np
import matplotlib.pyplot as plt
import math
import os
import scipy

from chgnet.model import CHGNetCalculator
from ase.io.sdf import read_sdf

from build import *
from pqeq import *

P = dict(); Xo = dict(); Jo = dict(); Z = dict(); Rc = dict(); Rs = dict(); Ks = dict()
global par; par = { 'P': P, 'Xo': Xo, 'Jo': Jo, 'Z': Z, 'Rc': Rc, 'Rs': Rs, 'Ks': Ks }

def main():
    cyclo = read_sdf("CYCLOHEXANE-3D-structure-CT1001745819.sdf")
    cyclo.get_charges = lambda: np.array([-2]*6 + [1]*12)
    cyclo.s = cyclo.copy()
    shellPositionsC(cyclo)
    return


def shellPositionsC(atoms, h: float = 1e-6):
    a,s = atoms, atoms.s
    rc,rs = atoms.positions, atoms.s.positions
    Z = atoms.numbers; q = atoms.get_charges()
    qc, qs = q + Z, -Z

    K = np.array([ Ks[atoms[i].symbol] for i,_ in enumerate(atoms) ]).reshape(len(atoms),1)

    interForces = -1. * np.array([ shellForces(atoms, i, h) for i,_ in enumerate(atoms) ])
    intraForces = -1. * (rc - rs) * K

    rs = rs + (interForces + intraForces) / K
    return rs


def shellForces(atoms, i, h: float = 1e-6):
    Z = atoms.numbers; q = atoms.get_charges()
    qc, qs = q + Z, -Z
    s = atoms.s[i]

    ficjc = [ [], [], [] ]
    ficjs = [ [], [], [] ]
    fisjc = [ [], [], [] ]
    fisjs = [ [], [], [] ]

    for d in range(3):
        s.position[d] += h
        ficjc[d].append(np.sum([ C(atoms[i], atoms[j]) * qc[i]*qc[j] for j,_ in enumerate(atoms) ]))
        ficjs[d].append(np.sum([ C(atoms[i], atoms.s[j]) * qc[i]*Z[j] for j,_ in enumerate(atoms) ]))
        fisjc[d].append(np.sum([ C(atoms.s[i], atoms[j]) * Z[i]*qc[j] for j,_ in enumerate(atoms) ]))
        fisjs[d].append(np.sum([ C(atoms.s[i], atoms.s[j]) * Z[i]*Z[j] for j,_ in enumerate(atoms) ]))

        s.position[d] -= 2.*h
        ficjc[d].append(np.sum([ C(atoms[i], atoms[j]) * qc[i]*qc[j] for j,_ in enumerate(atoms) ]))
        ficjs[d].append(np.sum([ C(atoms[i], atoms.s[j]) * qc[i]*Z[j] for j,_ in enumerate(atoms) ]))
        fisjc[d].append(np.sum([ C(atoms.s[i], atoms[j]) * Z[i]*qc[j] for j,_ in enumerate(atoms) ]))
        fisjs[d].append(np.sum([ C(atoms.s[i], atoms.s[j]) * Z[i]*Z[j] for j,_ in enumerate(atoms) ]))

        s.position[d] += h
        ficjc[d] = (ficjc[d][0] - ficjc[d][1]) / (2.*h)
        ficjs[d] = (ficjs[d][0] - ficjs[d][1]) / (2.*h)
        fisjc[d] = (fisjc[d][0] - fisjc[d][1]) / (2.*h)
        fisjs[d] = (fisjs[d][0] - fisjs[d][1]) / (2.*h)

    return np.array(ficjc) - np.array(ficjs) - np.array(fisjc) + np.array(fisjs)
    

if __name__ == "__main__":
    loadParams()
    main()
