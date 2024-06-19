import numpy as np
import matplotlib.pyplot as plt
import math
import os
import scipy
from multiprocessing import Pool

from ase.io.sdf import read_sdf
from ase.geometry import get_distances
from chgnet.model import CHGNetCalculator

from time import time
from tqdm import tqdm

from build import *
enum = lambda arr: tqdm(enumerate(arr))

P = dict(); Xo = dict(); Jo = dict(); Z = dict(); Rc = dict(); Rs = dict(); Ks = dict()
global par; par = { 'P': P, 'Xo': Xo, 'Jo': Jo, 'Z': Z, 'Rc': Rc, 'Rs': Rs, 'Ks': Ks }

def main():
    loadParams()
    #atoms = buildNumber(f"resources/cifs/betaCristobalite.cif", 100)
    structure, atoms = buildArray(f"resources/cifs/betaCristobalite.cif", [3,3,3])
    atoms = relaxStruct(structure)["trajectory"].atoms
    q,qc = pqeq(atoms)
    print(f"{q = }")
    print(f"{qc = }")
    plt.plot(range(len(q)), q)
    plt.plot(range(len(q)), qc)
    plt.show()
    return


def loadParams():
    global par

    with open("params.csv") as infile:
        indata: list[str] = [ line.strip().split(',') for line in infile if '#' not in line ]
    for param in indata:
        #el: str = param[0]
        el = param[0]
        for p,v in zip( par.keys(), np.array(param[1:], dtype = np.float64) ):
            par[p][el] = v
    
    par["Ks"] = { k: v * 0.04336 for k,v in Ks.items() }
    return


def pqeq(atoms):
    loadParams()
    try:
        atoms.get_charges()
    except RuntimeError:
        atoms.get_charges = lambda: atoms.arrays["oxi_states"]
        atoms.s = atoms.copy()

    #atoms.s.positions = shellPositionsC(atoms, h=1e-6)
    atoms.get_charges = lambda: atoms.arrays["oxi_states"]
    atoms.s.positions = shellPositions(atoms)
    H = Hij(atoms)
    Hinv = np.linalg.inv(H)
    A = Ai(atoms)
    sH = scipy.sparse.csc_matrix(H)
    sH_iLU = scipy.sparse.linalg.spilu(sH)
    M = scipy.sparse.linalg.LinearOperator( H.shape, sH_iLU.solve )

    #qt = np.array([ np.sum([ Hinv[i,j] * -A[j] for j,_ in enum(atoms) ]) for i,_ in enum(atoms) ])
    qtc = scipy.sparse.linalg.cg(H, -A, M=M)[0]

    #qh = np.array([ np.sum([ Hinv[i,j] * -1 for j,_ in enum(atoms) ]) for i,_ in enum(atoms) ])
    qhc = scipy.sparse.linalg.cg(H, -1.*np.ones(len(atoms)), M=M)[0]

    #mu = np.sum(qt) / np.sum(qh)
    muc = np.sum(qtc) / np.sum(qhc)

    #q = qt - mu * qh
    qc = qtc - muc * qhc
    #q = Hinv @ ( -A + mu * np.ones(len(A)) )
    return qc


def pqeqE(atoms, q):
    global par
    a,s = atoms, atoms.s
    Z = np.ones(len(atoms))
    #Z = atoms.numbers
    ric,ris = atoms.positions, atoms.s.positions

    creationEnergy = np.sum([ Xo[atoms[i].symbol] * q[i] + 0.5 * Jo[atoms[i].symbol] * q[i]**2 + 0.5 * Ks[atoms[i].symbol] * (ric - ris)**2 for i,_ in enum(atoms) ])

    #pairwsieEnergy = np.sum( [ np.sum([ C(atoms, i, j) * q[i]*q[j] ] for j in range(i)]) for i in range(len(atoms))])
    pairwiseEnergy = 14.4 * np.sum([np.sum([ C(atoms[i], atoms[j]) * q[i]*q[j] - C(atoms[i], atoms.s[j])*q[i]*Z[j] - C(atoms.s[i], atoms[j]) * Z[i]*q[j] + C(atoms.s[i], atoms.s[j]) * Z[i]*Z[j] for j in range(i) ]) for i,_ in enum(atoms)])

    return creationEnergy + pairwiseEnergy


def alpha(a):
    global par
    Rik = Rc[a.symbol]
    lambda_pqeq = 0.462770
    return 0.5 * lambda_pqeq / (Rik**2)


def C(ai, aj, cell = None):
    aik = alpha(ai); ajl = alpha(aj)
    alph = aik * ajl / (aik + ajl)

    rij = get_distances(ai.position, aj.position, cell = ai.atoms.cell, pbc = True)[1][0,0]

    if math.isclose(rij, 0):
        egc = 2 * np.sqrt(alph) / np.pi
    else:
        egc = math.erf(np.sqrt( alph * rij**2 )) / rij
    return egc


def dC(ai,aj):
    aik = alpha(ai); ajl = alpha(aj)
    alph = aik * ajl / (aik + ajl)

    ri = ai.position; rj = aj.position
    r,r2 = get_distances(ai.position, aj.position)
    r = r[0,0]
    r2 = r2[0,0]**2

    s = math.erf(np.sqrt( alph * r2 ))
    sp = 2. * np.sqrt(alph) * np.exp(-alph * r**2) / np.sqrt(np.pi)

    if math.isclose(r2, 0.):
        fgc = 0. * r
    else:
        fgc = (sp * r - s * 1.) / r2

    return fgc


def Hij(atoms):
    global par
    H = np.array([ [ 14.4 * C(atoms[i],atoms[j]) for i,_ in enum(atoms) ] for j,_ in enum(atoms) ])
    for i,a in enum(atoms):
        H[i,i] = Jo[a.symbol]
    return H


def Ai(atoms):
    global par
    Z = np.ones(len(atoms))
    A = np.array([ Xo[atoms[i].symbol] + np.sum([ Z[j] * 14.4 * ( C(atoms[i],atoms[j]) - C(atoms[i],atoms.s[j]) ) for j in range(i) ]) for i,_ in enum(atoms) ])
    return A


def Tap(r, rcut: float = 10):
    tparams = [ 1, 0, 0, -35, 84, -70, 20 ]
    return np.sum([ (r/rcut)**a for a in tparams ])


def shellPositions(atoms):
    Z = np.ones(len(atoms))
    #Z = atoms.numbers
    q = atoms.get_charges()
    qc, qs = q + Z, -Z

    #Ficjc = -1. * np.array([np.sum([ dC(a,i,j) * qc[i]*qc[j] for j,_ in enum(atoms) ], axis = 0) for i,_ in enum(atoms) ])
    #Ficjs = -1. * np.array([np.sum([ dC(a,i,j) * qc[i]*Z[j] for j,_ in enum(atoms) ], axis = 0) for i,_ in enum(atoms) ])
    Fisjc = -1. * np.array([np.sum([ dC(atoms.s[i],atoms[j]) * Z[i]*qc[j] for j,_ in enum(atoms) ], axis = 0) for i,_ in enum(atoms) ])
    Fisjs = -1. * np.array([np.sum([ dC(atoms.s[i],atoms.s[j]) * Z[i]*Z[j] for j,_ in enum(atoms) ], axis = 0) for i,_ in enum(atoms) ])
    K = np.array([ Ks[atoms[i].symbol] for i,_ in enum(atoms) ]).reshape(len(atoms),1)

    interForces = - Fisjc + Fisjs

    rs = atoms.s.positions + interForces / K
    return rs


def shellPositionsC(atoms, h: float = 1e-6):
    a,s = atoms, atoms.s
    rcs = get_distances(atoms.positions, atoms.s.positions, cell = atoms.cell, pbc = True)[1][0,0]
    q = atoms.get_charges()
    Z = np.ones(len(atoms))
    #Z = atoms.numbers
    qc, qs = q + Z, -Z

    K = np.array([ Ks[atoms[i].symbol] for i,_ in enum(atoms) ]).reshape(len(atoms),1)

    interForces = -1. * np.array([ shellForces(atoms, i, h) for i,_ in enum(atoms) ])
    intraForces = -1. * rcs * K

    rs = atoms.s.positions + (interForces + intraForces) / K
    return rs


def shellForces(atoms, i, h: float = 1e-6):
    q = atoms.get_charges()
    Z = np.ones(len(atoms))
    #Z = atoms.numbers
    qc, qs = q + Z, -Z

    ficjc = [ [], [], [] ]
    ficjs = [ [], [], [] ]
    fisjc = [ [], [], [] ]
    fisjs = [ [], [], [] ]

    for d in range(3):
        atoms.s.positions[d] += h
        ficjc[d].append(np.sum([ C(atoms[i], atoms[j]) * qc[i]*qc[j] for j,_ in enum(atoms) ]))
        ficjs[d].append(np.sum([ C(atoms[i], atoms.s[j]) * qc[i]*Z[j] for j,_ in enum(atoms) ]))
        fisjc[d].append(np.sum([ C(atoms.s[i], atoms[j]) * Z[i]*qc[j] for j,_ in enum(atoms) ]))
        fisjs[d].append(np.sum([ C(atoms.s[i], atoms.s[j]) * Z[i]*Z[j] for j,_ in enum(atoms) ]))

        atoms.s.positions[d] -= 2.*h
        ficjc[d].append(np.sum([ C(atoms[i], atoms[j]) * qc[i]*qc[j] for j,_ in enum(atoms) ]))
        ficjs[d].append(np.sum([ C(atoms[i], atoms.s[j]) * qc[i]*Z[j] for j,_ in enum(atoms) ]))
        fisjc[d].append(np.sum([ C(atoms.s[i], atoms[j]) * Z[i]*qc[j] for j,_ in enum(atoms) ]))
        fisjs[d].append(np.sum([ C(atoms.s[i], atoms.s[j]) * Z[i]*Z[j] for j,_ in enum(atoms) ]))

        atoms.s.positions[d] += h
        ficjc[d] = (ficjc[d][0] - ficjc[d][1]) / (2.*h)
        ficjs[d] = (ficjs[d][0] - ficjs[d][1]) / (2.*h)
        fisjc[d] = (fisjc[d][0] - fisjc[d][1]) / (2.*h)
        fisjs[d] = (fisjs[d][0] - fisjs[d][1]) / (2.*h)

    return 14.4 * (np.array(ficjc) - np.array(ficjs) - np.array(fisjc) + np.array(fisjs))
    

if __name__ == "__main__":
    loadParams()
    main()
