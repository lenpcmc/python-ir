import numpy as np
import matplotlib.pyplot as plt
import math
import os
import scipy

from chgnet.model import CHGNetCalculator
from ase.io.sdf import read_sdf

from build import *

P = dict(); Xo = dict(); Jo = dict(); Z = dict(); Rc = dict(); Rs = dict(); Ks = dict()
global par; par = { 'P': P, 'Xo': Xo, 'Jo': Jo, 'Z': Z, 'Rc': Rc, 'Rs': Rs, 'Ks': Ks }

def main():
    loadParams()
    cyclo = read_sdf("CYCLOHEXANE-3D-structure-CT1001745819.sdf")
    #cyclo.arrays["oxi_states"] = np.array([-2]*6 + [1]*12)
    atoms = buildNumber(f"resources/cifs/alphaCristobalite.cif", 1)
    atoms.get_charges = lambda: np.zeros(len(atoms))
    atoms.s = atoms.copy()
    print('Starting Energy')
    print(pqeqE(np.zeros(len(atoms)) ,atoms ) )

    import scipy.optimize as so
    x = so.minimize(pqeqE,x0=np.zeros(len(atoms)), args=(atoms,))
    print(x)
    q = x.x
    #q = np.array([-2]*6 + [1]*12)
    #q,mu = pqeq(atoms)
    print('Ending Energy')
    print(pqeqE(x.x,atoms))
    print(f"{q = }")
    #print(f"{mu = }")
    #q, mu = pqeq(cyclo)
    #plt.plot( range(len(cyclo)), q, marker = 'o', label = "Exact" )
    #plt.legend()
    #plt.show()

    #dq = np.array(q + 1e-3).reshape( (len(q),1) )
    #ep = np.array([ pqeqE(cyclo, p, mu) for p in q + dq ])
    #en = np.array([ pqeqE(cyclo, p, mu) for p in q - dq ])
    #print(ep-en)

    #pqeqe = lambda q: pqeqE(cyclo, q)
    #o = scipy.optimize.minimize(pqeqe, q, tol=10)
    #print(o)
    #print(o.x)

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
    return


def pqeq(atoms):
    loadParams()
    try:
        atoms.get_charges()
    except RuntimeError:
        atoms.get_charges = lambda: atoms.arrays["oxi_states"]
        atoms.s = atoms.copy()

    #atoms.s.positions = shellPositionsC(atoms, h=1e-10)
    atoms.s.positions = shellPositions(atoms)
    H = Hij(atoms)
    Hinv = np.linalg.inv(H)
    A = Ai(atoms)
    sH = scipy.sparse.csc_matrix(H)
    sH_iLU = scipy.sparse.linalg.spilu(sH)
    M = scipy.sparse.linalg.LinearOperator( H.shape, sH_iLU.solve )

    qt = np.array([ np.sum([ Hinv[i,j] * -A[j] for j,_ in enumerate(atoms) ]) for i,_ in enumerate(atoms) ])
    qtc = scipy.sparse.linalg.cg(H, -A, M=M)[0]
    print(f"{qt = }")
    print(f"{qtc = }")

    qh = np.array([ np.sum([ Hinv[i,j] * -1 for j,_ in enumerate(atoms) ]) for i,_ in enumerate(atoms) ])
    qhc = scipy.sparse.linalg.cg(H, -1.*np.ones(len(atoms)), M=M)[0]
    print(f"{qh = }")
    print(f"{qhc = }")

    mu = np.sum(qt) / np.sum(qh)
    muc = np.sum(qtc) / np.sum(qhc)
    print(f"{mu = }")
    print(f"{muc = }")

    qc = qtc - muc * qhc
    q = Hinv @ ( -A + mu * np.ones(len(A)) )
    print(f"{q = }")
    print(f"{qc = }")
    print(f"{np.sum(q) = }")
    print(f"{np.sum(qc) = }")
    
    return q, mu


def pqeqE(q ,atoms):
    global par
    a,s = atoms, atoms.s
    #q = atoms.get_charges() if q == None else q
    #q *= 14.4
    Z = np.zeros(len(atoms))
    #Z = np.ones(len(atoms))
    # ric,ris = atoms.positions, atoms.s.positions
    ric = atoms.positions
    ris = np.copy(ric)

    dij = atoms.get_all_distances()

    creationEnergy = np.sum([ Xo[atoms[i].symbol] * q[i] + 0.5 * Jo[atoms[i].symbol] * q[i]**2 + 0.5 * Ks[atoms[i].symbol] * (ric - ris)**2 for i,a in enumerate(atoms) ])
    print(creationEnergy)

    #pairwsieEnergy = np.sum( [ np.sum([ C(atoms, i, j) * q[i]*q[j] ] for j in range(i)]) for i in range(len(atoms))])
    #pairwiseEnergy = np.sum([ np.sum([ C(atoms, i, j) * q[i]*q[j] - C(atoms, i,j)*q[i]*Z[j] - C(atoms,i,j) * Z[i]*q[j] + C(atoms,i,j) * Z[i]*Z[j] for j in range(i) ]) for i in range(len(atoms))])

    C = 1./atoms.get_all_distances()
    pairwiseEnergy = 0.
    for i in range(len(atoms)):
       for j in range(i):
          pairwiseEnergy += C[i,j] * q[i] * q[j]

    #print(q)
    pairwiseEnergy *= 14.4
    #print( creationEnergy + pairwiseEnergy )
    print('pw: ' + str(pairwiseEnergy))
    #exit()
    return creationEnergy + pairwiseEnergy
    return creationEnergy + pairwiseEnergy - mu * np.sum(q)


def alpha(a):
    global par
    Rik = Rc[a.symbol]
    lambda_pqeq = 0.462770
    return 0.5 * lambda_pqeq / (Rik**2)


def C(atoms, i, j):
    aik = alpha(atoms[i]); ajl = alpha(atoms[j])
    alph = aik * ajl / (aik + ajl)

    dij = atoms.get_all_distances(mic=True)[i,j]
    #ri = ai.position; rj = aj.position
    #r = ri - rj
    r2 = dij**2.

    if math.isclose(r2, 0):
        egc = 2 * np.sqrt(alph) / np.pi
    else:
        egc = math.erf(np.sqrt( alph * r2 )) / np.sqrt(r2)
    return egc


def dC(a,i,j):
    ai = a[i]
    aj = a[j]
    aik = alpha(ai); ajl = alpha(aj)
    alph = aik * ajl / (aik + ajl)

    ri = ai.position; rj = aj.position
    r = ri - rj
    r2 = np.sum(r**2)

    s = math.erf(np.sqrt( alph * r2 ))
    sp = 2. * np.sqrt(alph) * np.exp(-alph * r**2) / np.sqrt(np.pi)

    if math.isclose(r2, 0.):
        fgc = 0. * r
    else:
        fgc = (sp * r - s * 1.) / r2

    return fgc


def Hij(atoms):
    global par
    H = np.array([ [ C(atoms,i,j) for i,_ in enumerate(atoms) ] for j,_ in enumerate(atoms) ])
    for i,a in enumerate(atoms):
        H[i,i] = Jo[a.symbol]
    return H


def Ai(atoms):
    global par
    Z = np.ones(len(atoms)) * 14.4
    #Z = np.ones(len(atoms))

    A = np.array([ Xo[atoms[i].symbol] + np.sum([ Z[j] * ( C(atoms,i,j) - C(atoms,i,j) ) for j in range(i-1) ]) for i,_ in enumerate(atoms) ])
    
    #A = np.zeros((len(atoms),1))
    #for i,_ in enumerate(atoms):
    #    A[i] = Xo[atoms[i].symbol] + np.sum([ Z[j] * (C(atoms[i], atoms[j]) - C(atoms[i], atoms.s[j])) for j in range(i) ])

    #A = np.array([ Xo[atoms[i].symbol] for i,_ in enumerate(atoms) ])
    #A += np.array([ np.sum([ Z[j] * (C(atoms[i], atoms[j]) - C(atoms[i], atoms.s[j])) for i in range(j,len(atoms)) ]) for j,_ in enumerate(atoms) ])
    #A = np.array([ Xo[atoms[i].symbol] for i,_ in enumerate(atoms) ])
    return A


def shellPositions(atoms):
    a,s = atoms, atoms.s
    rc,rs = atoms.positions, atoms.s.positions
    Z = np.ones(len(atoms)) * 14.4; q = atoms.get_charges() * 14.4
    #Z = np.ones(len(atoms)); q = atoms.get_charges()
    qc, qs = q + Z, -Z

    Ficjc = -1. * np.array([np.sum([ dC(a,i,j) * qc[i]*qc[j] for j,_ in enumerate(atoms) ], axis = 0) for i,_ in enumerate(atoms) ])
    Ficjs = -1. * np.array([np.sum([ dC(a,i,j) * qc[i]*Z[j] for j,_ in enumerate(atoms) ], axis = 0) for i,_ in enumerate(atoms) ])
    Fisjc = -1. * np.array([np.sum([ dC(a,i,j) * Z[i]*qc[j] for j,_ in enumerate(atoms) ], axis = 0) for i,_ in enumerate(atoms) ])
    Fisjs = -1. * np.array([np.sum([ dC(a,i,j) * Z[i]*Z[j] for j,_ in enumerate(atoms) ], axis = 0) for i,_ in enumerate(atoms) ])
    K = np.array([ Ks[atoms[i].symbol] for i,_ in enumerate(atoms) ]).reshape(len(atoms),1)

    interForces = Ficjc - Ficjs - Fisjc + Fisjs
    intraForces = -1. * (rc - rs) * K
    #intraForces = 1. * (rc - rs) * K

    rs = rs + (interForces + intraForces) / K
    return rs


def shellPositionsC(atoms, h: float = 1e-6):
    a,s = atoms, atoms.s
    rc,rs = atoms.positions, atoms.s.positions
    Z = np.ones(len(atoms)); q = atoms.get_charges()
    qc, qs = q + Z, -Z

    K = np.array([ Ks[atoms[i].symbol] for i,_ in enumerate(atoms) ]).reshape(len(atoms),1)

    interForces = -1. * np.array([ shellForces(atoms, i, h) for i,_ in enumerate(atoms) ])
    intraForces = -1. * (rc - rs) * K

    rs = rs + (interForces + intraForces) / K
    return rs


def shellForces(atoms, i, h: float = 1e-6):
    Z = np.ones(len(atoms)); q = atoms.get_charges()
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
