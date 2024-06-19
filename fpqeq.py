import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    return


def pqeqEnergy(ric: np.ndarray[float], ris: np.ndarray[float], qi: np.ndarray[float]):
    qsfp: np.ndarray = atoms.get_charges()
    qsdv: np.ndarray = 0.

    Gnew = get_gradient(hess)
    hs = gs
    ht = gt

    GEst2 = 1.99
    for i in nmax:
        Est, hshs, hsht = get_hsh()
        Gest2 = 

        g_h = np.array([gw * hs, gt * ht])
        h_hsh = np.array([hs * hshs, ht * hsht])

        lmin = g_h / h_hsh

        # Line Minimization
        qs += lmin[0] * hs
        qt += lmin[1] * ht

        # Current Electronegativity
        ssum = np.sum(qs)
        tsum = np.sum(qt)

        mu = ssum / tsum

        # Update Atom Charges
        q = qs - mu * qt

        # New Gradient Direction
        Gold = Gnew[...]

        Gnew = get_gradient(hess)
        hs = gs + (Gnew[0]/Gold[0]) * hs
        ht = gt + (Gnew[0]/Gold[0]) * ht

    update_shell_positions()


def get_hsh(Est: float):
    for i,a in enumerate(atoms):
        ity = a.symbol

        hshs[i] = eta[ity] * hs[i]
        hsht[i] = eta[ity] * ht[i]

        qic = q[i] + Zpqeq(ity)
        shell[i] = position[i] + spos[i]

        dr2 = np.sum(spos[i]**2)
        Est += chi[ity] * q[i] + 0.5 * eta[ity] * q[i]**2

        for j,b in enumerate(atoms):
            jty = b.symbol

            qjc = q[j] + Zpqeq(ity)
            shell[j] = position[j] + spos[j]

            Ccicj = hess[j,i] * qic * qjc

            if (isPolarizable(ity)):
                dr = shell[i] - pos[j]
                Csicj = get_coulomb_and_dcoulomb_pqeq(dr, ity, jty)
                Csicj = - Cclmb0_qeq * Csicj * qjc * Zpqeq(ity)

                if (isPolarizable(jty)):
                    dr = shell[i] - shell[j]
                    Csisj = get_coulomb_and_dcoulomb_pqeq(dr, ity, jty)
                    Csisj = Cclmb0_qeq * Csisj * qjc * Zpqeq(ity) * Zpqeq(jty)

            hshs[i] += hess[j,i] * hs(j)
            hsht[i] += hess[j,i] * ht(j)

            Est1 = 0.5 * (Ccicj + Csisj)
            Est += Est1 + Csicj
        
    return Est, hshs, hsht


def get_gradient(hess):
    for i,a in enumerate(atoms):
        gssum = 0.
        gtsum = 0.

        for j,_ in enumerate(atoms):
            gssum += hess[j,i] * qs[j]
            gtsum += hess[j,i] * qt[j]

        ity = a.symbol
        gs[i] = - chi[ity] - eta[ity] * qs[i] - gssum - fpqeq[i]
        gt[i] = - chi[ity] - eta[ity] * qt[i] - gtsum
    
    Gnew = np.array([gs**2, gt**2])
    return Gnew
    


qsfp: np.ndarray = ase.Atoms.get_charges()
qsdv: np.ndarray = 0.

qsfp: np.ndarray = ase.Atoms.get_charges()
qt = 0.
