import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate_plot
import os

import scienceplots
plt.style.use(['science','no-latex'])

from ase.atom import Atom
from ase.visualize.plot import plot_atoms
from chgnet.model import CHGNet, StructOptimizer, CHGNetCalculator
from chgnet.model.dynamics import TrajectoryObserver
from pymatgen.core import Structure
from pymatgen.io.ase import MSONAtoms, AseAtomsAdaptor

from vdos import *
from dyn import *
from rruffIR import rruffIR

im_root = "./resources/images/"
arr_root = "./resources/arrays/"
cif_root = "./resources/cifs/"

rreff = "resources/rruff/processed/data/Wollastonite__R040008-1__Infrared__Infrared_Data_Processed__1001.txt"

def main():
    atoms = buildNumber(f"{cif_root}/alphaCristobalite.cif", 1)
    Dyn = dynamical(atoms)
    vibrations,density = vdos(Dyn)
    v = vibrations[4].reshape(12,3)
    pos = atoms.positions
    path = np.array([ pos + 0.1 * v * np.sin( i/60 * 2. * np.pi ) for i in range(120) ])
    #fig,ax = plt.subplots()
    #ax.axis("off")
    #atomPlot(atoms, ax)
    #plt.show()
    anim = animateAtoms(atoms, path)

    return


def animateAtoms(atoms, path):
    # Init
    fig,ax = plt.subplots(subplot_kw = {"projection": "3d"})

    plt.rcParams.update({
        "figure.facecolor": (1,1,1,0),
        "axes.facecolor": (1,1,1,0),
        "savefig.facecolor": (1,1,1,0),
        })

    bounds = pathBounds(path)
    colors = atomColors(atoms)
    
    xl = np.array([ path[:,i,0] for i,_ in enumerate(atoms) ])
    yl = np.array([ path[:,j,1] for j,_ in enumerate(atoms) ])
    zl = np.array([ path[:,k,2] for k,_ in enumerate(atoms) ])

    # Animation Generator
    def animate(frame):
        while bool(ax.collections): ax.collections[0].remove()

        xp = path[frame,:,0]; yp = path[frame,:,1]; zp = path[frame,:,2]; 
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
        ax.set_zlim(*bounds[2])
        ax.scatter(xs = xp, ys = yp, zs = zp, s = 100, c = colors, depthshade = False)

        for h,(xi,yj,zk) in enumerate(zip(xl,yl,zl)):
            ax.plot( xs = xi[:frame+1], ys = yj[:frame+1], zs = zk[:frame+1], linewidth = 2, c = colors[h], label = atoms[h].symbol )

        return ax

    anim = animate_plot(fig, animate, frames = 120)
    #writer = FFMpegWriter(fps = 30)
    #anim.save("vibratingCristobalite.mp4", writer = writer)
    anim.save("vibratingCristobalite.webp", fps=20, bitrate=5000, savefig_kwargs={"transparent": True, "facecolor": (1,1,1,0)})
    #anim.to_html5_video("vibratingCristobalite.html")
    #plt.show()
    return anim


def pathBounds(path):
    xbounds = np.min(path[...,0]), np.max(path[...,0])
    ybounds = np.min(path[...,1]), np.max(path[...,1])
    zbounds = np.min(path[...,2]), np.max(path[...,2])
    return xbounds, ybounds, zbounds


def atomColors(atoms):
    colorMap = {
            "O": "#9d1c1c",
            "Si": "#78a1b5",
            }
    return np.array([ colorMap[a.symbol] for a in atoms ])


if __name__ == "__main__":
    main()
    pass
