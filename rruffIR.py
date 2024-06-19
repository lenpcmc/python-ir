import numpy as np
import matplotlib.pyplot as plt
import os

import scienceplots
plt.style.use(['science','no-latex'])

rruff_root = f"resources/rruff/processed/"

def main():
    for file in os.listdir(f"{rruff_root}/data"):
        w,v = rruffIR(f"{rruff_root}/data/{file}")
        w *= 0.03

        fig,ax = plt.subplots()
        ax.plot(w, v)

        descriptor = file[: file.index("__Infrared__") ].replace("__", ' ')
        ax.set_title(f"{descriptor}")
        ax.set_xlabel(r"$\omega$ [THz]")
        ax.xaxis.set_tick_params(which = "minor", bottom = False)
        ax.set_ylabel(r"Absorption [A.U.]")
        ax.set_yticklabels([])
        ax.yaxis.set_tick_params(which = "minor", bottom = False)

        plt.savefig(f"{rruff_root}/images/{descriptor}.png", dpi = 500)
        plt.close()
        #plt.show()

    return


def rruffIR(fname: str) -> (np.ndarray, np.ndarray):
    with open(fname) as infile:
        indata = np.array([ line.strip().split(',') for line in infile if ',' in line and '#' not in line ], dtype = np.float64)
    wlen = indata[:, 0]
    val = indata[:, 1]
    return wlen, val


if __name__ == "__main__":
    main()
