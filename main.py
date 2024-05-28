import numpy as np
import os
import matplotlib.pyplot as plt

from rrufIR import *

def main():
    for file in os.listdir("rrufProcessed"):
        wlen, val = rrufIR(f"rrufProcessed/{file}")
        plt.plot(wlen, val)
        plt.show()
    return


if __name__ == "__main__":
    main()