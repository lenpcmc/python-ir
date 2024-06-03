import numpy as np
import os
import matplotlib.pyplot as plt

from rruffIR import *

def main():
    for file in os.listdir("rruffProcessed"):
        wlen, val = rruffIR(f"rruffProcessed/{file}")
        plt.plot(wlen, val)
        plt.show()
    return


if __name__ == "__main__":
    main()
