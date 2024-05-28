import numpy as np
import matplotlib.pyplot as plt

def rrufIR(fname: str):
    with open(fname) as infile:
        indata = np.array([ line.strip().split(',') for line in infile if '#' not in line and line != '\n' ], dtype = np.float64)
        wlen = indata[:, 0]
        val = indata[:, 1]
    return wlen, val